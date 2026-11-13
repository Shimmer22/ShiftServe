"""新模型先准入、后后台扫描；复用既有服务及阶段计时协议。"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

from a2o.profiling.pd_sweep import validate
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import is_network_error

ROOT = Path(__file__).resolve().parents[3]


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def campaign_jobs(config):
    jobs=[]
    for rep in range(config['rounds']):
        block=[('prefill',p,[]) for p in config['P']]
        if rep<config['decode_rounds']:
            block.append(('decode',config['decode_P'],config['D']))
        random.Random(config['seed']+rep).shuffle(block)
        jobs.extend((rep,stage,p,points) for stage,p,points in block)
    return jobs


def paired_order(config,rep,stage,p):
    index=config['P'].index(p) if stage=='prefill' else 0
    return ['cpu','htp'] if (rep+index)%2==0 else ['htp','cpu']


class Campaign:
    def __init__(self, config, out, binary):
        self.cfg, self.out, self.binary = config, out, binary
        self.service = out / (config['name'] + '-service')
        self.transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
        self.identity = None

    def status(self, phase, **fields):
        save(self.out / 'status.json', dict(phase=phase, updated_unix=time.time(), pid=os.getpid(), **fields))
        print(json.dumps(dict(phase=phase, **fields), ensure_ascii=False), flush=True)

    def health(self):
        with urllib.request.urlopen(self.cfg['endpoint'] + '/health', timeout=1800) as stream:
            return json.load(stream)

    def post(self, route, body, label):
        # 网络重试先等待服务互斥锁，确认前一请求退出后才允许重发。
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.cfg['endpoint'] + route, json.dumps(body).encode(),
                                             {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=1800) as stream:
                    return json.load(stream)
            except urllib.error.HTTPError as exc:
                (self.out / (label + '.http-error.txt')).write_bytes(exc.read())
                raise
            except Exception as exc:
                if not is_network_error(exc) or attempt == 7:
                    raise
                self.status('network_retry', request=label, attempt=attempt + 1, error=str(exc))
                time.sleep(15)
                try:
                    if self.identity is not None and self.health() != self.identity:
                        raise RuntimeError('服务身份改变，拒绝继续')
                except Exception as health_error:
                    if not is_network_error(health_error):
                        raise
        raise RuntimeError('unreachable')

    def control(self, action):
        cfg = self.cfg
        args = [sys.executable, str(ROOT / 'scripts/run/route_phone.py'), action, '--condition', 'route',
                '--binary', str(self.binary), '--registry',
                str(ROOT / 'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'),
                '--output', str(self.service), '--context-size', str(cfg['context_size']),
                '--cpu-threads', str(cfg['cpu_threads']), '--model', cfg['model_path'],
                '--model-id', cfg['model_id'], '--model-sha256', cfg['model_sha256'],
                '--opbatch', str(cfg['opbatch']), '--opqueue', str(cfg['opqueue']),
                '--kv-offload', str(int(cfg['kv_offload'])),
                '--flash-attention', str(int(cfg['flash_attention'])), '--replay-cache', '--logs-only', '--retry-network']
        if cfg.get('single_resident'):
            args.append('--single-resident')
        with (self.out / ('service-' + action + '.log')).open('a') as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def check_identity(self):
        identity = self.health()
        cfg = self.cfg
        for key, expected in [('condition', 'route'), ('model', cfg['model_id']),
                              ('cpu_threads', cfg['cpu_threads']), ('context_size', cfg['context_size']),
                              ('htp_kv_offload', cfg['kv_offload']), ('flash_attention', cfg['flash_attention'])]:
            if identity.get(key) != expected:
                raise RuntimeError(f'服务配置不一致：{key}: {identity.get(key)} != {expected}')
        if identity.get('single_resident',False)!=cfg.get('single_resident',False):
            raise RuntimeError('模型驻留配置不一致')
        prior = self.out / 'health.json'
        if prior.exists() and json.loads(prior.read_text()) != identity:
            raise RuntimeError('服务构建身份变化，必须新建结果目录')
        save(prior, identity)
        self.identity = identity

    def fixture(self):
        file = self.out / 'fixture.json'
        if file.exists():
            return json.loads(file.read_text())
        cfg = self.cfg
        prompt = (ROOT / cfg['source_prompt']).read_text()
        response = json.loads((ROOT / cfg['source_response']).read_text())['choices'][0]['text']
        ids = self.post('/tokenize', {'content': prompt}, 'tokenize-prompt')['tokens']
        output = self.post('/tokenize', {'content': response}, 'tokenize-output')['tokens']
        assert len(ids) >= cfg['H'] + max(cfg['P'])
        assert len(output) >= max(cfg['D'])
        value = dict(prompt_ids=ids, output_ids=output, model=cfg['model_id'],
                     source_prompt_sha256=digest(ROOT / cfg['source_prompt']),
                     source_response_sha256=digest(ROOT / cfg['source_response']),
                     scope='旧轨迹原始文本以新模型重新分词；固定token微基准，不是新模型自然生成质量')
        save(file, value)
        return value

    def state_resource_sample(self,label):
        remote=self.transport.remote_root+'/'+self.service.name
        try:
            pid=int(self.transport.call('shell','cat '+remote+'/pid').stdout)
            raw=self.transport.call('shell','cat /proc/'+str(pid)+'/status; cat /proc/'+str(pid)+'/stat').stdout.decode()
            save(self.out/(label+'.resources.json'),dict(pid=pid,raw=raw,
                 scope='整个状态请求的前/后边界，包含bootstrap与预热，不是逐重复窗口'))
        except Exception as exc:
            save(self.out/(label+'.resources.json'),dict(error=repr(exc),available=False))

    def body(self, fixture, h, p, points, order):
        return dict(prefix_ids=fixture['prompt_ids'][:h], input_ids=fixture['prompt_ids'][h:h+p],
                    generated_ids=fixture['output_ids'][:max(points, default=0)],
                    decode_points=points, backend_order=order)

    def measure(self, path, body, **meta):
        if path.exists():
            value = json.loads(path.read_text())
            assert value['request'] == body
            validate(body, value['response'])
            return value['response']
        result = self.post('/a2o/stage-sweep', body, path.stem)
        raw = path.with_suffix('.raw.json')
        save(raw, dict(request=body, response=result, **meta))
        validate(body, result)
        for row in result['rows']:
            assert row['n_threads'] == (self.cfg['cpu_threads'] if row['backend'] == 'cpu' else 4)
        raw.replace(path)
        return result

    def preflight(self):
        self.status('starting_service')
        self.control('start')
        self.check_identity()
        fixture = self.fixture()
        paths=['cc','hh'] if self.cfg.get('single_resident') else ['cc','hh','hc','ch']
        for path in paths:
            self.status('free_generation_json_memory', path=path)
            target = self.out / ('preflight-' + path + '.json')
            if not target.exists():
                result = self.post('/a2o/model-preflight', {'path': path}, 'preflight-' + path)
                save(target.with_suffix('.raw.json'), result)
                assert result['passed'], result
                save(target, result)
        self.status('short_state_smoke')
        self.measure(self.out / 'smoke.json', self.body(fixture, 32, 8, [1, 2], ['cpu', 'htp']))
        self.status('history_and_warmup', H=self.cfg['H'])
        self.measure(self.out / 'warmup.json', self.body(fixture, self.cfg['H'], 40, [1, 8], ['htp', 'cpu']))
        remote = self.transport.remote_root + '/' + self.service.name
        pid = int(self.transport.call('shell', 'cat ' + remote + '/pid').stdout)
        (self.out / 'maps.txt').write_bytes(self.transport.call('shell', 'cat /proc/' + str(pid) + '/maps').stdout)
        save(self.out / 'preflight.json', dict(passed=True, binary_sha256=digest(self.binary),
             model_sha256=self.cfg['model_sha256'], health=self.identity,
             paths=paths, scope='文本正常响应、JSON、同上下文两轮记忆、完整状态与检查点；单驻留模式只验CC/HH，HC/CH另在显式双驻留条件验证'))
        self.status('preflight_passed', service_retained=True, model_lock_retained=True)

    def collect(self):
        admitted = json.loads((self.out / 'preflight.json').read_text())
        assert admitted['passed'] and admitted['binary_sha256'] == digest(self.binary)
        self.check_identity()
        assert admitted['health'] == self.identity
        cfg = self.cfg
        fixture = self.fixture()
        measurements = self.out / 'measurements'
        measurements.mkdir(exist_ok=True)
        jobs = campaign_jobs(cfg)
        schedule=[dict(round=rep,stage=stage,P=p,D=max(points,default=0),
                       decode_points=points,backend_order=paired_order(cfg,rep,stage,p))
                  for rep,stage,p,points in jobs]
        if (self.out/'schedule.json').exists():
            assert json.loads((self.out/'schedule.json').read_text())==schedule,'采样顺序改变，拒绝混合'
        else:
            save(self.out/'schedule.json',schedule)
        for completed, (rep, stage, p, points) in enumerate(jobs):
            order = paired_order(cfg,rep,stage,p)
            self.status('measuring', completed=completed, total=len(jobs), stage=stage, round=rep, P=p, D=max(points, default=0))
            self.measure(measurements / f'r{rep:02d}-{stage}-p{p:03d}.json',
                         self.body(fixture, cfg['H'], p, points, order), round=rep, stage=stage)
        self.summarize(stage_only=True)
        if cfg.get('single_resident'):
            self.status('enabling_dual_state_costs')
            transition=self.post('/a2o/state-residency',{'mode':'dual'},'state-residency')
            save(self.out/'state-residency.json',transition)
            self.identity=self.health()
            assert not self.identity['single_resident']
            save(self.out/'state-health.json',self.identity)
            for path in ['hc','ch']:
                result=self.post('/a2o/model-preflight',{'path':path},'dual-state-preflight-'+path)
                save(self.out/('dual-state-preflight-'+path+'.json'),result)
                assert result['passed'],result
        for h in cfg['state_H']:
            self.status('state_costs', H=h)
            path = measurements / f'state-h{h}.json'
            if not path.exists():
                body = dict(prefix_ids=fixture['prompt_ids'][:h], input_ids=fixture['prompt_ids'][h:h+40],
                            repetitions=cfg['state_repetitions'])
                self.state_resource_sample(path.stem+'-before')
                result = self.post('/a2o/state-cost-sweep', body, path.stem)
                self.state_resource_sample(path.stem+'-after')
                save(path.with_suffix('.raw.json'), dict(request=body, response=result))
                assert result['mode'] == 'state_cost_sweep_v1'
                assert len(result['rows']) == cfg['state_repetitions'] + 1
                for row in result['rows']:
                    assert row['state_verified'] and row['snapshot_bytes'] > 0 and row['handoff_bytes'] > 0
                    assert all(math.isfinite(row[k]) and row[k] >= 0 for k in
                               ['snapshot_ms', 'prepare_ms', 'htp_to_cpu_ms', 'cpu_restore_ms'])
                save(path, dict(request=body, response=result))
        self.summarize()
        self.status('stopping_service', completed=len(jobs), total=len(jobs))
        self.control('stop')
        save(self.out / 'completion.json', dict(completed=True, service_stopped=True, lock_released=True))
        self.status('complete', completed=len(jobs), total=len(jobs))

    def summarize(self,stage_only=False):
        rows = []
        for path in sorted((self.out / 'measurements').glob('r*.json')):
            if '.raw.' in path.name:
                continue
            data = json.loads(path.read_text())
            validate(data['request'], data['response'])
            for row in data['response']['rows']:
                base = dict(round=data['round'], stage=data['stage'], backend=row['backend'], H=row['H'], P=row['P'],
                            n_threads=row['n_threads'], restore_ms=row['restore_ms'], prefill_ms=row['prefill_ms'],
                            numerical_match=data['response']['logit_comparison']['passed'],
                            logits_cosine=data['response']['logit_comparison']['cosine'],
                            single_resident=row.get('single_resident',False),
                            model_context_init_ms=row.get('backend_setup',{}).get('model_context_init_ms',0),
                            backend_warmup_ms=row.get('backend_setup',{}).get('warmup_ms',0),
                            prefill_major_faults=row['resources_after_prefill']['major_faults']-row['resources_before']['major_faults'],
                            decode_run_major_faults=row['resources_after_decode']['major_faults']-row['resources_after_prefill']['major_faults'],
                            swap_after_kib=row['resources_after_decode'].get('VmSwap_kib', 0))
                if data['stage'] == 'prefill':
                    rows.append(dict(base, D=0, tokens=row['P'], elapsed_ms=row['prefill_ms']))
                else:
                    rows.extend(dict(base, D=p['D'], tokens=p['D'], elapsed_ms=p['decode_ms']) for p in row['decode_checkpoints'])
        expected = 2 * (len(self.cfg['P']) * self.cfg['rounds'] + len(self.cfg['D']) * self.cfg['decode_rounds'])
        assert len(rows) == expected, (len(rows), expected)
        with (self.out / 'stage-times.csv').open('w') as stream:
            writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        groups = {}
        for row in rows:
            key = tuple(row[k] for k in ['stage', 'backend', 'H', 'P', 'D'])
            groups.setdefault(key, []).append(row)
        aggregates = []
        for key, samples in sorted(groups.items()):
            values = [r['elapsed_ms'] for r in samples]
            aggregates.append(dict(zip(['stage', 'backend', 'H', 'P', 'D'], key),
                n=len(values), mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                stdev_ms=statistics.stdev(values) if len(values)>1 else 0,
                min_ms=min(values), max_ms=max(values), numerical_passes=sum(r['numerical_match'] for r in samples),
                prefill_major_faults=sum(r['prefill_major_faults'] for r in samples),
                decode_run_major_faults=sum(r['decode_run_major_faults'] for r in samples),
                max_swap_kib=max(r['swap_after_kib'] for r in samples)))
        with (self.out / 'cost-table.csv').open('w') as stream:
            writer = csv.DictWriter(stream, list(aggregates[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(aggregates)
        if stage_only:
            save(self.out/'stage-summary.json',dict(stage_rows=len(rows),stage_costs_complete=True,
                 state_costs_complete=False,scope=self.cfg['scope']))
            return
        state_rows = []
        for path in sorted((self.out / 'measurements').glob('state-h*.json')):
            if '.raw.' not in path.name:
                state_rows.extend(json.loads(path.read_text())['response']['rows'])
        with (self.out / 'state-times.csv').open('w') as stream:
            writer = csv.DictWriter(stream, list(state_rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(state_rows)
        state_groups = []
        for h in self.cfg['state_H']:
            samples = [r for r in state_rows if r['H']==h and not r['warmup']]
            assert len(samples)==self.cfg['state_repetitions']
            value = dict(H=h, n=len(samples))
            for key in ['snapshot_bytes', 'handoff_bytes', 'snapshot_ms', 'prepare_ms',
                        'prepare_import_work_ms', 'prepare_verification_ms', 'htp_to_cpu_ms', 'cpu_restore_ms']:
                value[key+'_mean'] = statistics.mean(r[key] for r in samples)
                value[key+'_median'] = statistics.median(r[key] for r in samples)
            state_groups.append(value)
        with (self.out / 'state-cost-table.csv').open('w') as stream:
            writer = csv.DictWriter(stream, list(state_groups[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(state_groups)
        save(self.out / 'summary.json', dict(stage_rows=len(rows), state_rows=len(state_rows),
             numerical_mismatch_rows=sum(not r['numerical_match'] for r in rows),
             resource_scope='decode_run_major_faults覆盖完整连续decode运行，不是各D检查点的前缀缺页数',
             scope=self.cfg['scope'], filtering='全部原始有效耗时；不按慢点或数值标记自动剔除'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/qwen3-8b-key-costs.json')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--phase', choices=['preflight', 'collect'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = (args.output or ROOT / 'results' / config['name']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = dict(config=config, binary_sha256=digest(args.binary),
                source_prompt_sha256=digest(ROOT / config['source_prompt']),
                source_response_sha256=digest(ROOT / config['source_response']))
    if (out / 'plan.json').exists():
        assert json.loads((out / 'plan.json').read_text()) == plan, '禁止混合构建/模型/配置/输入'
    else:
        save(out / 'plan.json', plan)
    if (out / 'completion.json').exists():
        return
    campaign = Campaign(config, out, args.binary.resolve())
    try:
        getattr(campaign, args.phase)()
    except BaseException:
        campaign.status('failed', phase_failed=args.phase, error=traceback.format_exc())
        # 只停止本轮owner对应服务；保留所有失败日志与原始结果。
        try:
            campaign.control('stop')
            save(out / 'failure-cleanup.json', dict(stopped=True, lock_released=True))
        except Exception:
            save(out / 'failure-cleanup.json', dict(stopped=False, error=traceback.format_exc()))
        raise
