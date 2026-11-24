"""固定 H 的分阶段设备扫描；原始响应先落盘，断线串行重试。"""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import is_network_error, retry_network

ROOT = Path(__file__).resolve().parents[3]


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def validate(request, response):
    if response['mode'] != 'stage_sweep_v1':
        raise ValueError('错误的测量协议')
    assert response['cache']['prefix_tokens'] == len(request['prefix_ids'])
    assert [r['backend'] for r in response['rows']] == request['backend_order']
    for row in response['rows']:
        assert (row['H'], row['P'], row['D']) == tuple(len(request[k]) for k in ('prefix_ids', 'input_ids', 'generated_ids'))
        assert row['state_restore_verified'] and row['finite_logits']
        assert math.isfinite(row['prefill_ms']) and row['prefill_ms'] > 0
        points = row['decode_checkpoints']
        assert [p['D'] for p in points] == request['decode_points']
        times = [p['decode_ms'] for p in points]
        assert all(math.isfinite(t) and t > 0 for t in times)
        assert times == sorted(times)
        if 'cpu_prefill_threads' in request:
            expected = request['cpu_prefill_threads'] if row['backend'] == 'cpu' else 4
            assert row['n_threads'] == 4 and row['n_threads_batch'] == expected
    if 'cpu_prefill_threads' in request:
        assert response['cache']['history_cpu_threads'] == 4 and response['cache']['snapshot_checksum']
    comparison = response['logit_comparison']
    assert math.isfinite(comparison['cosine']) and math.isfinite(comparison['nrmse'])
    assert -1.000001 <= comparison['cosine'] <= 1.000001 and comparison['nrmse'] >= 0
    assert comparison['passed'] == (comparison['cosine'] >= 0.99)


def summarize(out, config):
    rows = []
    cache_sources = {}
    first_backend = Counter()
    candidates = config.get('cpu_prefill_thread_candidates', [])
    snapshot_ids = set()
    for path in sorted((out / 'measurements').glob('r*.json')):
        if '.raw.' in path.name:
            continue
        data = json.loads(path.read_text())
        validate(data['request'], data['response'])
        source = data['response']['cache']['source']
        cache_sources[source] = cache_sources.get(source, 0) + 1
        threads = data['request'].get('cpu_prefill_threads', 0)
        if candidates:
            snapshot_ids.add(data['response']['cache']['snapshot_checksum'])
        if data['stage'] == 'prefill':
            first = data['response']['rows'][0]
            first_backend[first['P'], threads, first['backend']] += 1
        for row in data['response']['rows']:
            base = dict(round=data['round'], stage=data['stage'], backend=row['backend'], H=row['H'],
                        n_threads=row.get('n_threads', 4), n_threads_batch=row.get('n_threads_batch', 4),
                        numerical_match=data['response']['logit_comparison']['passed'],
                        P=row['P'], restore_ms=row['restore_ms'], prefill_ms=row['prefill_ms'])
            if candidates:
                base['cpu_prefill_threads'] = threads
            if data['stage'] == 'prefill':
                rows.append(dict(base, D=0, tokens=row['P'], elapsed_ms=row['prefill_ms']))
            else:
                rows.extend(dict(base, D=p['D'], tokens=p['D'], elapsed_ms=p['decode_ms'])
                            for p in row['decode_checkpoints'])
    count_per_round = len(config['P']) * max(1, len(candidates))
    expected = config['rounds'] * 2 * (count_per_round + len(config['D']))
    assert len(rows) == expected, (len(rows), expected)
    assert cache_sources.get('memory', 0) + cache_sources.get('disk', 0) > 0
    if config.get('balanced_backend_order'):
        assert first_backend == Counter({(p, threads, backend): config['rounds'] // 2
                                        for p in config['P'] for threads in candidates or [0]
                                        for backend in ('cpu', 'htp')})
    if candidates:
        warmup = json.loads((out / 'warmup.json').read_text())
        assert snapshot_ids == {warmup['response']['cache']['snapshot_checksum']}
        assert cache_sources.get('bootstrap', 0) == 0
    with (out / 'stage-times.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(completed=True, stage_rows=len(rows),
        paired_requests=config['rounds'] * (count_per_round + bool(config['D'])), cache_sources=cache_sources,
        H=config['H'], P=config['P'], D=config['D'], scope=config['scope'])
    if candidates:
        medians = {}
        points = []
        for threads in candidates:
            for p in config['P']:
                pairs = {(r['round'], r['backend']):r['elapsed_ms'] / 1000 for r in rows
                         if r['cpu_prefill_threads'] == threads and r['P'] == p}
                cpu = [pairs[(i, 'cpu')] for i in range(config['rounds'])]
                htp = [pairs[(i, 'htp')] for i in range(config['rounds'])]
                medians[threads,p] = statistics.median(cpu)
                points.append(dict(cpu_prefill_threads=threads, P=p, cpu_s=medians[threads,p],
                                   htp_s=statistics.median(htp),
                                   cpu_wins=sum(c < h for c,h in zip(cpu,htp)),
                                   paired_ratio=statistics.median(c / h for c,h in zip(cpu,htp))))
        scores = {threads: math.exp(statistics.mean(
            math.log(medians[threads,p] / medians[4,p]) for p in config['P'])) for threads in candidates}
        summary.update(snapshot_checksum=next(iter(snapshot_ids)), history_cpu_threads=4,
                       cpu_geomean_relative_to_4=scores, selected_cpu_threads=min(scores,key=scores.__getitem__),
                       thread_points=points)
    save(out / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pd-stage-sweep-h4096.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--runtime-overlay', type=Path, help='使用已封版的双后端构建')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    candidates = config.get('cpu_prefill_thread_candidates', [])
    if candidates and (not args.runtime_overlay or config['D'] or config.get('cpu_threads') != 4
                       or 4 not in candidates or len(set(candidates)) != len(candidates)
                       or any(not isinstance(t,int) or t not in range(4,9) for t in candidates)
                       or not config.get('balanced_backend_order') or config['rounds'] % 2):
        parser.error('固定历史的线程试验要求 4–8 线程、包含 4 线程基线、偶数轮、无 Decode 及共享权重构建')
    out = (args.output or ROOT / 'results' / config['name']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(phase, **fields):
        save(out / 'status.json', dict(phase=phase, updated_unix=time.time(), pid=__import__('os').getpid(), **fields))
    def run(command, name):
        with (out / name).open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        source = ROOT / config['source']
        choice = json.loads(source.read_text())['choices'][0]
        h = config['H']
        prompt, generated = choice['prompt_token_ids'], choice['token_ids']
        assert len(prompt) >= h + max(config['P']) and len(generated) >= max(config['D'], default=0)
        plan = dict(config=config, binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    fixture='同一真实prompt前H个token；后续P个token；D使用同一真实输出前缀。属于长度受控微基准，不是原任务原样重放。',
                    timing='纯阶段时间；完整状态恢复、logits检查、快照读写单独计时；decode逐token调用并同步',
                    backend=(f"CPU与HTP均显式加载；Qwen3.5-9B-Q4_0；F16状态；CPU {config['cpu_threads']}线程；HTP主机4线程；batch128；无整理/预搬移"
                             if 'cpu_threads' in config else 'CPU与HTP均显式加载；Qwen3.5-9B-Q4_0；F16状态；4线程；batch128；无整理/预搬移'))
        if args.runtime_overlay:
            if args.binary.resolve() != (args.runtime_overlay / 'agent').resolve():
                raise ValueError('binary must match the sealed overlay')
            plan['build'] = json.loads((args.runtime_overlay / 'manifest.json').read_text())
            plan['model'] = json.loads((ROOT / 'configs/model-default.json').read_text())
            plan['backend'] = 'CPU与HTP显式加载并共享权重，F16状态，CPU及HTP主机均4线程，batch128'
            if candidates:
                plan['backend'] = 'CPU以4线程建立固定历史。仅CPU增量Prefill改变batch线程数，HTP主机4线程，batch128'
                plan['diagnostic_only'] = True
        if config.get('htp_probe_skel'):
            plan['htp_probe_skel_sha256'] = hashlib.sha256((ROOT / config['htp_probe_skel']).read_bytes()).hexdigest()
            plan['diagnostic_only'] = True
        if (out / 'plan.json').exists():
            assert json.loads((out / 'plan.json').read_text()) == plan, '禁止混合不同构建/配置'
        else:
            save(out / 'plan.json', plan)
        if (out / 'summary.json').exists() and json.loads((out / 'status.json').read_text())['phase'] == 'complete':
            return
        service = out / (config['name'] + '-service')
        status('starting_service')
        launch = [sys.executable, str(ROOT / 'scripts/run/route_phone.py'), 'start', '--condition', 'route',
             '--binary', str(args.binary.resolve()),
             '--output', str(service), '--context-size', str(config['context_size']),
             '--cpu-threads', str(config.get('cpu_threads', 4)),
             '--replay-cache', '--retry-network']
        if args.runtime_overlay:
            model = plan['model']
            launch += ['--runtime-overlay', str(args.runtime_overlay.resolve()), '--shared-weights',
                       '--model', model['model_path'], '--model-id', model['model_id'],
                       '--model-sha256', model['model_sha256'], '--kv-hwm-trim', '0',
                       '--verify-kv-state', '0', '--gap-ms', '0', '--memory-ledger']
        else:
            launch += ['--registry',
                       str(ROOT / 'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so')]
        if config.get('htp_probe_skel'):
            launch += ['--probe-skel', str(ROOT / config['htp_probe_skel']), '--htp-profile', '2']
        run(launch, 'service-launch.log')
        trace_transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json') if config.get('htp_probe_skel') else None
        remote_log = trace_transport.remote_root + '/' + service.name + '/runtime.log' if trace_transport else None
        if config.get('replay_cache_source_run'):
            transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
            remote = transport.remote_root + '/' + service.name
            cache_source = transport.remote_root + '/' + config['replay_cache_source_run'] + '/replay-cache'
            cache_target = remote + '/replay-cache'
            q = shlex.quote
            guard = 'test "$(cat ' + q(transport.remote_root + '/model.lock/owner') + ')" = ' + q(service.name)
            command = guard + ' && test -d ' + q(cache_source) + ' && '
            command += 'if [ -e ' + q(cache_target) + ' ]; then test "$(readlink ' + q(cache_target) + ')" = ' + q(cache_source)
            command += '; else ln -s ' + q(cache_source) + ' ' + q(cache_target) + '; fi'
            retry_network(lambda: transport.call('shell', command))
        endpoint = os.environ.get('A2O_PHONE_HTTP_BASE', config['endpoint'])
        def health():
            with urllib.request.urlopen(endpoint + '/health', timeout=2400) as stream:
                return json.load(stream)
        def stable_identity(response):
            return {key: value for key, value in response.items() if key != 'init_ms'}
        identity = stable_identity(retry_network(health))
        assert identity.get('stage_sweep_version') == 1 and identity['condition'] == 'route'
        assert identity['context_size'] == config['context_size']
        assert identity.get('cpu_threads', 4) == config.get('cpu_threads', 4)
        if (out / 'health.json').exists():
            assert stable_identity(json.loads((out / 'health.json').read_text())) == identity
        else:
            save(out / 'health.json', identity)
        def post(body, label, path='/a2o/stage-sweep'):
            while True:
                try:
                    request = urllib.request.Request(endpoint + path,
                        json.dumps(body).encode(), {'Content-Type': 'application/json'})
                    # 正式小P请求通常只需数秒；避免丢失响应后空等40分钟。
                    # 首次完整历史建立仍保留长超时，防止重复昂贵bootstrap。
                    timeout = config.get('request_timeout_s', 120) if label.startswith('r') else 2400
                    with urllib.request.urlopen(request, timeout=timeout) as stream:
                        return json.load(stream)
                except urllib.error.HTTPError as exc:
                    (out / (label + '.http-error.txt')).write_bytes(exc.read())
                    raise
                except Exception as exc:
                    if not is_network_error(exc):
                        raise
                    with (out / 'network-events.jsonl').open('a') as log:
                        log.write(json.dumps(dict(time=time.time(), label=label, error=str(exc))) + '\n')
                    status('waiting_for_network', request=label, error=str(exc))
                    # 健康检查与计算共用服务锁，等待丢失响应的旧请求执行完。
                    assert stable_identity(retry_network(health)) == identity, '重连后服务身份改变'
        def measure(path, body, **meta):
            if path.exists():
                data = json.loads(path.read_text())
                assert data['request'] == body
                validate(body, data['response'])
                return
            offset = int(trace_transport.call('shell', 'wc -c < ' + shlex.quote(remote_log)).stdout) if trace_transport else 0
            result = post(body, path.stem)
            if trace_transport:
                trace = trace_transport.call('shell', 'tail -c +' + str(offset+1) + ' ' + shlex.quote(remote_log)).stdout
                trace_path = path.with_suffix('.profile.log')
                trace_path.write_bytes(trace)
                meta['profile_trace'] = dict(file=trace_path.name, sha256=hashlib.sha256(trace).hexdigest(), start_byte=offset)
            raw = path.with_suffix('.raw.json')
            if raw.exists():
                raw.rename(path.with_suffix(f'.raw.{time.time_ns()}.json'))
            save(raw, dict(request=body, response=result, **meta))
            validate(body, result)
            if 'cpu_threads' in config and 'cpu_prefill_threads' not in body:
                for row in result['rows']:
                    expected_threads = config['cpu_threads'] if row['backend'] == 'cpu' else 4
                    assert row['n_threads'] == row['n_threads_batch'] == expected_threads
            raw.replace(path)
        def body(p, d, order, prefix=None):
            return dict(prefix_ids=prompt[:h] if prefix is None else prefix,
                        input_ids=prompt[h:h+p], generated_ids=generated[:d],
                        decode_points=config['D'] if d else [], backend_order=order)
        if not (out / 'normal-response.json').exists():
            status('normal_response_check')
            result = post(dict(messages=[dict(role='user', content='Reply with exactly OK.')],
                               max_tokens=16), 'normal-response', '/v1/chat/completions')
            save(out / 'normal-response.raw.json', result)
            message = result['choices'][0]['message']
            assert 'ok' in (message.get('content') or '').lower(), '正常响应检查未返回OK'
            assert result['usage']['completion_tokens'] > 0
            assert not result['__verbose']['truncated']
            save(out / 'normal-response.json', result)
        if not (out / 'smoke.json').exists():
            status('mechanism_smoke')
            tiny = dict(prefix_ids=prompt[:32], input_ids=prompt[32:40], generated_ids=generated[:2],
                        decode_points=[1, 2], backend_order=['cpu', 'htp'])
            measure(out / 'smoke.json', tiny, stage='smoke', round=-1)
        if not (out / 'warmup.json').exists():
            status('building_history_cache', H=h, note='只计算一次完整历史，后续内存或磁盘恢复')
            warm = body(config.get('warmup_P', 40), 0 if candidates else 8, ['htp', 'cpu'])
            if candidates:
                warm['cpu_prefill_threads'] = 4
            else:
                warm['decode_points'] = [1, 8]
            measure(out / 'warmup.json', warm, stage='warmup', round=-1)
        measurements = out / 'measurements'
        measurements.mkdir(exist_ok=True)
        completed = 0
        total = config['rounds'] * (len(config['P']) * max(1,len(candidates)) + bool(config['D']))
        for rep in range(config['rounds']):
            jobs = [('prefill', p, 0, threads) for p in config['P'] for threads in candidates or [0]]
            if config['D']:
                jobs.append(('decode', config['decode_P'], max(config['D']), 0))
            random.Random(config['seed'] + rep).shuffle(jobs)
            for index, (stage, p, d, threads) in enumerate(jobs):
                order_index = config['P'].index(p) if config.get('balanced_backend_order') and stage == 'prefill' else index
                if candidates:
                    order_index += candidates.index(threads)
                order = ['cpu', 'htp'] if (rep + order_index) % 2 == 0 else ['htp', 'cpu']
                request = body(p, d, order)
                suffix = f'-cpu{threads}' if candidates else ''
                path = measurements / f'r{rep:02d}-{stage}-p{p:03d}{suffix}.json'
                thread_status = {'cpu_prefill_threads': threads} if candidates else {}
                request.update(thread_status)
                status('measuring', completed=completed, total=total, round=rep, stage=stage, P=p, D=d, **thread_status)
                measure(path, request, round=rep, stage=stage)
                completed += 1
        summarize(out, config)
        status('stopping_service', completed=completed, total=total)
        if args.runtime_overlay:
            run([sys.executable, str(ROOT / 'scripts/run/route_phone.py'), 'stop',
                 '--condition', 'route', '--output', str(service), '--logs-only', '--retry-network'],
                'service-stop.log')
            status('complete', completed=completed, total=total)
            return
        # 仅停止自己的已验证服务，不拉取大状态缓存；缓存继续保留在手机。
        transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
        remote = transport.remote_root + '/' + service.name
        q = shlex.quote
        def ssh(command):
            return retry_network(lambda: transport.call('shell', command)).stdout.decode().strip()
        owner = 'test "$(cat ' + q(transport.remote_root + '/model.lock/owner') + ')" = ' + q(service.name)
        pid = int(ssh('cat ' + q(remote + '/pid')))
        ticks = ssh("awk '{print $22}' /proc/" + str(pid) + '/stat')
        guard = owner + ' && test "$(awk \'{print $22}\' /proc/' + str(pid) + '/stat)" = ' + q(ticks)
        guard += ' && tr "\\000" " " < /proc/' + str(pid) + '/cmdline | /system/bin/grep -F ' + q(remote + '/service')
        ssh(guard + ' && kill ' + str(pid))
        for _ in range(30):
            if ssh('if kill -0 ' + str(pid) + ' 2>/dev/null; then echo alive; else echo stopped; fi') == 'stopped':
                break
            time.sleep(1)
        else:
            raise RuntimeError('服务未退出，保留模型锁')
        retry_network(lambda: transport.call('pull', remote + '/runtime.log', str(out / 'runtime.log')))
        transport.lock_owner = service.name
        retry_network(transport.release_lock)
        subprocess.run(transport.prefix + ['termux-wake-unlock'], capture_output=True, check=True)
        status('complete', completed=completed, total=total)
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
