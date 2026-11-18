#!/usr/bin/env python3
"""串行手机消融，逐轮验收冻结 token 和继承的 KV。"""

import argparse
import collections
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post, _http_post_resilient

BASE = 'http://10.29.131.50:18780'
ARMS = ('full_cpu', 'shared_cc', 'shared_m1', 'shared_recent', 'shared_prior')


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(path, data, timeout=30, retries=4):
    return _http_post_resilient(BASE+path,data,timeout=timeout,retries=retries)


def fit(xs, values):
    xbar, ybar = sum(xs)/len(xs), sum(values)/len(values)
    spread = sum((x-xbar)**2 for x in xs)
    slope = max(0., sum((x-xbar)*(y-ybar) for x,y in zip(xs,values))/spread) if spread else 0.
    intercept = ybar-slope*xbar
    if intercept < 0:
        intercept = 0.
        slope = sum(x*y for x,y in zip(xs,values))/sum(x*x for x in xs)
    return dict(intercept_ms=intercept,slope_ms_per_token=slope)


class Campaign:
    def __init__(self, args, corpus):
        self.args = args
        self.corpus = corpus
        self.output = args.output
        self.fixture_hash = digest(args.fixtures)
        self.model = read(args.model)
        self.build = read(args.build/'manifest.json')['artifacts']
        self.context = math.ceil(max(len(t['request']['frozen_prompt_ids'])+t['D']
                            for c in corpus['cases'] for t in c['turns'])/256)*256
        if self.context > 14336:
            raise ValueError('Frozen cases exceed device admission')
        reserved = set(corpus['calibration_cases'])
        self.calibration = [c for c in corpus['cases'] if c['id'] in reserved]
        if len(self.calibration)!=len(reserved):
            raise ValueError('incomplete held-out calibration cohort')
        cases = [c for c in corpus['cases'] if c['id'] not in reserved]
        self.cases = sorted(cases,key=lambda c:(c['turns'][0]['P'],c['id']))
        self.status = dict(status='running',stage='starting',completed={})
        self.plan = dict(fixtures_sha256=self.fixture_hash, build=self.build, context=self.context,
                         model=self.model, cpu_registry_sha256=digest(args.cpu_registry),
                         conditions=args.conditions,gap_ms=args.gap_ms,
                         cases=[c['id'] for c in self.cases],reserved=sorted(reserved))

    def state(self, **items):
        self.status.update(items)
        save(self.output/'status.json',self.status)
        print({k:v for k,v in items.items()},flush=True)

    def service(self, action, arm, directory):
        cpu = arm=='full_cpu'
        policy = {'shared_prior':'prefill_prior','shared_recent':'recent'}.get(arm)
        command = [sys.executable,str(ROOT/'scripts/run/route_phone.py'),action,
                   '--condition','cpu' if cpu else 'route','--output',str(directory),'--logs-only','--retry-network']
        if action=='start':
            m=self.model
            command += ['--runtime-overlay',str(self.args.build),'--model',m['model_path'],
                        '--model-id',m['model_id'],'--model-sha256',m['model_sha256'],
                        '--context-size',str(self.context),'--cpu-repack','0','--kv-hwm-trim','0']
            command += ['--registry',str(self.args.cpu_registry)] if cpu else ['--shared-weights']
            if policy:
                command += ['--kv-reduce','1','--kv-policy',policy,'--kv-recent-tokens','32',
                            '--kv-tail-tokens','16','--compact-anchor','1']
        subprocess.run(command,cwd=ROOT,check=True)
        if action=='start':
            health = read(directory/'health.json')
            if (health['shared_weights']!= (not cpu) or health['kv_reduce']!= bool(policy) or
                health['context_size']!=self.context or health['frozen_decode_version']!=1 or
                read(directory/'manifest.json')['runtime_overlay']!=self.build):
                raise RuntimeError(f'mismatched phone condition {arm}')
        elif cpu:
            log = (directory/'device/runtime.log').read_text()
            if re.findall(r'backend_ptrs.size\(\) = (\d+)',log)!=['1'] or 'HTP0-REPACK model buffer size' in log:
                raise RuntimeError('Full CPU backend audit failed')

    def verify_renderer(self, case):
        for t in case['turns']:
            body = {k:v for k,v in t['request'].items() if k in ('messages','tools')}
            rendered = request('/apply-template',body)['prompt']
            ids = request('/tokenize',{'content':rendered})['tokens']
            if ids!=t['request']['frozen_prompt_ids']:
                point = next((i for i,(a,b) in enumerate(zip(ids,t['request']['frozen_prompt_ids'])) if a!=b),min(len(ids),len(t['request']['frozen_prompt_ids'])))
                return f"turn {t['index']} phone prompt differs at token {point}"
        return None

    def prepare(self, directory):
        gate = self.output/'renderer-audit.json'
        if gate.exists():
            cleared=read(gate)
        else:
            cleared={}
            for case in self.cases+self.calibration:
                self.state(stage='renderer_audit',case=case['id'])
                cleared[case['id']]=self.verify_renderer(case)
                save(directory/'renderer-progress.json',cleared)
            save(gate,cleared)
        self.cases=[c for c in self.cases if cleared[c['id']] is None]
        if not self.cases:
            raise RuntimeError('no compatible evaluation chains')
        costs = self.output/'route-costs.json'
        if costs.exists():
            return read(costs)
        rows=[]
        calibration_data=self.output/'calibration'
        calibration_data.mkdir(exist_ok=True)
        for case in self.calibration:
            if cleared[case['id']] is not None:
                raise RuntimeError(f"held-out fixture mismatch: {case['id']}")
            warm=case['turns'][1]
            H,P=warm['H'],warm['P']
            ids=warm['request']['frozen_prompt_ids']
            sizes=(24,P) if case['benchmark']=='toolsandbox' else (P,)
            for length in sizes:
                self.state(stage='calibration',case=case['id'],H=H,P=length)
                raw=calibration_data/f"{case['id']}-{length}.json"
                if not raw.exists():
                    probe=_http_post(BASE+'/a2o/calibrate',dict(prefix_ids=ids[:H],input_ids=ids[H:H+length],
                                    decode_evaluations=1,repetitions=1),timeout=2400)
                    save(raw,probe)
                evidence=read(raw)
                row=evidence['rows'][0]
                if row['cosine']<0.99:
                    raise RuntimeError(f'calibration numerical check failed: {raw.name}')
                rows.append(dict(H=H,P=length,**row))
        table=dict(kind='linear_hp_v1',model_id=self.model['model_id']+'-shared-held-out',
                   h_min=min(r['H'] for r in rows),h_max=max(r['H']+r['P'] for r in rows),
                   p_min=min(r['P'] for r in rows),p_max=max(r['P'] for r in rows),routing_p_min=24)
        for key,field,size in [('cpu_restore','cc_restore_ms',lambda r:r['H']),
                               ('cpu_to_htp','hc_prepare_ms',lambda r:r['H']),
                               ('cpu_prefill','cpu_prefill_ms',lambda r:r['P']),
                               ('htp_prefill','htp_prefill_ms',lambda r:r['P']),
                               ('htp_to_cpu','htp_to_cpu_ms',lambda r:r['H']+r['P'])]:
            table[key]=fit([size(row) for row in rows],[row[field] for row in rows])
        save(costs,dict(model=table))
        return dict(model=table)

    def replay(self, arm, case, path):
        request('/a2o/reset',{})
        results=[]
        compact = arm in ('shared_prior','shared_recent')
        routed = arm not in ('full_cpu','shared_cc')
        for i, turn in enumerate(case['turns']):
            self.state(stage='replay',condition=arm,case=case['id'],turn=i,H=turn['H'],P=turn['P'],D=turn['D'])
            prep, gap = None, None
            if i:
                begin=time.monotonic()
                if routed:
                    prep=request('/a2o/tool-start',{},timeout=12)
                time.sleep(max(0,self.args.gap_ms/1000-(time.monotonic()-begin)))
                gap=(time.monotonic()-begin)*1000
            body=dict(turn['request'],max_tokens=turn['D'],expected_decode_tokens=turn['D'])
            if compact:
                body['frozen_kv_keep_tokens']=case['keep_tokens'] if i else self.context
            rid=f"{path.parent.parent.name}:{arm}:{case['id']}:{i}"
            response=_agent_chat_resilient(BASE,body,rid,total_timeout=2400)
            timing=response['timings']
            save(path/f'turn-{i:02d}.json',dict(response=response,prepare=prep,tool_gap_ms=gap,
                                               observed_source_gap_ms=turn['observed_gap_ms']))
            if (response['__verbose']['tokens']!=body['frozen_decode_ids'] or response['__verbose']['stop_type']!='forced' or
                (timing['H_anchor'],timing['P_evaluated'],timing['predicted_n'])!=(turn['H'],turn['P'],turn['D'])):
                raise RuntimeError('frozen shape or decode tokens changed')
            last = results[-1]['kv_reduce'] if results else None
            effective=last['anchor_live_tokens'] if last and last.get('compact_anchor') else turn['H']
            if timing['H_effective']!=effective or (not routed and timing['route']!='cpu'):
                raise RuntimeError('effective history or backend mismatch')
            results.append(timing)
        warm=results[1:]
        summary=dict(benchmark=case['benchmark'],source_model=case['source_model'],id=case['id'],arm=arm,
                     logical_rounds=len(results),warm_rounds=len(warm),
                     post_return_ms=sum(t['total_ms'] for t in warm),
                     inference_ms=sum(t['total_ms'] for t in results),
                     decode_ms_per_token=sum(t['predicted_ms'] for t in warm)/sum(t['predicted_n'] for t in warm),
                     htp_rounds=sum(t['route']=='htp' for t in warm),
                     prior_ranked_tokens=sum((t['kv_reduce'] or {}).get('prior_ranked_tokens',0) for t in warm),
                     max_sampled_rss_mib=max(t['decode_resources'][p]['VmRSS_kib']/1024 for t in results for p in ('before','after')),
                     max_sampled_swap_mib=max(t['decode_resources'][p]['VmSwap_kib']/1024 for t in results for p in ('before','after')),
                     handoff_bytes=sum(t['handoff_bytes'] for t in warm))
        save(path/'summary.json',summary)

    def summarize(self):
        rows=[read(p) for p in sorted(self.output.glob('*/*/summary.json'))]
        baselines={row['id']:row['post_return_ms'] for row in rows if row['arm']=='full_cpu'}
        pairs=collections.defaultdict(list)
        for row in rows:
            if row['id'] in baselines:
                pairs[(row['benchmark'],row['arm'])].append(baselines[row['id']]/row['post_return_ms'])
        aggregates=[dict(benchmark=benchmark,arm=arm,paired_chains=len(values),
                         median_speedup=statistics.median(values))
                    for (benchmark,arm),values in sorted(pairs.items())]
        save(self.output/'summary.json',dict(runs=rows,pairs=aggregates,
             by_condition=dict(collections.Counter(r['arm'] for r in rows))))

    def run(self):
        plan=self.output/'plan.json'
        if plan.exists() and read(plan)!=self.plan:
            raise ValueError('resume requires same build, fixtures, hardware settings and arms')
        save(plan,self.plan)
        epoch=f'{time.time_ns()}'
        calibration=self.output/f'{self.output.name}-cal-{epoch}'
        if not (self.output/'route-costs.json').exists():
            try:
                self.state(stage='starting_calibration')
                self.service('start','shared_m1',calibration)
                costs=self.prepare(calibration)
            finally:
                self.service('stop','shared_m1',calibration)
        else:
            costs=self.prepare(calibration)
        for arm in self.args.conditions:
            pending=[c for c in self.cases if not (self.output/arm/c['id']/'summary.json').exists()]
            if not pending:
                continue
            service=self.output/f'{self.output.name}-{arm}-{epoch}'
            try:
                self.state(stage='starting_condition',condition=arm)
                self.service('start',arm,service)
                if arm not in ('full_cpu','shared_cc'):
                    request('/a2o/costs',costs)
                for case in pending:
                    path=self.output/arm/case['id']
                    path.mkdir(parents=True,exist_ok=True)
                    if any(path.iterdir()):
                        failed=path/f'previous-{time.time_ns()}'
                        saved=list(path.iterdir())
                        failed.mkdir()
                        for item in saved:
                            if item.is_file():
                                item.rename(failed/item.name)
                    self.replay(arm,case,path)
                    self.status['completed'][arm]=self.status['completed'].get(arm,0)+1
                    self.summarize()
                    self.state(stage='case_complete',condition=arm,case=case['id'])
            finally:
                self.service('stop',arm,service)
        self.summarize()
        self.state(status='passed',stage='complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--cpu-registry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=ROOT/'configs/model-default.json')
    parser.add_argument('--conditions',choices=ARMS,nargs='+',default=list(ARMS))
    parser.add_argument('--gap-ms',type=int,default=1000)
    args=parser.parse_args()
    for key in ('fixtures','build','cpu_registry','output','model'):
        setattr(args,key,getattr(args,key).resolve())
    if args.gap_ms<0 or len(set(args.conditions))!=len(args.conditions):
        parser.error('invalid tool gap or duplicate arm')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=Campaign(args,read(args.fixtures))
        try:
            campaign.run()
        except Exception as exc:
            campaign.state(status='failed',error=str(exc))
            traceback.print_exc()
            raise


if __name__=='__main__':
    main()
