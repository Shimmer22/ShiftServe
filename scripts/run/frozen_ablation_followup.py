#!/usr/bin/env python3
"""跟踪固定批次，验收原始响应并在手机释放后串行补测。"""

import argparse
import collections
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import statistics
import select
import shlex
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from a2o.runtime.device_transport import DeviceTransport
from scripts.run.frozen_ablation import BASE, Campaign, _agent_chat_resilient, _http_post, digest, read, request, save


def audit(directory):
    plan = read(directory/'plan.json')
    fixtures = directory/'fixtures.json'
    if digest(fixtures)!=plan['fixtures_sha256']:
        raise ValueError('primary fixture digest changed')
    corpus = read(fixtures)
    compatible = read(directory/'renderer-audit.json')
    cases = {c['id']:c for c in corpus['cases'] if c['id'] in plan['cases'] and compatible[c['id']] is None}
    costs = read(directory/'route-costs.json')['model']
    observations, complete = [], {}
    for path in sorted(directory.glob('*/*/summary.json')):
        row=read(path)
        arm, case = row['arm'], cases[row['id']]
        if arm not in plan['conditions']:
            raise ValueError(f'unexpected arm {arm}')
        complete[(arm,case['id'])] = row
        effective=0
        for i,t in enumerate(case['turns']):
            raw=read(path.parent/f'turn-{i:02d}.json')
            resp=raw['response']
            m=resp['timings']
            if (resp['__verbose']['tokens']!=t['request']['frozen_decode_ids'] or
                resp['__verbose']['stop_type']!='forced' or m['H_effective']!=effective or
                (m['H_anchor'],m['P_evaluated'],m['predicted_n']) != (t['H'],t['P'],t['D'])):
                raise ValueError(f'raw validation failed: {arm}/{case["id"]}/{i}')
            if arm in ('shared_cc','full_cpu') and m['route']!='cpu':
                raise ValueError(f'wrong baseline backend: {arm}/{case["id"]}/{i}')
            kv=m['kv_reduce'] or {}
            effective=kv.get('anchor_live_tokens',len(t['request']['frozen_prompt_ids'])-4)
            if i:
                resources=m['decode_resources']
                observations.append(dict(arm=arm,id=case['id'],benchmark=case['benchmark'],index=i,
                    H=t['H'],H_eff=m['H_effective'],P=t['P'],D=t['D'],route=m['route'],
                    availability=m['availability'], observed_gap_ms=t['observed_gap_ms'],
                    device_gap_ms=raw['tool_gap_ms'],post_return_ms=m['total_ms'],
                    prefill_ms=m['prompt_ms'],wait_ms=m['wait_ms'],restore_ms=m['restore_ms'],
                    handoff_bytes=m['handoff_bytes'],prior_ranked_tokens=kv.get('prior_ranked_tokens',0),
                    peak_sampled_swap_kib=max(resources[p]['VmSwap_kib'] for p in ('before','after')),
                    major_faults=resources['after']['major_faults']-resources['before']['major_faults'],
                    extrapolated=(m['H_effective']<costs['h_min'] or m['H_effective']+t['P']>costs['h_max'] or
                                  not costs['p_min']<=t['P']<=costs['p_max'])))
    groups=[]
    for bench in sorted({c['benchmark'] for c in cases.values()}):
        chain_ids={c['id'] for c in cases.values() if c['benchmark']==bench}
        source_ids={c['id']:c['source'] for c in cases.values() if c['id'] in chain_ids}
        for ref,arm in zip(plan['conditions'],plan['conditions'][1:]):
            paired=[x for x in chain_ids if (ref,x) in complete and (arm,x) in complete]
            if not paired:
                continue
            tasks=collections.defaultdict(list)
            for x in paired:
                tasks[source_ids[x]].append(complete[(ref,x)]['post_return_ms']/complete[(arm,x)]['post_return_ms'])
            groups.append(dict(benchmark=bench, reference=ref, condition=arm, paired_chains=len(paired),
                independent_source_tasks=len(tasks),
                median_source_speedup=statistics.median(statistics.median(v) for v in tasks.values())))
    estimate = estimate_time(plan,cases,complete)
    out=dict(expected_chains=len(cases)*len(plan['conditions']),verified_chains=len(complete),
             calibration_domain={k:costs[k] for k in ('h_min','h_max','p_min','p_max')},
             planned_outside_P=sum(not costs['p_min']<=t['P']<=costs['p_max'] for c in cases.values() for t in c['turns'][1:]),
             planned_outside_H=sum(not costs['h_min']<=t['H'] or t['H']+t['P']>costs['h_max']
                                   for c in cases.values() for t in c['turns'][1:]),
             planned_warm_turns=sum(len(c['turns'])-1 for c in cases.values()),estimate=estimate,
             by_source_task=groups,observations=observations,
             scope={'gap':'primary fixed gap; recorded source gap retained separately',
                    'decode_horizon':'primary uses the frozen trace D; production prediction requires separate evaluation',
                    'memory':'process samples around decode, DMA and migration peaks outside this measure'})
    save(directory/'systems-audit.json',out)
    return out,cases,costs


def estimate_time(plan,cases,complete):
    long_cpu=[(r['inference_ms']-r['post_return_ms'])/cases[name]['turns'][0]['P']
              for (arm,name),r in complete.items() if arm=='full_cpu' and cases[name]['turns'][0]['P']>=3000]
    cold_rate=statistics.median(long_cpu) if long_cpu else 80.
    estimated=0.
    for arm in plan['conditions']:
        ratios=[r['inference_ms']/complete[('full_cpu',name)]['inference_ms']
                for (a,name),r in complete.items() if a==arm and ('full_cpu',name) in complete
                    and cases[name]['turns'][0]['P']>=3000]
        factor=statistics.median(ratios) if ratios else 1.
        for name,c in cases.items():
            if (arm,name) in complete:
                continue
            cpu=complete.get(('full_cpu',name))
            ms=(cpu['inference_ms'] if cpu else c['turns'][0]['P']*cold_rate +
                sum(105*t['P']+220*t['D']+1500 for t in c['turns'][1:]))
            estimated+=ms*factor
    now=datetime.now().astimezone()
    return dict(as_of=now.isoformat(timespec='seconds'),work_remaining_hours=estimated/3_600_000,
                eta=(now+timedelta(milliseconds=estimated)).isoformat(timespec='minutes'),
                range=[(now+timedelta(milliseconds=estimated*k)).isoformat(timespec='minutes') for k in (.82,1.24)],
                note='按已完成长链实测与同链 CPU 对照估算，含当前轮全量余量，未计外部断网')


def phone_owner():
    device=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
    return device.call('shell',f'test ! -f {device.remote_root}/model.lock/owner || cat {device.remote_root}/model.lock/owner').stdout.decode().strip()


def sample_memory(primary, stopped):
    device=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
    prefix=shlex.quote(primary.name+'-')
    command=f"""
    root={shlex.quote(device.remote_root)}
    prefix={prefix}
    while :; do
        if ! owner=$(cat "$root/model.lock/owner" 2>/dev/null); then sleep 2; continue; fi
        case "$owner" in "$prefix"*) ;; *) sleep 2; continue;; esac
        pid=$(cat "$root/$owner/pid" 2>/dev/null)
        if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
            printf '%s %s %s ' "$(date +%s)" "$owner" "$pid"
            awk '/^VmRSS:|^VmSwap:|^RssAnon:|^RssFile:|^RssShmem:/ {{printf "%s%s ",$1,$2}}' "/proc/$pid/status"
            awk '/^MemAvailable:|^SwapFree:/ {{printf "%s%s ",$1,$2}}' /proc/meminfo
            printf '\n'
        fi
        sleep 2
    done
    """
    with (primary/'memory-samples.jsonl').open('a',buffering=1) as out:
        while not stopped.is_set():
            with subprocess.Popen(device.shell_command(command),stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True) as ssh:
                try:
                    while not stopped.is_set():
                        if not select.select([ssh.stdout],[],[],3)[0]:
                            continue
                        line=ssh.stdout.readline()
                        if not line:
                            break
                        parts=line.split()
                        if len(parts)>=3 and parts[0].isdigit() and parts[2].isdigit():
                            values={k:int(v) for k,v in (x.split(':',1) for x in parts[3:] if ':' in x) if v.isdigit()}
                            out.write(json.dumps(dict(device_unix_s=int(parts[0]),owner=parts[1],
                                                      pid=int(parts[2]),**values))+'\n')
                finally:
                    if ssh.poll() is None:
                        ssh.terminate()
            stopped.wait(3)


def launch_resume(args):
    cmd=[sys.executable,'-u',str(ROOT/'scripts/run/frozen_ablation.py'),
         '--fixtures',str(args.primary/'fixtures.json'),'--build',str(args.primary_build),
         '--cpu-registry',str(args.cpu_registry),'--model',str(args.model),'--output',str(args.primary)]
    with (args.primary/'driver.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (args.primary/'driver.pid').write_text(str(child.pid)+'\n')
    return child


def still_running(primary):
    pid=int((primary/'driver.pid').read_text())
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return 'frozen_ablation.py' in result.stdout and str(primary) in result.stdout


def probe(args,cases,costs):
    out=args.primary/'supplement'
    out.mkdir(exist_ok=True)
    selected=[c for c in cases.values() if c['benchmark'].startswith('tau2') and c['turns'][1]['observed_gap_ms'] is not None]
    if not selected:
        raise RuntimeError('no independently recorded short tool gap')
    long=min(selected,key=lambda c:(abs(c['turns'][1]['H']-7500),abs(c['turns'][1]['P']-60),c['id']))
    short=min((c for c in selected if c['id']!=long['id']),
              key=lambda c:(abs(c['turns'][1]['H']-5000),abs(c['turns'][1]['P']-60),c['id']))
    large=max(selected,key=lambda c: max(t['P'] for t in c['turns'][1:]))
    selected=[short,long]
    # Reuse the primary fixture and context size. Only the supplemental runtime changes.
    service_args=argparse.Namespace(output=out,fixtures=args.primary/'fixtures.json',build=args.probe_build,
        model=args.model,cpu_registry=args.cpu_registry,conditions=['shared_m1'],gap_ms=1000)
    runner=Campaign(service_args,read(service_args.fixtures))
    service=out/f'{args.primary.name}-supplement-{time.time_ns()}'
    save(out/'plan.json',dict(cancellation_cases=[c['id'] for c in selected], extrapolation_case=large['id'],
         build=runner.build,context=runner.context,comparison='same-binary forced CC, observed G, synchronous join vs retired worker'))
    try:
        runner.service('start','shared_m1',service)
        request('/a2o/costs',{'model':costs})
        results=[]
        for c_index,case in enumerate(selected):
            seed, turn = case['turns'][:2]
            for mode in (('join','retire') if c_index==0 else ('retire','join')):
                directory=out/case['id']/mode
                directory.mkdir(parents=True,exist_ok=True)
                if (directory/'warm.json').exists():
                    results.append(read(directory/'warm.json')['timings'])
                    continue
                runner.state(stage='cancellation_probe',case=case['id'],retirement=mode)
                request('/a2o/reset',{})
                reply=_agent_chat_resilient(BASE,seed['request'],f'{service.name}:{case["id"]}:{mode}:seed')
                save(directory/'seed.json',reply)
                payload=dict(turn['request'],frozen_tool_gap_ms=turn['observed_gap_ms'],
                             frozen_prepare_retirement=mode,frozen_force_cc_after_prepare=True)
                reply=_agent_chat_resilient(BASE,payload,f'{service.name}:{case["id"]}:{mode}:warm')
                m=reply['timings']
                if (m['route']!='cpu' or m['route_reason']!='frozen_probe_cc' or m['H_anchor']!=turn['H']
                    or reply['__verbose']['tokens']!=turn['request']['frozen_decode_ids']):
                    raise RuntimeError('supplemental cancellation fixture mismatch')
                save(directory/'warm.json',reply)
                results.append(m)
        target=max(large['turns'][1:],key=lambda t:t['P'])
        extra=out/'large-p.json'
        if target['P']>costs['p_max'] and not extra.exists():
            ids=target['request']['frozen_prompt_ids']
            h=target['H']
            runner.state(stage='out_of_domain_probe',case=large['id'],H=h,P=target['P'])
            result=_http_post(BASE+'/a2o/calibrate',dict(prefix_ids=ids[:h],input_ids=ids[h:],
                              decode_evaluations=1,repetitions=1),timeout=2400)
            save(extra,result)
        save(out/'summary.json',dict(cancellation=results,large_p=read(extra) if extra.exists() else None))
    finally:
        runner.service('stop','shared_m1',service)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--primary-build',type=Path)
    p.add_argument('--probe-build',type=Path)
    p.add_argument('--cpu-registry',type=Path)
    p.add_argument('--model',type=Path,default=ROOT/'configs/model-qwen3-4b.json')
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    for key in ('primary','primary_build','probe_build','cpu_registry','model'):
        value=getattr(args,key)
        if value:
            setattr(args,key,value.resolve())
    if args.audit_only:
        result,_,_=audit(args.primary)
        print(json.dumps({k:v for k,v in result.items() if k not in ('observations','by_source_task')},ensure_ascii=False,indent=2))
        return
    if not all((args.primary_build,args.probe_build,args.cpu_registry)):
        p.error('supervisor requires primary/probe builds and a CPU registry')
    state=args.primary/'supervisor-status.json'
    lock=(args.primary/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    retries=read(state).get('retries',0) if state.exists() else 0
    child=None
    stopped=threading.Event()
    monitor=threading.Thread(target=sample_memory,args=(args.primary,stopped),daemon=True)
    monitor.start()
    try:
        while True:
            report,cases,costs=audit(args.primary)
            status=read(args.primary/'status.json')
            save(state,dict(stage='watching_primary',retries=retries,primary=status,estimate=report['estimate']))
            if status['status']=='passed':
                if report['verified_chains']!=report['expected_chains']:
                    raise RuntimeError('primary exit status passed but raw audit is incomplete')
                break
            if not still_running(args.primary):
                # Only retry safe transport errors on an unlocked phone. Inference and
                # validation failures remain visible for diagnosis.
                error=status.get('error','')
                transient=status['status']=='running' or any(x in error.lower() for x in ('connection refused','connection reset','network retries','<urlopen error'))
                if not transient or retries>=2:
                    raise RuntimeError(f'Primary stopped and requires diagnosis: {status}')
                try:
                    if not phone_owner():
                        retries+=1
                        child=launch_resume(args)
                        save(state,dict(stage='retrying_primary',retries=retries,pid=child.pid))
                except (OSError, subprocess.SubprocessError):
                    time.sleep(20)
            if child:
                child.poll()
            time.sleep(60)
        while True:
            try:
                if not phone_owner():
                    break
            except (OSError, subprocess.SubprocessError):
                pass
            time.sleep(20)
        save(state,dict(stage='running_supplement',retries=retries))
        probe(args,cases,costs)
        save(state,dict(stage='passed',retries=retries))
    except Exception as exc:
        save(state,dict(stage='failed',error=str(exc),retries=retries))
        traceback.print_exc()
        raise
    finally:
        stopped.set()
        monitor.join(timeout=5)
        lock.close()


if __name__=='__main__':
    main()
