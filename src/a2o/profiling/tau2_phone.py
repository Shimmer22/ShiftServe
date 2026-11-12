"""先独立校准，再按真实 G 异步回放；手机模型持锁串行，无生成动作执行。"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from a2o.profiling.bfcl_attention import save
from a2o.profiling.pd_sweep import validate
from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import post

ROOT=Path(__file__).resolve().parents[3]


def interpolate(points,x):
    points=sorted(points)
    if x<=points[0][0]:return points[0][1]
    if x>=points[-1][0]:return points[-1][1]
    for (a,y),(b,z) in zip(points,points[1:]):
        if a<=x<=b:return y+(z-y)*(x-a)/(b-a)
    raise AssertionError(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--requests',type=Path,default=ROOT/'results/20261112-tau2-always-prepare/pilot-six/requests.json')
    p.add_argument('--fixtures',type=Path,default=ROOT/'results/20261112-tau2-always-prepare/pilot-six/tokenizer-fixtures.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/20261112-tau2-phone-pilot')
    p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    guard=(out/'driver.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**kw):
        save(out/'status.json',dict(phase=phase,pid=__import__('os').getpid(),updated_unix=time.time(),**kw))
    previous=ROOT/'results/20261112-tau2-state-curves/status.json'
    while True:
        prev=json.loads(previous.read_text())
        if prev['phase']=='complete':break
        if prev['phase']=='failed':raise RuntimeError('previous device campaign failed; refusing takeover')
        status('waiting_previous_calibration',previous=prev)
        time.sleep(5)
    requests=json.loads(a.requests.read_text());fixtures=json.loads(a.fixtures.read_text())
    assert len(requests)==len(fixtures)==6 and a.rounds>0
    source=ROOT/'results/20261108-bfcl-text-pilot/multi_turn_miss_func_15/t02-s01.response.json'
    choice=json.loads(source.read_text())['choices'][0]
    # 仅独立成本夹具延长参考 token 流；正式 τ² 序列绝不重复、裁剪或补齐。
    stream=(choice['prompt_token_ids']*3)[:16384]
    generated=choice['token_ids'][:64]
    max_h=max(r['H'] for r in requests)
    upper=min(15360,math.ceil(max_h/2048)*2048)
    hs=sorted({4096,8192,upper})
    ps=sorted({31,32,40,64,128,*[r['P'] for r in requests]})
    plan=dict(binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest(),
        fixtures_sha256=hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),context_size=16384,
        cpu_threads=6,htp_host_threads=4,model='Qwen3.5-9B-Q4_0',batch=128,kv_policy='none',pre_move=False,
        H=hs,P=ps,rounds=a.rounds,preparation_policy='always',return_policy='ready-only',
        fixture='成本夹具独立于正式轨迹，固定BFCL token流延长；正式回放用原始完整H/P/D，保存上一输入完整锚点',
        scope='冻结token性能pilot；真实异步准备；外部工作按实测G等待；非手机任务质量或live E2E')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:save(out/'plan.json',plan)
    if (out/'completion.json').exists():return
    endpoint='http://10.29.131.50:18780';service=out/(out.name+'-service')
    def run(cmd,log):
        with (out/log).open('a') as output:
            subprocess.run(cmd,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    def call(folder,key,route,body):
        folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.json')
        if path.exists():
            record=json.loads(path.read_text());assert record['request']==body
            return record['response']
        response=retry_network(lambda:post(endpoint,route,body,1800))
        record=dict(request=body,response=response,endpoint=route)
        raw=folder/(key+'.raw.json')
        if raw.exists():raw=folder/(key+f'.attempt-{len(list(folder.glob(key+"*.raw.json"))):02d}.raw.json')
        save(raw,record)
        if route=='/a2o/stage-sweep':validate(body,response)
        elif route=='/a2o/state-cost-sweep':
            assert response['mode']=='state_cost_sweep_v1'
            assert len(response['rows'])==body['repetitions']+1
            for row in response['rows']:
                assert row['state_verified'] and (row['H'],row['P'])==(len(body['prefix_ids']),len(body['input_ids']))
        elif route=='/a2o/gap-replay':
            assert response['mode']=='always_prepare_gap_replay_v1'
            assert [r['condition'] for r in response['results']]==body['order']
            for row in response['results']:
                assert (row['H'],row['P'],row['D'])==tuple(len(body[k]) for k in ['prefix_ids','input_ids','generated_ids'])
                assert row['actual_gap_ms']>=body['gap_ms']
                if row['condition']=='route':assert row['always_prepare_started']
                assert all(math.isfinite(row[k]) and row[k]>=0 for k in ['prepare_ms','prefill_ms','handoff_ms','decode_ms','post_return_ms','retirement_tail_ms'])
        save(path,record)
        return response
    def sweep(key,h,n,points=()):
        body=dict(prefix_ids=stream[:h],input_ids=stream[h:h+n],
                  generated_ids=generated[:max(points,default=0)],decode_points=list(points),
                  backend_order=['htp','cpu'] if sum(key.encode())%2 else ['cpu','htp'])
        return call(out/'calibration',key,'/a2o/stage-sweep',body)
    try:
        status('starting_service')
        run([sys.executable,str(ROOT/'scripts/run/route_phone.py'),'start','--condition','route',
             '--binary',str(a.binary.resolve()),'--registry',
             str(ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'),
             '--output',str(service),'--context-size','16384','--cpu-threads','6','--replay-cache','--retry-network'],
             'service-launch.log')
        response=post(endpoint,'/v1/chat/completions',dict(messages=[dict(role='user',content='Reply with ready.')],max_tokens=32))
        save(out/'normal-response.raw.json',response)
        assert response['choices'][0]['finish_reason']!='length' and response['choices'][0]['message'].get('content')
        checks=[]
        for r,fixture in zip(requests,fixtures):
            assert r['boundary']==fixture['boundary']
            for key,expected in [('prompt',r['prefix_ids']+r['input_ids']),('generated',r['generated_ids'])]:
                actual=post(endpoint,'/tokenize',dict(content=fixture[key+'_text']))['tokens']
                checks.append(dict(boundary=r['boundary'],sequence=key,tokens=len(actual),passed=actual==expected))
                save(out/'tokenizer-checks.json',checks)
                assert actual==expected,'HF/GGUF token IDs differ'
        # 两个短机制请求覆盖 ready/HTP 和 pending/CPU；不进入性能统计。
        for gap in [0,200]:
            smoke=call(out/'validation',f'gap-{gap}','/a2o/gap-replay',dict(
                prefix_ids=stream[:32],input_ids=stream[32:40],generated_ids=generated[:4],
                gap_ms=gap,order=['cpu','htp','route'],prediction=dict(cpu_prefill_ms=1000,htp_prefill_handoff_ms=.001,kind='mechanism_test_only')))
            route=next(row for row in smoke['results'] if row['condition']=='route')
            assert route['route']==('cpu' if gap==0 else 'htp'),route
        state=[];stage=[]
        for h in hs:
            status('calibrating_state',H=h)
            state_response=call(out/'calibration',f'state-H{h}','/a2o/state-cost-sweep',dict(prefix_ids=stream[:h],input_ids=stream[h:h+40],repetitions=3))
            state.extend(row for row in state_response['rows'] if not row['warmup'])
            points=ps if h==4096 else [64,128]
            for n in points:
                for r in range(3):
                    status('calibrating_prefill',H=h,P=n,round=r)
                    response=sweep(f'r{r:02d}-H{h}-P{n}',h,n)
                    stage.extend(response['rows'])
            status('calibrating_decode',H=h)
            response=sweep(f'decode-H{h}',h,40,[8,32,64])
            save(out/f'decode-H{h}.json',response)
        state_summary=[]
        for h in hs:
            rows=[r for r in state if r['H']==h]
            state_summary.append(dict(H=h,state_bytes=rows[0]['snapshot_bytes'],
                **{key:statistics.median(r[key] for r in rows) for key in ['prepare_ms','snapshot_ms','htp_to_cpu_ms','cpu_restore_ms']}))
        def mean(h,n,backend):
            return statistics.mean(row['prefill_ms'] for row in stage if row['H']==h and row['P']==n and row['backend']==backend)
        # P 精确查表保留阶跃；H 只在实测边界内插值，并用独立 P128 检查。
        sensitivity={backend:[(h,mean(h,64,backend)) for h in hs] for backend in ['cpu','htp']}
        validations=[]
        for h in hs:
            for backend in ['cpu','htp']:
                predicted=mean(4096,128,backend)*mean(h,64,backend)/mean(4096,64,backend)
                actual=mean(h,128,backend)
                validations.append(dict(H=h,backend=backend,predicted_ms=predicted,measured_ms=actual,relative_error=abs(predicted-actual)/actual))
        valid=max(row['relative_error'] for row in validations)<=.15
        costs=dict(kind='exact_p_table_h_sensitivity_v1',state=state_summary,H_sensitivity=sensitivity,
                   validation=validations,validated=valid,formal_results_used=False,
                   P_table=[dict(P=n,cpu_ms=mean(4096,n,'cpu'),htp_ms=mean(4096,n,'htp')) for n in ps])
        save(out/'cost-model.json',costs)
        predictions={}
        for r in requests:
            cp=mean(4096,r['P'],'cpu')*interpolate(sensitivity['cpu'],r['H'])/mean(4096,64,'cpu')
            hp=mean(4096,r['P'],'htp')*interpolate(sensitivity['htp'],r['H'])/mean(4096,64,'htp')
            migration=interpolate([(row['H']+40,row['htp_to_cpu_ms']) for row in state_summary],r['H']+r['P'])
            cpu_restore=interpolate([(row['H'],row['cpu_restore_ms']) for row in state_summary],r['H'])
            predictions[r['boundary']]=dict(kind=costs['kind'],cpu_prefill_ms=cp,cpu_restore_ms=cpu_restore,
                htp_prefill_handoff_ms=hp+migration if valid else cp+1e9,
                estimated_cpu_prefill_ms=cp,estimated_cpu_restore_ms=cpu_restore,
                estimated_htp_prefill_ms=hp,estimated_handoff_ms=migration,validated=valid,
                H_domain=[min(hs),max(hs)],P_exact=True,formal_results_used=False)
        save(out/'predictions.json',predictions)
        all_rows=[]
        for b,r in enumerate(requests):
            for repetition in range(a.rounds):
                modes=['cpu','htp','route'];shift=(b+repetition)%3;order=modes[shift:]+modes[:shift]
                status('measuring',boundary=b,round=repetition,total_boundaries=len(requests),rounds=a.rounds,
                       H=r['H'],P=r['P'],D=r['D'])
                response=call(out/'measurements',f'r{repetition:02d}-b{b:02d}','/a2o/gap-replay',dict(
                    prefix_ids=r['prefix_ids'],input_ids=r['input_ids'],generated_ids=r['generated_ids'],
                    gap_ms=r['gap_ms'],order=order,prediction=predictions[r['boundary']]))
                all_rows.extend(dict(boundary=b,round=repetition,domain=r['domain'],task_id=r['task_id'],**row) for row in response['results'])
                print(json.dumps(dict(boundary=b,round=repetition,results=[dict(condition=x['condition'],route=x['route'],ms=x['post_return_ms']) for x in response['results']])),flush=True)
        with (out/'timings.csv').open('w') as stream_out:
            writer=csv.DictWriter(stream_out,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        totals={mode:sum(row['post_return_ms'] for row in all_rows if row['condition']==mode)/1000 for mode in ['cpu','htp','route']}
        summary=dict(completed=True,boundaries=len(requests),rounds=a.rounds,condition_evaluations=len(all_rows),
            post_return_s=totals,speedup_vs_cpu=totals['cpu']/totals['route'],speedup_vs_htp=totals['htp']/totals['route'],
            route_prefill_cpu=sum(row['condition']=='route' and row['route']=='cpu' for row in all_rows),
            route_prefill_htp=sum(row['condition']=='route' and row['route']=='htp' for row in all_rows),
            preparation_ready=sum(row['condition']=='route' and row['availability']=='ready' for row in all_rows),
            cost_model_validated=valid,scope=plan['scope'])
        save(out/'summary.json',summary)
        run([sys.executable,str(ROOT/'scripts/run/route_phone.py'),'stop','--condition','route','--output',str(service),'--logs-only','--retry-network'],'service-stop.log')
        save(out/'completion.json',summary)
        status('complete',**summary)
    except BaseException as exc:
        status('failed',error=repr(exc),note='原始响应、状态和锁保留；禁止静默换后端或覆盖已有测量。')
        raise


if __name__=='__main__':
    main()
