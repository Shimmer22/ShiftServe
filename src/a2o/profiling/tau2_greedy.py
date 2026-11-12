"""复用独立成本表，显式接受预测误差；原始六边界、全token、三轮串行比较。"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from a2o.profiling.bfcl_attention import save
from a2o.profiling.tau2_phone import interpolate
from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import post

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/20261112-tau2-greedy-return')
    parser.add_argument('--rounds',type=int,default=3)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    guard=(out/'driver.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=ROOT/'results/20261112-tau2-always-prepare/pilot-six'
    previous=ROOT/'results/20261112-tau2-phone-pilot'
    requests=json.loads((source/'requests.json').read_text())
    fixtures=json.loads((source/'tokenizer-fixtures.json').read_text())
    model=json.loads((previous/'cost-model.json').read_text())
    old_predictions=json.loads((previous/'predictions.json').read_text())
    assert json.loads((previous/'completion.json').read_text())['completed']
    assert len(requests)==len(fixtures)==6 and a.rounds>0
    predictions={}
    for r in requests:
        old=old_predictions[str(r['boundary'])]
        predictions[r['boundary']]=dict(kind='greedy_return_reused_prediction_v1',
            cpu_prefill_ms=old['estimated_cpu_prefill_ms'],cpu_restore_ms=old['estimated_cpu_restore_ms'],
            htp_prefill_handoff_ms=old['estimated_htp_prefill_ms']+old['estimated_handoff_ms'],
            prepare_ms=interpolate([(x['H'],x['prepare_ms']) for x in model['state']],r['H']),
            validated=model['validated'],prediction_error_accepted=True,formal_results_used=False)
    plan=dict(binary_sha256=sha(a.binary),requests_sha256=sha(source/'requests.json'),
        fixtures_sha256=sha(source/'tokenizer-fixtures.json'),cost_model_sha256=sha(previous/'cost-model.json'),
        prior_predictions_sha256=sha(previous/'predictions.json'),driver_sha256=sha(Path(__file__)),
        return_policy='greedy',preparation_policy='always',allow_unvalidated_prediction=True,
        prediction_acceptance='用户明确接受当前预测可靠性；保留validated=false，不强制CPU',
        context_size=16384,cpu_threads=6,htp_host_threads=4,model='Qwen3.5-9B-Q4_0',
        batch=128,kv_policy='none',pre_move=False,rounds=a.rounds,boundaries=len(requests),
        remaining_estimate='max(independently_calibrated_prepare_ms - elapsed_prepare_ms, 0)',
        admission='remaining_prepare_ms + htp_prefill_ms + handoff_ms < cpu_restore_ms + cpu_prefill_ms',
        wait_budget='cpu_restore_ms + cpu_prefill_ms - htp_prefill_ms - handoff_ms',
        scope='冻结token；真实异步准备和等待计时；按实测G等待外部事件；不测完整任务质量或live E2E')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:save(out/'plan.json',plan)
    if (out/'completion.json').exists():return
    save(out/'predictions.json',predictions)
    service=out/(out.name+'-service');endpoint='http://10.29.131.50:18780'
    def status(phase,**kw):
        save(out/'status.json',dict(phase=phase,pid=__import__('os').getpid(),updated_unix=time.time(),**kw))
    def run(cmd,log):
        with (out/log).open('a') as output:
            subprocess.run(cmd,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    def measure(folder,key,body):
        folder.mkdir(exist_ok=True)
        path=folder/(key+'.json')
        if path.exists():
            d=json.loads(path.read_text());assert d['request']==body
            return d['response']
        response=retry_network(lambda:post(endpoint,'/a2o/gap-replay',body,1800))
        record=dict(request=body,response=response)
        raw=folder/(key+'.raw.json')
        if raw.exists():raw=folder/(key+f'.attempt-{len(list(folder.glob(key+"*.raw.json"))):02d}.raw.json')
        save(raw,record)
        assert response['mode']=='always_prepare_gap_replay_v2'
        assert [r['condition'] for r in response['results']]==body['order']
        for row in response['results']:
            assert (row['H'],row['P'],row['D'])==tuple(len(body[k]) for k in ['prefix_ids','input_ids','generated_ids'])
            for name in ['actual_gap_ms','prepare_ms','wait_ms','restore_ms','prefill_ms','handoff_ms','decode_ms','post_return_ms','retirement_tail_ms']:
                assert math.isfinite(row[name]) and row[name]>=0
            assert row['post_return_ms']+1e-6>=sum(row[k] for k in ['wait_ms','restore_ms','prefill_ms','handoff_ms','decode_ms'])
            if row['condition']=='route':
                assert row['always_prepare_started'] and row['return_policy']=='greedy'
                dec=row['decision'];assert dec['continue_prepare']==(dec['continue_cost_ms']<dec['cpu_cost_ms'])
                if row['route']=='htp':assert dec['continue_prepare'] and row['availability']=='ready'
        save(path,record)
        return response
    try:
        status('starting_service')
        run([sys.executable,str(ROOT/'scripts/run/route_phone.py'),'start','--condition','route',
            '--binary',str(a.binary.resolve()),'--registry',
            str(ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'),
            '--output',str(service),'--context-size','16384','--cpu-threads','6','--replay-cache','--retry-network'],
            'service-launch.log')
        if not (out/'normal-response.json').exists():
            normal=post(endpoint,'/v1/chat/completions',dict(messages=[dict(role='user',content='Reply with ready.')],max_tokens=32))
            save(out/'normal-response.raw.json',normal)
            assert normal['choices'][0]['finish_reason']!='length' and normal['choices'][0]['message'].get('content')
            save(out/'normal-response.json',normal)
        checks=[]
        status('validating_tokens')
        for r,f in zip(requests,fixtures):
            assert r['boundary']==f['boundary']
            for name,expected in [('prompt',r['prefix_ids']+r['input_ids']),('generated',r['generated_ids'])]:
                actual=post(endpoint,'/tokenize',dict(content=f[name+'_text']))['tokens']
                checks.append(dict(boundary=r['boundary'],sequence=name,tokens=len(actual),passed=actual==expected))
                save(out/'tokenizer-checks.json',checks);assert actual==expected
        status('validating_greedy_branches')
        r=requests[0]
        smoke_cases=[('continue',0,1000,.001,100,'htp','greedy_continue_prepare'),
                     ('abandon',0,1,1000,100,'cpu','greedy_abandon_prepare'),
                     ('ready',200,1000,.001,100,'htp','greedy_ready_htp'),
                     ('timeout',0,1.001,1,0,'cpu','greedy_wait_timeout')]
        for key,gap,cpu,hc,prepare,expected,reason in smoke_cases:
            response=measure(out/'validation',key,dict(prefix_ids=r['prefix_ids'][:32],
                input_ids=r['prefix_ids'][32:40],generated_ids=r['generated_ids'][:4],
                gap_ms=gap,order=['cpu','htp','route'],return_policy='greedy',allow_unvalidated_prediction=True,
                prediction=dict(cpu_prefill_ms=cpu,cpu_restore_ms=0,htp_prefill_handoff_ms=hc,
                                prepare_ms=prepare,validated=False,kind='mechanism_test_only')))
            route=next(x for x in response['results'] if x['condition']=='route')
            assert route['route']==expected and route['route_reason']==reason,route
        save(out/'validation-complete.json',dict(passed=True,token_sequences=len(checks),branches=[c[0] for c in smoke_cases]))
        rows=[]
        for b,r in enumerate(requests):
            for repetition in range(a.rounds):
                modes=['cpu','htp','route'];shift=(b+repetition)%3
                status('measuring',boundary=b,round=repetition,completed=len(rows),total=6*a.rounds*3,
                       H=r['H'],P=r['P'],D=r['D'],return_policy='greedy')
                response=measure(out/'measurements',f'r{repetition:02d}-b{b:02d}',dict(
                    prefix_ids=r['prefix_ids'],input_ids=r['input_ids'],generated_ids=r['generated_ids'],
                    gap_ms=r['gap_ms'],order=modes[shift:]+modes[:shift],return_policy='greedy',
                    allow_unvalidated_prediction=True,prediction=predictions[r['boundary']]))
                for row in response['results']:
                    rows.append(dict(boundary=b,round=repetition,domain=r['domain'],**row))
                print(json.dumps(dict(boundary=b,round=repetition,
                    results=[{k:x[k] for k in ['condition','route','route_reason','wait_ms','post_return_ms']} for x in response['results']])),flush=True)
        with (out/'timings.csv').open('w') as output:
            writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        totals={mode:sum(r['post_return_ms'] for r in rows if r['condition']==mode)/1000 for mode in ['cpu','htp','route']}
        route=[r for r in rows if r['condition']=='route']
        summary=dict(completed=True,return_policy='greedy',boundaries=6,rounds=a.rounds,condition_evaluations=len(rows),
            post_return_s=totals,speedup_vs_cpu=totals['cpu']/totals['route'],speedup_vs_htp=totals['htp']/totals['route'],
            htp_routes=sum(r['route']=='htp' for r in route),cpu_routes=sum(r['route']=='cpu' for r in route),
            wait_attempts=sum(r['wait_ms']>0 for r in route),wait_timeouts=sum(r['route_reason']=='greedy_wait_timeout' for r in route),
            wait_s=sum(r['wait_ms'] for r in route)/1000,prediction_validated=model['validated'],
            allow_unvalidated_prediction=True,scope=plan['scope'])
        save(out/'summary.json',summary)
        status('stopping_service',completed=len(rows),total=len(rows))
        run([sys.executable,str(ROOT/'scripts/run/route_phone.py'),'stop','--condition','route',
             '--output',str(service),'--logs-only','--retry-network'],'service-stop.log')
        save(out/'completion.json',summary);status('complete',**summary)
    except BaseException as exc:
        status('failed',error=repr(exc),note='原始日志、状态与锁保留；设备失败不伪装为CPU策略回退。')
        raise


if __name__=='__main__':
    main()
