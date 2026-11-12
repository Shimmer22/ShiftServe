#!/usr/bin/env python3
"""手机串行补测状态成本及 H 敏感性；复用运行时并完整保留响应。"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling.bfcl_attention import save
from a2o.profiling.pd_sweep import validate
from a2o.workloads.toolsandbox import post
from a2o.runtime.network_retry import retry_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/20261112-tau2-state-curves')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = ROOT / 'results/20261108-bfcl-text-pilot/multi_turn_miss_func_15/t02-s01.response.json'
    choice = json.loads(source.read_text())['choices'][0]
    prompt, generated = choice['prompt_token_ids'], choice['token_ids']
    endpoint = 'http://10.29.131.50:18780'
    service = out / (out.name + '-service')
    jobs = []
    # 每个 H 连续完成，避免单份内存缓存在多个历史长度之间反复重建。
    for h in [512, 1024, 2048, 4096, 6144, 7168]:
        jobs.append((f'state-H{h}', '/a2o/state-cost-sweep', dict(
            prefix_ids=prompt[:h], input_ids=prompt[h:h+40], repetitions=5)))
        if h in [1024, 4096, 6144]:
            shapes = [('prefill', p, []) for p in [31, 40, 64, 128]] + [('decode', 40, [8, 32, 64])]
            for r in range(3):
                order = shapes.copy()
                random.Random(20261112 + h + r).shuffle(order)
                for i, (stage, p, points) in enumerate(order):
                    backend_order = ['cpu', 'htp'] if (r+i) % 2 == 0 else ['htp', 'cpu']
                    jobs.append((f'r{r:02d}-H{h}-{stage}-P{p}', '/a2o/stage-sweep', dict(
                        prefix_ids=prompt[:h], input_ids=prompt[h:h+p],
                        generated_ids=generated[:max(points, default=0)],
                        decode_points=points, backend_order=backend_order)))
    plan = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model='Qwen3.5-9B-Q4_0', context=8192, cpu_threads=6, htp_host_threads=4,
        state_H=[512, 1024, 2048, 4096, 6144, 7168], state_repetitions=5,
        sensitivity_H=[1024,4096,6144], P=[31,40,64,128], D=[8,32,64], rounds=3,
        kv_policy='none', pre_move=False, jobs=[x[0] for x in jobs])
    if (out / 'plan.json').exists():
        assert json.loads((out / 'plan.json').read_text()) == plan
    else:
        save(out / 'plan.json', plan)
    def status(phase, **kw):
        save(out / 'status.json', dict(phase=phase, updated_unix=time.time(), **kw))
    def run(command, log):
        with (out / log).open('a') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        status('starting_service')
        run([sys.executable, str(ROOT/'scripts/run/route_phone.py'), 'start', '--condition', 'route',
            '--binary', str(args.binary.resolve()), '--registry',
            str(ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'),
            '--output', str(service), '--context-size', '8192', '--cpu-threads', '6',
            '--replay-cache', '--retry-network'], 'service-launch.log')
        health = post(endpoint, '/a2o/reset', {})
        save(out/'reset.json', health)
        if not (out/'normal-response.json').exists():
            body = dict(messages=[dict(role='user',content='Reply with the word ready.')], max_tokens=32)
            response = post(endpoint, '/v1/chat/completions', body)
            save(out/'normal-response.raw.json',response)
            assert response['choices'][0]['finish_reason'] != 'length'
            assert response['choices'][0]['message'].get('content', '').strip()
            save(out/'normal-response.json',response)
        folder = out/'measurements'
        folder.mkdir(exist_ok=True)
        for i, (key, route, body) in enumerate(jobs):
            path = folder/(key+'.json')
            status('measuring', completed=i, total=len(jobs), current=key)
            if path.exists():
                continue
            response = retry_network(lambda:post(endpoint,route,body,1800))
            record = dict(request=body,response=response,endpoint=route)
            save(folder/(key+'.raw.json'),record)
            if route.endswith('stage-sweep'):
                validate(body,response)
            else:
                assert response['mode']=='state_cost_sweep_v1'
                assert len(response['rows'])==6
                for row in response['rows']:
                    assert row['state_verified'] and (row['H'],row['P'])==(len(body['prefix_ids']),40)
                    for name in ['snapshot_ms','prepare_ms','htp_to_cpu_ms','cpu_restore_ms']:
                        assert row[name]>0
            save(path,record)
            print(key,flush=True)
        save(out/'completion.json',dict(completed=True,jobs=len(jobs)))
        run([sys.executable,str(ROOT/'scripts/run/route_phone.py'),'stop','--condition','route',
            '--output',str(service),'--logs-only','--retry-network'],'service-stop.log')
        status('complete',completed=len(jobs),total=len(jobs))
    except BaseException as exc:
        status('failed',error=repr(exc),note='保留原始输出与锁，核对现场后恢复。')
        raise


if __name__=='__main__':
    main()
