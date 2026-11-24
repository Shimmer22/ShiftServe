#!/usr/bin/env python3
"""在 wins 串行比较同一历史快照上的共享、独立 CPU 权重。"""

import argparse
import csv
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling.pd_sweep import save
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network

MAIN = ROOT / 'results/mainline-eval'
UPSTREAM = MAIN / 'prefill-fixed-history-thread-audit'
OUT = MAIN / 'prefill-weight-ab-h4096-8b'
BUILD = MAIN / 'prefill-weight-ab-build'
POINTS = (16, 24, 32, 48, 64, 96)
ORDER = ('shared', 'independent', 'independent', 'shared')
PREFIX_LENGTH = 4096


@dataclass(frozen=True)
class Sweep:
    output: Path
    build: Path
    points: tuple
    order: tuple
    repetitions: int

    @property
    def total(self):
        return len(self.points) * len(self.order) * self.repetitions


def read(path):
    return json.loads(path.read_text())


def snapshot_source(transport, checksum, prefix):
    source = transport.remote_root + '/' + UPSTREAM.name + '-service/replay-cache'
    quoted = shlex.quote(source)
    command = ('for f in ' + quoted + '/*.json; do '
               'if grep -Fq ' + shlex.quote('"state_checksum":"' + checksum + '"') +
               ' "$f"; then printf "%s\\n" "$f"; fi; done')
    matches = retry_network(lambda: transport.call('shell', command)).stdout.decode().splitlines()
    if len(matches) != 1:
        raise RuntimeError('expected exactly one recorded H=4096 cache file')
    path = matches[0]
    meta = json.loads(retry_network(lambda: transport.call('shell', 'cat ' + shlex.quote(path))).stdout)
    if meta['metadata']['prefix_ids'] != prefix or meta['state_checksum'] != checksum:
        raise RuntimeError('wrong frozen history snapshot')
    if meta['metadata']['context'] != 8192 or meta['metadata']['batch'] != 128:
        raise RuntimeError('wrong cache shape')
    return path


def stage_snapshot(transport, attempt, source):
    remote = transport.remote_root
    cache = Path(remote) / attempt.name / 'replay-cache'
    guard = 'test "$(cat ' + shlex.quote(remote + '/model.lock/owner') + ')" = ' + shlex.quote(attempt.name)
    commands = (shlex.join(['mkdir','-p',str(cache)]),
                shlex.join(['ln','-sfn',source,str(cache/'prefill-ab.json')]),
                shlex.join(['ln','-sfn',source+'.state',str(cache/'prefill-ab.json.state')]))
    retry_network(lambda: transport.call('shell', guard + ' && ' + ' && '.join(commands)))


def measure(directory, label, request, arm):
    file = directory / f'{label}.json'
    if file.exists():
        recorded = read(file)
        if recorded['request'] != request:
            raise RuntimeError('existing A/B request differs')
        result = recorded['response']
    else:
        base = os.environ['A2O_PHONE_HTTP_BASE']
        req = Request(base + '/a2o/prefill-weight-ab', json.dumps(request).encode(),
                      {'Content-Type':'application/json'})
        def send():
            with urlopen(req, timeout=900) as response:
                return json.load(response)
        result = retry_network(send)
        save(file.with_suffix('.raw.json'), dict(request=request,response=result))
    expected_backend = 'htp' if arm == 'htp' else 'cpu'
    if (result['mode'] != 'cpu_prefill_ab_v1' or result['shared_weights'] is not (arm == 'shared')
            or result.get('backend','cpu') != expected_backend
            or result['snapshot_checksum'] != request['snapshot_checksum']
            or result['H'] != PREFIX_LENGTH or result['P'] != len(request['input_ids'])
            or result['n_threads'] != 4 or result['n_threads_batch'] != 4
            or not result['restore_verified'] or not math.isfinite(result['prefill_ms'])
            or result['prefill_ms'] <= 0):
        raise RuntimeError(f'A/B sample validation failed: {label}')
    if not file.exists():
        file.with_suffix('.raw.json').replace(file)
    return result


def run_block(sweep, transport, index, arm, prompt, source, checksum, model, completed):
    shared = arm == 'shared'
    directory = sweep.output / f'block-{index}-{arm}'
    directory.mkdir(exist_ok=True)
    service = directory / f'{sweep.output.name}-{directory.name}-service'
    condition = {'shared':'route','independent':'cpu','htp':'htp'}[arm]
    common = ['--condition',condition,'--output',str(service),'--logs-only','--retry-network']
    launch = ['--runtime-overlay',str(sweep.build),'--model',model['model_path'],
              '--model-id',model['model_id'],'--model-sha256',model['model_sha256'],'--context-size','8192',
              '--cpu-threads','4','--cpu-repack','0','--kv-hwm-trim','0','--verify-kv-state','0','--gap-ms','0',
              '--replay-cache','--memory-ledger']
    if shared:
        launch.append('--shared-weights')
    entry = [sys.executable,'scripts/run/route_phone.py']
    def request(p):
        return dict(prefix_ids=prompt[:PREFIX_LENGTH],input_ids=prompt[PREFIX_LENGTH:PREFIX_LENGTH+p],
                    snapshot_checksum=checksum)
    save(sweep.output/'status.json',dict(phase='starting_service',block=index,arm=arm,completed=completed,total=sweep.total))
    rows = []
    try:
        with (directory/'service-launch.log').open('a') as log:
            subprocess.run([*entry,'start',*common,*launch],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        health = read(service/'health.json')
        if (health['build_id'] != read(sweep.build/'manifest.json')['build_id'] or health['shared_weights'] is not shared
                or health['context_size'] != 8192 or health['cpu_threads'] != 4 or health['htp_kv_offload']):
            raise RuntimeError('A/B service settings mismatch')
        stage_snapshot(transport,service,source)
        measure(directory,'warmup',request(32),arm)
        for rep in range(sweep.repetitions):
            order = list(sweep.points)
            random.Random(2026112304 + (index//2)*sweep.repetitions + rep).shuffle(order)
            for p in order:
                save(sweep.output/'status.json',dict(phase='measuring',block=index,arm=arm,round=rep,P=p,
                                            completed=completed+len(rows),total=sweep.total,snapshot_checksum=checksum))
                result = measure(directory,f'r{rep}-p{p:03d}',request(p),arm)
                rows.append(dict(block=index,pair=index//2,round=rep,arm=arm,P=p,
                                 prefill_ms=result['prefill_ms'],top_token=result['top_token'],
                                 swap_kib=result['resources_after']['VmSwap_kib']))
    finally:
        with (directory/'service-stop.log').open('a') as log:
            subprocess.run([*entry,'stop',*common],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    runtime = (service/'device/runtime.log').read_text()
    if ('HTP0-REPACK model buffer size' in runtime) is not (arm != 'independent'):
        raise RuntimeError(f'A/B weight placement mismatch in block {index}')
    if arm == 'independent' and 'CPU_Mapped model buffer size' not in runtime:
        raise RuntimeError('independent weights missing CPU mapped buffer')
    return rows


def summarize(sweep, rows, checksum):
    arm_a, arm_b = sweep.order[:2]
    pairs = {}
    for row in rows:
        pairs.setdefault((row['pair'],row['round'],row['P']),{})[row['arm']] = row
    if (len(rows)!=sweep.total or len(pairs)!=sweep.total//2
            or any(set(arms)!={arm_a,arm_b} for arms in pairs.values())):
        raise RuntimeError('A/B results incomplete')
    summary = []
    for p in sweep.points:
        trials = [arms for key,arms in pairs.items() if key[2]==p]
        summary.append(dict(P=p,pairs=len(trials), **{
             f'{arm}_s':statistics.median(a[arm]['prefill_ms'] for a in trials)/1000 for arm in (arm_a,arm_b)},
             **{f'{arm_a}_over_{arm_b}':statistics.median(a[arm_a]['prefill_ms']/a[arm_b]['prefill_ms'] for a in trials),
                f'{arm_a}_wins':sum(a[arm_a]['prefill_ms']<a[arm_b]['prefill_ms'] for a in trials)},
             top_token_matches=sum(a[arm_a]['top_token']==a[arm_b]['top_token'] for a in trials)))
    with (sweep.output/'stage-times.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(sweep.output/'summary.json',dict(total_pairs=len(pairs),snapshot_checksum=checksum,points=summary,
                                 peak_sampled_swap_kib=max(x['swap_kib'] for x in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-crossing', action='store_true')
    args = parser.parse_args()
    sweep = (Sweep(MAIN/'prefill-crossing-confirmation-h4096-8b',MAIN/'prefill-crossing-confirmation-build',
                   (16,24,32),('independent','htp','htp','independent'),3)
             if args.confirm_crossing else Sweep(OUT,BUILD,POINTS,ORDER,2))
    out = sweep.output
    transport=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(out/'status.json',dict(phase='waiting_for_thread_probe',completed=0,total=sweep.total,pid=os.getpid()))
        with (UPSTREAM/'driver.lock').open('a') as upstream:
            fcntl.flock(upstream,fcntl.LOCK_EX)
            try:
                if read(UPSTREAM/'status.json')['phase']!='complete':
                    raise RuntimeError('shared snapshot upstream has not completed')
                warmup=read(UPSTREAM/'warmup.json')
                checksum=warmup['response']['cache']['snapshot_checksum']
                config=read(UPSTREAM/'config.json')
                model=read(UPSTREAM/'plan.json')['model']
                prompt=read(ROOT/config['source'])['choices'][0]['prompt_token_ids']
                source=snapshot_source(transport,checksum,prompt[:PREFIX_LENGTH])
                rows=[]
                for index,arm in enumerate(sweep.order):
                    rows.extend(run_block(sweep,transport,index,arm,prompt,source,checksum,model,len(rows)))
                summarize(sweep,rows,checksum)
                save(out/'status.json',dict(phase='complete',completed=len(rows),total=sweep.total,snapshot_checksum=checksum))
            except Exception as exc:
                save(out/'status.json',dict(phase='failed',error=str(exc)))
                raise


if __name__=='__main__':
    main()
