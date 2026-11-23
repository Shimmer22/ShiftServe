#!/usr/bin/env python3
"""以两轮反向的线程组顺序测量手机 Prefill。"""

import argparse
from collections import Counter
import csv
import fcntl
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.run.t6_prefill_followup import BUILD, CONFIG, MAIN, SHA, read, save, snapshot_phone, verify_sweep_fixture
from a2o.runtime.network_retry import is_network_error

OUT = MAIN / 'prefill-motivation-h4096-8b-thread-audit'
EXTENSION = MAIN / 'prefill-motivation-h4096-8b-thread-randomized'
PRIOR_RUN = 'prefill-motivation-h4096-8b-service'
OTHER_DRIVER_LOCK = MAIN / 't6-horizon-mobile-smoke/driver.lock'
POINTS = (16, 24, 32, 48, 64, 96)
THREADS = (4, 6, 8)
ROUNDS_PER_BLOCK = 2


def wait_for_device(out):
    while True:
        try:
            owner = next((line.removeprefix('owner=') for line in snapshot_phone().splitlines()
                          if line.startswith('owner=')), None)
        except (OSError, subprocess.SubprocessError) as exc:
            if not is_network_error(exc):
                raise
            owner = 'waiting_for_ssh'
        if owner is None or owner.startswith(out.name + '-'):
            return
        save(out / 'status.json', dict(phase='waiting_for_phone', owner=owner))
        time.sleep(20)


def sweep(config, directory):
    if (directory / 'status.json').exists() and read(directory / 'status.json')['phase'] == 'complete':
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'config.json'
    if path.exists() and read(path) != config:
        raise ValueError(f'sweep configuration changed: {directory.name}')
    save(path, config)
    wait_for_device(directory.parent)
    save(directory.parent / 'status.json', dict(phase='running', run=directory.name, cpu_threads=config['cpu_threads']))
    with (directory / 'driver.log').open('a') as log:
        subprocess.run([sys.executable, '-u', 'scripts/run/termux_http_forward.py', '--port', '28793', '--',
                        sys.executable, '-u', 'scripts/collect/pd_stage_sweep.py', '--config', str(path),
                        '--output', str(directory), '--binary', str(BUILD / 'agent'),
                        '--runtime-overlay', str(BUILD)], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def summarize(out, thread_counts):
    points = []
    for threads in thread_counts:
        files = [f for block in (1, 2) for f in (out / f'block{block}-cpu{threads}/measurements').glob('r*.json')
                 if '.raw.' not in f.name]
        if len(files) != len(POINTS) * ROUNDS_PER_BLOCK * 2:
            raise ValueError(f'incomplete CPU {threads}-thread data')
        paired = {p: [] for p in POINTS}
        lead = Counter()
        for f in files:
            data = read(f)
            p = len(data['request']['input_ids'])
            if len(data['request']['prefix_ids']) != 4096:
                raise ValueError(f'unexpected H: {f}')
            rows = {row['backend']: row for row in data['response']['rows']}
            if rows['cpu']['n_threads_batch'] != threads or rows['htp']['n_threads_batch'] != 4:
                raise ValueError(f'unexpected thread count: {f}')
            lead[p, data['request']['backend_order'][0]] += 1
            paired[p].append(tuple(rows[backend]['prefill_ms'] / 1000 for backend in ('cpu', 'htp')))
        for p, times in paired.items():
            if len(times) != 4 or any(lead[p, arm] != 2 for arm in ('cpu', 'htp')):
                raise ValueError(f'unbalanced backend order at {threads=} {p=}')
            ratios = [cpu / htp for cpu, htp in times]
            points.append(dict(cpu_threads=threads, P=p,
                               cpu_median_s=statistics.median(cpu for cpu, _ in times),
                               htp_median_s=statistics.median(htp for _, htp in times),
                               cpu_htp_ratio_median=statistics.median(ratios),
                               cpu_wins=sum(r < 1 for r in ratios), pairs=len(times)))
    with (out / 'thread-points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    by_threads = {threads: {r['P']: r for r in points if r['cpu_threads'] == threads} for threads in thread_counts}
    scores = {threads: math.exp(statistics.mean(
        math.log(by_threads[threads][p]['cpu_median_s'] / by_threads[4][p]['cpu_median_s']) for p in POINTS))
              for threads in thread_counts}
    selected = min(thread_counts, key=scores.__getitem__)
    comparison = by_threads[selected].values()
    cpu_points = [r['P'] for r in comparison if r['cpu_wins'] >= 3 and r['cpu_htp_ratio_median'] <= .95]
    htp_points = [r['P'] for r in comparison if r['cpu_wins'] <= 1 and r['cpu_htp_ratio_median'] >= 1.05]
    crossing = len(cpu_points) >= 2 and len([p for p in htp_points if p > min(cpu_points)]) >= 2
    result = dict(selected_cpu_threads=selected, cpu_geomean_relative_to_4=scores,
                  cpu_advantage_points=cpu_points, htp_advantage_points=htp_points, crossing=crossing,
                  selection='各P的CPU中位耗时对4线程做等权几何平均。交叉门槛要求至少两个CPU占优点，且至少两个更长的HTP占优点。占优要求至少3/4对获胜且配对耗时比中位数相差至少5%。')
    save(out / 'diagnosis.json', result)
    return result


def campaign(out, thread_counts, orders):
    original = read(CONFIG)
    if read(BUILD / 'manifest.json')['git_commit'] != SHA:
        raise ValueError('unexpected sealed build')
    verify_sweep_fixture()
    for block, order in enumerate(orders, start=1):
        for threads in order:
            directory = out / f'block{block}-cpu{threads}'
            config = dict(original, name=f'{out.name}-{directory.name}', P=list(POINTS), warmup_P=32,
                          rounds=ROUNDS_PER_BLOCK, seed=original['seed'] + block * 100, cpu_threads=threads)
            if threads == 4:
                config['replay_cache_source_run'] = PRIOR_RUN
            elif out == EXTENSION and threads in THREADS:
                config['replay_cache_source_run'] = f'{OUT.name}-block1-cpu{threads}-service'
            elif block == 2:
                config['replay_cache_source_run'] = f'{out.name}-block1-cpu{threads}-service'
            sweep(config, directory)
    save(out / 'status.json', dict(phase='diagnosis_complete', **summarize(out, thread_counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extend', action='store_true', help='追加随机轮转的 4–8 线程对照')
    args = parser.parse_args()
    out = EXTENSION if args.extend else OUT
    thread_counts = tuple(range(4, 9)) if args.extend else THREADS
    order = list(thread_counts)
    if args.extend:
        random.Random(2026112302).shuffle(order)
    orders = (order, order[::-1])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(out / 'status.json', dict(phase='waiting_for_first_thread_sweep' if args.extend else 'waiting_for_horizon_smoke',
                                      pid=os.getpid(), H=4096, P=POINTS, cpu_threads=thread_counts, orders=orders,
                                      planned_pairs=len(POINTS) * len(thread_counts) * ROUNDS_PER_BLOCK * 2))
        upstream_lock = OUT / 'driver.lock' if args.extend else OTHER_DRIVER_LOCK
        with upstream_lock.open('a') as upstream:
            fcntl.flock(upstream, fcntl.LOCK_EX)
            try:
                if args.extend and read(OUT / 'status.json')['phase'] != 'diagnosis_complete':
                    raise RuntimeError('first thread sweep did not complete')
                campaign(out, thread_counts, orders)
            except Exception as exc:
                save(out / 'status.json', dict(phase='failed', error=str(exc)))
                raise


if __name__ == '__main__':
    main()
