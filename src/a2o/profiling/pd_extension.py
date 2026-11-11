"""跨批次追加固定H扫描；等待原任务释放模型锁，禁止并行占用手机。"""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from a2o.profiling.pd_sweep import ROOT, save


def merge_runs(base, extra, out):
    plans = [json.loads((p / 'plan.json').read_text()) for p in [base, extra]]
    for key in ['binary_sha256', 'source_sha256']:
        assert plans[0][key] == plans[1][key], '构建或token源不一致'
    for key in ['H', 'rounds', 'cpu_threads', 'context_size', 'D']:
        assert plans[0]['config'][key] == plans[1]['config'][key], key
    assert not set(plans[0]['config']['P']) & set(plans[1]['config']['P']), 'P重复'
    config = dict(plans[0]['config'])
    config.update(name=out.name, P=sorted(plans[0]['config']['P'] + plans[1]['config']['P']))
    expected = list(range(24, 65)) + [66, 68, 72, 80, 96, 128]
    assert config['P'] == expected and config['rounds'] == 10 and config['cpu_threads'] == 6
    rows, counts, rounds, manifest = [], collections.Counter(), {}, []
    for folder in [base, extra]:
        assert json.loads((folder / 'status.json').read_text())['phase'] == 'complete'
        path = folder / 'stage-times.csv'
        with path.open() as stream:
            data = list(csv.DictReader(stream))
        for row in data:
            assert row['stage'] == 'prefill' and int(row['H']) == config['H']
            threads = 6 if row['backend'] == 'cpu' else 4
            assert int(row['n_threads']) == int(row['n_threads_batch']) == threads
            key = (row['backend'], int(row['P']))
            counts[key] += 1
            seen = rounds.setdefault(key, set())
            assert int(row['round']) not in seen
            seen.add(int(row['round']))
            rows.append(dict(row, source_run=folder.name))
        manifest.append(dict(directory=str(folder), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(data)))
    assert len(counts) == 94 and set(counts.values()) == {10} and len(rows) == 940
    assert all(value == set(range(10)) for value in rounds.values())
    with (out / 'stage-times.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    save(out / 'plan.json', dict(config=config, binary_sha256=plans[0]['binary_sha256'],
        source_sha256=plans[0]['source_sha256'], input_runs=manifest,
        scope='原批次与随后追加批次合并，批次内P乱序；各行source_run保留采集批次，全部有效耗时统一统计'))
    save(out / 'summary.json', dict(completed=True, H=4096, P=expected, rounds=10,
        paired_requests=470, stage_rows=940, cpu_threads=6, htp_host_threads=4, input_runs=manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(args.config.read_text())
    extra = ROOT / 'results' / config['name']
    def status(phase, **fields):
        save(out / 'status.json', dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), total=470, **fields))
    try:
        save(out / 'extension-plan.json', dict(base=str(args.base.resolve()),
             supplement=str(extra), config=str(args.config.resolve()),
             P=list(range(24, 65)) + [66, 68, 72, 80, 96, 128],
             repetitions=10, cpu_threads=6, htp_host_threads=4,
             scope='原扫描继续完成，随后串行补测五点；两批次各自乱序，保留逐行来源'))
        while True:
            current = json.loads((args.base / 'status.json').read_text())
            if current['phase'] == 'failed':
                raise RuntimeError('原扫描失败，保留现场，不启动追加采集：' + str(current))
            if current['phase'] == 'complete':
                break
            status('waiting_for_base', completed=current.get('completed', 0), base_status=current)
            time.sleep(15)
        command = [sys.executable, str(ROOT / 'scripts/collect/pd_stage_sweep.py'),
                   '--config', str(args.config.resolve()), '--binary', str(args.binary.resolve())]
        with (out / 'supplement.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                progress = json.loads((extra / 'status.json').read_text()) if (extra / 'status.json').exists() else {}
                status('supplement_running', completed=420 + progress.get('completed', 0), supplement_status=progress)
                time.sleep(15)
            assert child.returncode == 0, '追加采集失败，查看supplement.log'
        status('merging', completed=470)
        merge_runs(args.base.resolve(), extra, out)
        status('complete', completed=470)
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
