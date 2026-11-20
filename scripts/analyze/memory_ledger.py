#!/usr/bin/env python3
"""从手机运行日志生成端侧阶段账本及 Swap 增长区间。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.profiling.memory_ledger import parse_runtime_log, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', type=Path, help='含 manifest.json 与 device/runtime.log 的服务目录')
    args = parser.parse_args()
    manifest = json.loads((args.service / 'manifest.json').read_text())
    with (args.service / 'device/runtime.log').open() as log:
        metadata, entries, buffers = parse_runtime_log(log)
    if manifest['memory_ledger'] is not True or metadata.get('memory_ledger_version') != 1:
        raise ValueError('服务未启用阶段内存账本')
    trim = manifest['kv_hwm_trim']
    if metadata['kv_hwm_trim'] is not trim or any(e['kv_hwm_trim'] is not trim for e in entries):
        raise ValueError('日志与启动清单的 kv_hwm_trim 不一致')
    samples = [entry['sample'] for entry in entries]
    points = summarize(samples)
    increases = [
        dict(previous_stage=previous['stage'], stage=current['stage'],
             swap_growth_kib=current['since_previous']['VmSwap_kib'],
             swap_kib=current['VmSwap_kib'], monotonic_ms=current['monotonic_ms'])
        for previous, current in zip(points, points[1:])
        if (current['since_previous']['VmSwap_kib'] or 0) > 0
    ]
    report = dict(metadata=metadata, condition=manifest['condition'], model_buffers=buffers,
                  swap_growth_intervals=increases, checkpoints=points, raw_samples=samples)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
