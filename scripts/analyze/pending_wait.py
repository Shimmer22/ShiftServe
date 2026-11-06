#!/usr/bin/env python3
"""离线分析有限等待配对和Cost-aware触发率，不访问手机。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling.pending_wait import analyze_pending_run, trigger_rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rates-only', action='store_true')
    a = p.parse_args()
    if a.rates_only:
        rows = [json.loads(line) for run in a.runs for line in (run / 'checked-rows.jsonl').read_text().splitlines()]
        result = dict(runs=[str(run.resolve()) for run in a.runs], trigger_rates=trigger_rates(rows))
    else:
        result = [analyze_pending_run(run) for run in a.runs]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(str(a.output))


if __name__ == '__main__':
    main()
