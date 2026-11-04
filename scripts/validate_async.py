#!/usr/bin/env python3
"""最小串行验证：全 CPU 对照 + 屏障控制的异步准备；不比较耗时。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runner import default_device, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    suite = ROOT / 'results' / f'{stamp}-async-mechanism-validation'
    suite.mkdir(parents=True)
    config = json.loads((ROOT / 'configs/async-mechanism-smoke.json').read_text())
    control_config = suite / 'cpu-control.json'
    control_config.write_text(json.dumps({**config, 'async_smoke': False}, indent=2))
    summary = {'scope': '基础机制验证；手机有并行竞争，所有耗时不用于性能结论；无真实 Tool、无图像输入',
               'timing_valid_for_performance': False, 'device': args.device,
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'passed': False, 'runs': []}
    def save():
        (suite / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    save()
    try:
        results = []
        for label, path in [('cpu-control', control_config),
                            ('async', ROOT / 'configs/async-mechanism-smoke.json')]:
            result = run(path, ROOT / 'scenarios/async-memory.json', args.device, 'async-' + label)
            results.append(result)
            summary['runs'].append({'label': label, 'response_check': result['response_check'],
                                    'result_directory': result['result_directory'],
                                    'mechanism_check': result.get('mechanism_check')})
            save()
            if not result['response_check']['passed']:
                raise RuntimeError(label + ' response/mechanism check failed; not starting another condition')
        a, b = results
        comparisons = []
        # Compare committed framing as well as generated IDs and JSON; no logits claim.
        for i, (left, right) in enumerate(zip(a['turns'], b['turns'])):
            comparisons.append({'turn': i, 'generated_ids_equal': left['generated_ids'] == right['generated_ids'],
                                'output_equal': left['output'] == right['output'],
                                'next_position_equal': left['next_position'] == right['next_position'],
                                'evaluated_tokens_equal': left['evaluated_tokens'] == right['evaluated_tokens'],
                                'committed_prefix_equal': (ROOT / a['result_directory'] / f'turn-{i}.prefix.txt').read_bytes()
                                    == (ROOT / b['result_directory'] / f'turn-{i}.prefix.txt').read_bytes()})
        summary['comparisons'] = comparisons
        summary['passed'] = len(comparisons) == 4 and all(all(v for k, v in c.items() if k != 'turn') for c in comparisons)
        save()
    except Exception as exc:
        summary['failure'] = str(exc)
        save()
        raise
    print(json.dumps({'suite': str(suite), **summary}, ensure_ascii=False, indent=2))
    if not summary['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
