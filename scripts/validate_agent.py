#!/usr/bin/env python3
"""Small sequential phone validation; results retain each condition independently."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.runner import ROOT, run, default_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['diagnosis', 'matrix', 'vision', 'vision-history', 'npu-cpu', 'multiturn'], required=True)
    p.add_argument('--device', default=default_device())
    a = p.parse_args()
    if a.suite == 'multiturn':
        jobs = [('multiturn-cpu', 'multiturn-screenshots'), ('multiturn-baseline', 'multiturn-screenshots')]
    elif a.suite == 'diagnosis':
        jobs = [('npu-none', 'original-context'), ('cpu-none', 'ready'),
                ('npu-roundtrip', 'history'), ('cpu-roundtrip', 'history')]
    elif a.suite == 'matrix':
        jobs = [(f'{mode}-{policy}' + ('-pre' if pre else ''), 'history')
                for mode in ['cpu', 'npu', 'hybrid'] for policy in ['none', 'recent']
                for pre in ([False, True] if mode == 'hybrid' else [False])]
    elif a.suite == 'vision':
        jobs = [(name, 'screenshot') for name in ['cpu-none', 'npu-none', 'hybrid-none']]
    elif a.suite == 'npu-cpu':
        jobs = [(name, 'screenshot-history') for name in
                ['hybrid-cpu-vision', 'hybrid-opencl-demand', 'hybrid-htp-demand', 'hybrid-htp-a2o']]
    else:
        jobs = [(name, 'screenshot-history') for name in
                ['hybrid-none', 'hybrid-none-pre', 'hybrid-recent', 'hybrid-recent-pre']]
    report = []
    for name, scenario in jobs:
        print(f'START {name} / {scenario}', flush=True)
        result = run(ROOT / 'configs' / (name + '.json'),
                     ROOT / 'scenarios' / (scenario + '.json'), a.device, name + '-' + scenario)
        report.append({'config': name, 'scenario': scenario, 'passed': result['response_check']['passed']})
        (ROOT / 'results' / (a.suite + '-latest.json')).write_text(json.dumps(report, indent=2))
    if not all(x['passed'] for x in report):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
