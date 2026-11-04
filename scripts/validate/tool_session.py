#!/usr/bin/env python3
"""串行 CPU 对照与跨轮/Router/实际 Tool 基础验证，不比较耗时。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import default_device
from a2o.runtime.tool_session import run_tool_session

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default=default_device())
    a = p.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / f'{stamp}-tool-session-validation'; out.mkdir(parents=True)
    summary = {'passed': False, 'timing_valid_for_performance': False, 'runs': [],
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'scope': '实际只读 adb wm size Tool；屏障和成本fixture仅验机制；无性能结论'}
    def save():
        (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    save()
    try:
        results = []
        for control in (True, False):
            result = run_tool_session(ROOT / 'configs/tool-session-smoke.json', ROOT / 'scenarios/tool-session-memory.json',
                                      a.device, control=control, mechanism=not control)
            summary['runs'].append(result); results.append(result); save()
            if not result['passed']:
                raise RuntimeError('basic response/lifecycle check failed; stopping suite')
        left, right = results
        lm, rm = [json.loads((ROOT / r['result_directory'] / 'manifest.json').read_text()) for r in results]
        summary['same_binary'] = lm['binary_sha256'] == rm['binary_sha256']
        summary['same_native_sources'] = lm['source_files_sha256'] == rm['source_files_sha256']
        comparisons = []
        for i, (l, r) in enumerate(zip(left['turns'], right['turns'])):
            paths = [ROOT / res['result_directory'] / 'device' / f'result.json.turn-{i}.prefix.txt' for res in results]
            raw_tools = [ROOT / res['result_directory'] / f'tool-{i}.raw.txt' for res in results]
            comparisons.append({'turn': i, 'output_equal': l['output'] == r['output'],
                'generated_ids_equal': l['generated_ids'] == r['generated_ids'],
                'prefix_equal': paths[0].read_bytes() == paths[1].read_bytes(),
                'tool_result_equal': raw_tools[0].read_bytes() == raw_tools[1].read_bytes(),
                'next_position_equal': l['next_position'] == r['next_position'],
                'evaluated_tokens_equal': l['evaluated_tokens'] == r['evaluated_tokens']})
        summary['comparisons'] = comparisons
        summary['passed'] = (len(comparisons) == 6 and summary['same_binary'] and summary['same_native_sources']
                             and all(all(v for k,v in row.items() if k != 'turn') for row in comparisons))
        save()
    except Exception as exc:
        summary['failure'] = str(exc); save(); raise
    print(json.dumps({'suite': str(out), **summary}, ensure_ascii=False, indent=2))
    if not summary['passed']:
        raise SystemExit(2)
