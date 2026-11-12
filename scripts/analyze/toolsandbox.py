#!/usr/bin/env python3
"""从已保存轨迹导出每工具 H/P/G/D 表；不把评分或运行退出码混为通过。"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    rows, groups = [], []
    for result in summary['results']:
        metrics = json.loads((args.run / result['name'] / 'metrics.json').read_text())
        for tool in metrics['tools']:
            rows.append(dict(scenario=result['name'], **{key: tool.get(key) for key in
                ['tool_id', 'tool', 'H', 'P_body', 'G_ms', 'execution_ms', 'H_common_prefix',
                 'P_append', 'D_next', 'issue_to_next_request_ms', 'result_to_next_request_ms',
                 'issue_to_response_ms', 'result_to_response_ms', 'tool_error']}))
        group = dict(name=result['name'], trace_complete=result['trace_complete'],
                     official_score=(result.get('evaluation') or {}).get('similarity'),
                     inference_calls=len(metrics['inference']), tool_calls=len(metrics['tools']))
        group['agent_e2e_ms'] = result.get('agent_e2e_ms')
        group['inference_http_ms'] = sum(step['host_request_ms'] for step in metrics['inference'])
        group['measurement_preparation_ms'] = sum(step['measurement_preparation_ms'] for step in metrics['inference'])
        group['tool_errors'] = sum(tool.get('tool_error') is not None for tool in metrics['tools'])
        for key in ['H', 'P_body', 'G_ms', 'P_append', 'D_next']:
            values = [tool[key] for tool in metrics['tools'] if tool.get(key) is not None]
            group[key] = dict(min=min(values), median=median(values), max=max(values)) if values else None
        group['D_all'] = [step['D'] for step in metrics['inference']]
        group['backend_timings'] = [step.get('backend_timings') for step in metrics['inference']]
        groups.append(group)
    if rows:
        with (args.run / 'tool-metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(groups=groups, total_tools=len(rows),
                  all_traces_complete=all(g['trace_complete'] for g in groups))
    (args.run / 'metrics-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
