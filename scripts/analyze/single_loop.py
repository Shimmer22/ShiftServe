#!/usr/bin/env python3
"""离线交叉核验单闭环证据，并导出 CPU/RSS 采样；不连接手机。"""
import argparse
import csv
import json
from pathlib import Path


def analyze(out):
    report = json.loads((out / 'summary.json').read_text())
    native = json.loads((out / 'device/result.json').read_text())
    host = [json.loads(line) for line in (out / 'host-events.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (out / 'native-events.jsonl').read_text().splitlines()]
    turns = report['turns']

    def index(name, turn=None):
        return next((i for i, e in enumerate(host) if e['event'] == name and
                     (turn is None or e.get('turn') == turn)), -1)

    order = [index('response_received', 0), index('action_issued'), index('page_complete'),
             index('page_rendered'), index('environment_change_confirmed'),
             index('observation_captured', 1), index('inference_issued', 1), index('response_received', 1)]
    checks = {'reported_passed': report['passed'], 'native_matches_protocol': native['turns'] == turns,
        'actual_feedback_order': min(order) >= 0 and order == sorted(set(order)),
        'single_session': sum(e['event'] == 'session_ready' for e in events) == 1
                          and sum(e['event'] == 'session_closed' for e in events) == 1,
        'two_distinct_live_screenshots': len({e['screenshot_sha256'] for e in host
                                            if e['event'] == 'observation_captured'}) == 2,
        'two_single_encodes': len(turns) == 2 and native['image_encode_calls'] == 2
                             and all(r['image_encode_calls'] == 1 for r in turns),
        'committed_history_reused': len(turns) == 2 and turns[1]['cached_tokens'] == turns[0]['evaluated_tokens']
                                  and turns[1]['cached_state_bytes'] == turns[0]['state_bytes'],
        'state_and_eog_checked': all(r['prefix_match'] and r['restore_checked'] and r['eog_committed'] for r in turns),
        'new_observation_confirmed_with_memory': json.loads(turns[-1]['output']) ==
                                               {'action': 'done', 'memo': report['task_code']},
        'actual_environment_changed': report['actual_click_received'] and report['actual_page_rendered']}
    timelines = [{'event': e['event'], 'turn': e.get('turn'),
                  'elapsed_ms': (e['host_monotonic_ns'] - host[0]['host_monotonic_ns']) / 1e6} for e in host]
    samples = []
    previous = None
    for block in (out / 'device-resources.log').read_text().split('SAMPLE\n')[1:]:
        lines = block.splitlines()
        if not lines: continue
        cpu = next((s for s in lines if s.startswith('cpu ')), None)
        rss = next((s for s in lines if s.startswith('VmRSS:')), None)
        if cpu is None or rss is None: continue
        values = list(map(int, cpu.split()[1:9]))
        total, idle = sum(values), values[3] + values[4]
        percent = None
        if previous and total > previous[0]:
            percent = 100 * (1 - (idle - previous[1]) / (total - previous[0]))
        samples.append({'device_boottime_ms': float(lines[0].split()[0]) * 1000,
                        'system_cpu_busy_percent': percent, 'model_rss_kib': int(rss.split()[1])})
        previous = total, idle
    with (out / 'resource-samples.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['device_boottime_ms', 'system_cpu_busy_percent', 'model_rss_kib'])
        writer.writeheader(); writer.writerows(samples)
    result = {'passed': all(checks.values()), 'checks': checks, 'host_timeline': timelines,
              'resource_samples': len(samples),
              'scope': 'single controlled browser task; CPU LLM/OpenCL vision; diagnostic sampling; no HTP utilization/energy/performance claim'}
    (out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    result = analyze(args.result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']: raise SystemExit(2)
