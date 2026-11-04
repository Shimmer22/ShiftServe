#!/usr/bin/env python3
"""从实际交接/prefill 测量离线模拟理想 Gap 重叠，导出第一版路由查表。"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.scheduling.cost_model import route_times
from a2o.validation.contracts import check_response


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.run / 'manifest.json').read_text())
    config = manifest['config']
    rows = [json.loads(line) for f in sorted(a.run.glob('H*.jsonl')) for line in f.read_text().splitlines()]
    groups = defaultdict(list)
    rejected = []
    for row in rows:
        checks = {b: check_response(row[b], {'check': {'type': 'json_action', 'action': 'wait'}}) for b in ['cpu', 'htp']}
        if not all(c['passed'] for c in checks.values()) or not row['numerical_passed']:
            rejected.append({'history_tokens': row['history_tokens'], 'new_tokens': row['new_tokens'],
                             'repetition': row['repetition'], 'checks': checks, 'numerical_passed': row['numerical_passed']})
            continue
        groups[(row['history_tokens'], row['new_tokens'])].append(row)
    table = []
    breakdown = []
    cell_status = []
    for h in config['history_tokens']:
        for p in config['new_tokens']:
            valid = len(groups.get((h, p), []))
            failed = sum(r['history_tokens'] == h and r['new_tokens'] == p for r in rejected)
            status = ('validated' if valid >= config['repetitions'] else
                      'validation_failed' if failed else 'insufficient_samples' if valid else 'not_measured')
            cell_status.append({'history_tokens': h, 'new_tokens': p, 'valid_trials': valid,
                                'rejected_trials': failed, 'status': status})
    for (h, p), trials in sorted(groups.items()):
        breakdown.append({'history_tokens': h, 'new_tokens': p, 'repetitions': len(trials),
                          'cpu_prefill_ms': median(t['cpu']['prefill_ms'] for t in trials),
                          'htp_prefill_ms': median(t['htp']['prefill_ms'] for t in trials),
                          'export_ms': median(t['htp']['state_export_ms'] for t in trials),
                          'import_ms': median(t['htp']['state_import_ms'] for t in trials),
                          'tiling_ms': None, 'prefill_batches': (p+config['batch_size']-1)//config['batch_size'],
                          'last_batch_tokens': (p-1)%config['batch_size']+1,
                          'cpu_decode_ms': median(t['cpu']['decode_ms'] for t in trials),
                          'after_htp_cpu_decode_ms': median(t['htp']['decode_ms'] for t in trials)})
        for gap in config['gap_ms']:
            costs = [route_times(t, gap) for t in trials]
            cpu, htp = median(c['cpu_ms'] for c in costs), median(c['htp_ms'] for c in costs)
            table.append({'history_tokens': h, 'new_tokens': p, 'gap_ms': gap, 'repetitions': len(trials),
                          'cpu_ms': cpu, 'htp_ms': htp, 'prepare_ms': median(c['prepare_ms'] for c in costs),
                          'return_ms': median(c['return_ms'] for c in costs),
                          'cpu_min_ms': min(c['cpu_ms'] for c in costs), 'cpu_max_ms': max(c['cpu_ms'] for c in costs),
                          'htp_min_ms': min(c['htp_ms'] for c in costs), 'htp_max_ms': max(c['htp_ms'] for c in costs),
                          'choice': ('htp' if htp < cpu else 'cpu') if len(trials)>=config['repetitions'] else 'insufficient_samples'})
    report = {'scope': 'ideal overlap lookup for measured text suffixes; no real Tool concurrency or GPU contention measured',
              'expected_cells': len(config['history_tokens'])*len(config['new_tokens']),
              'measured_cells': len({(r['history_tokens'], r['new_tokens']) for r in rows}),
              'validated_cells': sum(c['status'] == 'validated' for c in cell_status),
              'cell_status': cell_status, 'rejected_trials': rejected,
              'breakdown': breakdown, 'router_table': table}
    report['direct_cpu_checks'] = [
        {'history_tokens': r['history_tokens'], 'new_tokens': r['new_tokens'],
         'original_output': r['direct_cpu']['output'], 'restored_output': r['cpu']['output'],
         'original_first_is_eog': r['direct_cpu']['first_is_eog'],
         'restored_first_is_eog': r['cpu']['first_is_eog'],
         'logits_exact': r['direct_restore_logits_exact'], 'nrmse': r['direct_restore_nrmse'],
         'generated_ids_equal': r['direct_restore_ids_equal']}
        for r in rows if 'direct_cpu' in r]
    (a.run / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if table:
        with (a.run / 'router-table.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    for row in table:
        if row['gap_ms'] in [0, max(config['gap_ms'])]:
            print('H=', row['history_tokens'], 'P=', row['new_tokens'], 'G=', row['gap_ms'],
                  row['choice'], 'CPU=', round(row['cpu_ms'], 1), 'HTP=', round(row['htp_ms'], 1))
    print('validated cells:', report['validated_cells'], 'rejected trials:', len(rejected))


if __name__ == '__main__':
    main()
