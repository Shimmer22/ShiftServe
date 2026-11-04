#!/usr/bin/env python3
"""四路径校准：完整动作与Decode前缀分开，Gap仅做离线理想重叠。"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.validation.contracts import check_response
from a2o.scheduling.cost_model import path_cost


def main():
    p = argparse.ArgumentParser();p.add_argument('run', type=Path);a = p.parse_args()
    cfg = json.loads((a.run / 'manifest.json').read_text())['config']
    rows = [json.loads(l) for f in a.run.glob('H*.jsonl') for l in f.read_text().splitlines()]
    groups = defaultdict(list);rejected = []
    for r in rows:
        check = check_response(r, {'check': {'type': 'json_action', 'action': 'wait'}})
        if not (check['passed'] and r['eog'] and r['numerical_passed'] and r['generated_ids_equal']):
            rejected.append({'row': r, 'response_check': check});continue
        for d in [None, *cfg['decode_prefix_tokens']]:
            if d is not None and d > len(r['decode_prefix_ms']):continue
            for g in cfg['gap_ms']:
                key = (r['history_tokens'], r['new_tokens'], r['home'], d or 'complete', g, r['path'])
                groups[key].append(path_cost(r, g, d))
    table = []
    for (h,p,home,d,g,path),values in groups.items():
        table.append({'H':h,'P':p,'home':home,'D':d,'G_ms':g,'path':path,'n':len(values),
                      'total_ms':median(v['total_ms'] for v in values),
                      'min_ms':min(v['total_ms'] for v in values),'max_ms':max(v['total_ms'] for v in values),
                      'prepare_ms':median(v['prepare_ms'] for v in values),
                      'phase_switch_ms':median(v['phase_switch_ms'] for v in values),
                      'decode_ms':median(v['decode_ms'] for v in values)})
    cells = defaultdict(list)
    for r in table:
        cells[(r['H'],r['P'],r['home'],r['D'],r['G_ms'])].append(r)
    winners = []
    for (h,p,home,d,g),candidates in cells.items():
        valid = len(candidates)==4 and all(r['n']>=cfg['repetitions'] for r in candidates)
        best = min(candidates,key=lambda r:r['total_ms'])
        others = [r for r in candidates if r['path'] != best['path']]
        failed_paths = sorted({x['row']['path'] for x in rejected
                               if x['row']['history_tokens']==h and x['row']['new_tokens']==p and x['row']['home']==home})
        winners.append({'H':h,'P':p,'home':home,'D':d,'G_ms':g,
                        'choice':best['path'] if valid else 'validation_failed' if failed_paths else 'insufficient_samples',
                        'failed_paths':failed_paths,
                        'total_ms':best['total_ms'] if valid else None,
                        'margin_ms': min(r['total_ms'] for r in others)-best['total_ms'] if valid else None,
                        'observed_ranges_separated': valid and all(best['max_ms'] < r['min_ms'] for r in others)})
    flips = []
    for r in winners:
        if r['G_ms']==0 or r['choice'] not in cfg['paths']:continue
        zero = next((x for x in winners if all(x[k]==r[k] for k in ['H','P','home','D']) and x['G_ms']==0),None)
        if zero and zero['choice'] in cfg['paths'] and zero['choice']!=r['choice']:
            flips.append({'H':r['H'],'P':r['P'],'home':r['home'],'D':r['D'],'G_ms':r['G_ms'],
                          'without_gap':zero['choice'],'with_gap':r['choice'],
                          'observed_ranges_separated': zero['observed_ranges_separated'] and r['observed_ranges_separated']})
    result = {'scope':'text short-action calibration; complete cost includes EOG commit/state readiness; D is number of token forward evaluations, not predicted output length; prefixes are partial compute costs, not complete responses; ideal Gap overlap only; no contention/online scheduler validation',
              'trials':len(rows),'rejected':rejected,'table':table,'winners':winners,'gap_flips':flips,
              'response_passed_trials':sum(r['eog'] and check_response(r, {'check': {'type': 'json_action', 'action': 'wait'}})['passed'] for r in rows),
              'numerical_passed_trials':sum(r['numerical_passed'] for r in rows),
              'identical_generated_trials':sum(r['generated_ids_equal'] for r in rows),
              'complete_action_gap_flips':[r for r in flips if r['D']=='complete'],
              'partial_compute_gap_flips':[r for r in flips if r['D']!='complete']}
    (a.run/'path-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if table:
        with (a.run/'path-costs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for r in winners:
        if r['G_ms']==0:print(r)
    print('trials',len(rows),'rejected',len(rejected),'ideal_gap_flips',len(flips))


if __name__=='__main__':main()
