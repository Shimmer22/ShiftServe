#!/usr/bin/env python3
"""只读复核原策略图 P31：保留逐次阶段和前驱条件，不删除异常点。"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/20261105-xiaomi14pro-observation-main/figures/plot-data.json'
OUT = ROOT / 'results/20261105-post-tool-p31-audit'


def main():
    data = json.loads(SOURCE.read_text())
    OUT.mkdir(exist_ok=True)
    rows = data['rows']
    details, summary = [], []
    for h in (2048, 4096):
        by_id = {r['id']: r for r in rows if r['H'] == h}
        for gap in (0, 50, 200):
            for req in ('CC', 'router'):
                group = [r for r in by_id.values() if r['phase'] == 'measurement'
                         and r['P'] == 31 and r['strategy'] == 'after_return'
                         and r['requested_route'] == req and r['gap_requested_ms'] == gap]
                assert len(group) == 3
                assert all(r['route'] == 'CC' and not r['prepared_early']
                           and r['passed'] and r['response_check']['passed'] for r in group)
                summary.append(dict(H=h, gap_target_ms=gap, requested_route=req,
                    trial_ids=[r['id'] for r in group],
                    **{k: median(r[k] for r in group) for k in
                       ('tool_to_response_ms', 'prefill_ms', 'decode_ms', 'actual_gap_ms')}))
                for r in group:
                    prev = by_id[r['id'] - 1]
                    details.append({**{k: r[k] for k in ('H', 'P', 'id', 'repetition',
                        'gap_requested_ms', 'requested_route', 'route', 'actual_gap_ms',
                        'tool_to_response_ms', 'prefill_ms', 'decode_ms',
                        'wait_or_prepare_after_return_ms', 'suspend_during_tool_to_response_ms')},
                        'previous_condition': '/'.join(str(prev[k]) for k in
                            ('P', 'strategy', 'requested_route', 'route'))})
    provenance = []
    for source in data['sources']:
        directory = Path(source['directory'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        raw = [json.loads(line) for line in (directory / 'checked-rows.jsonl').read_text().splitlines()]
        plotted = [r for r in rows if r['H'] == manifest['config']['history_tokens']]
        assert raw == plotted, '图中行与原始检查行不一致'
        provenance.append(dict(directory=str(directory), seed=manifest['config']['seed'],
                               manifest_sha256=hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()))
    (OUT / 'audit.json').write_text(json.dumps(dict(source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        provenance=provenance, summary=summary, details=details), ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'trials.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(dict(output=str(OUT), checked_trials=len(details),
                          seeds=[p['seed'] for p in provenance]), ensure_ascii=False))


if __name__ == '__main__':
    main()
