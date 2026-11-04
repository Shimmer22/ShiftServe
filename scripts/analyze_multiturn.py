#!/usr/bin/env python3
"""离线汇总多轮输出、prefix、编码次数和 prefill 数值对照。"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = []
for manifest_path in sorted((ROOT / 'results').glob('*/manifest.json')):
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('scenario', {}).get('turns'):
        continue
    folder = manifest_path.parent
    row = {'run': str(folder.relative_to(ROOT)), 'config': manifest['config'], 'passed': False}
    if (folder / 'result.json').exists():
        r = json.loads((folder / 'result.json').read_text())
        row.update(status='complete', passed=r['response_check']['passed'],
                   checks=r['response_check'], turns=r['turns'], image_encode_calls=r['image_encode_calls'])
        comparisons = []
        for i, turn in enumerate(r['turns']):
            if not turn['compared']:
                continue
            a = np.fromfile(folder / f'turn-{i}.cpu.logits.f32', dtype=np.float32).astype(np.float64)
            b = np.fromfile(folder / f'turn-{i}.htp.logits.f32', dtype=np.float32).astype(np.float64)
            la, lb = a-a.max(), b-b.max()
            la -= np.log(np.exp(la).sum())
            lb -= np.log(np.exp(lb).sum())
            comparisons.append({'turn': i, 'n_vocab': len(a),
                                'kl_cpu_to_htp': float(np.sum(np.exp(la)*(la-lb))),
                                'top10_overlap': len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:])),
                                'top1_equal': bool(a.argmax() == b.argmax())})
        if comparisons:
            row['logits_comparison'] = comparisons
            (folder / 'logits-comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    else:
        row['status'] = 'aborted' if (folder / 'aborted.json').exists() else 'incomplete'
        if (folder / 'completed-turns.json').exists():
            row['turns'] = json.loads((folder / 'completed-turns.json').read_text())['turns']
    rows.append(row)
    print(row['run'], row['status'], 'passed=', row['passed'])
    for t in row.get('turns', []):
        print(' ', t['turn'], t['prefill_backend'], 'cached=', t['cached_tokens'],
              'image_encodes=', t['image_encode_calls'], 'cosine=', t['logits_cosine'] if t['compared'] else 'not compared')
(ROOT / 'results/multiturn-summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
