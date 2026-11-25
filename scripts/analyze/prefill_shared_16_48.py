#!/usr/bin/env python3
"""对同进程配对的 16–48 共享 CPU vs HTP Prefill 扫描执行 Q3+1.5*IQR 异常剔除并输出汇总表。"""

import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/mainline-eval/prefill-shared-16-48-h4096-8b'
DEFAULT_M_H_MS = 280.8249480724335


def quantile(ordered: list[float], q: float) -> float:
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def analyze(run_dir: Path, m_h_ms: float) -> dict:
    rows = list(csv.DictReader((run_dir / 'stage-times.csv').open()))
    groups = collections.defaultdict(list)
    paired = collections.defaultdict(dict)
    for r in rows:
        groups[(r['backend'], int(r['P']))].append(r)
        paired[(int(r['round']), int(r['P']))][r['backend']] = float(r['elapsed_ms'])

    kept, excluded = [], []
    for _, grp in sorted(groups.items()):
        vals = sorted(float(r['elapsed_ms']) for r in grp)
        q1, q3 = quantile(vals, 0.25), quantile(vals, 0.75)
        thr = q3 + 1.5 * (q3 - q1)
        gk = [r for r in grp if float(r['elapsed_ms']) <= thr]
        if len(gk) < 2:
            gk = grp
        kept.extend(gk)
        for r in grp:
            if r not in gk:
                excluded.append(dict(r, threshold_ms=thr, exclusion_rule='elapsed_ms > Q3 + 1.5*IQR'))

    for name, out_rows in (('cleaned-stage-times.csv', kept), ('excluded-stage-times.csv', excluded)):
        if out_rows:
            with (run_dir / name).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
                w.writeheader()
                w.writerows(out_rows)

    cleaned_g = collections.defaultdict(list)
    for r in kept:
        cleaned_g[(r['backend'], int(r['P']))].append(float(r['elapsed_ms']))

    table = []
    cvs = []
    ps = sorted({int(r['P']) for r in rows})
    n_rounds = len({int(r['round']) for r in rows})
    for p in ps:
        c_vals = cleaned_g[('cpu', p)]
        h_vals = cleaned_g[('htp', p)]
        c_med = statistics.median(c_vals)
        h_med = statistics.median(h_vals)
        c_cv = (statistics.stdev(c_vals) / c_med * 100.0) if len(c_vals) > 1 else 0.0
        h_cv = (statistics.stdev(h_vals) / h_med * 100.0) if len(h_vals) > 1 else 0.0
        cvs.extend([c_cv, h_cv])
        wins = sum(paired[(rnd, p)]['cpu'] < paired[(rnd, p)]['htp'] for rnd in range(n_rounds))
        med_ratio = statistics.median(paired[(rnd, p)]['cpu'] / paired[(rnd, p)]['htp'] for rnd in range(n_rounds))
        table.append({
            'H': 4096,
            'P': p,
            'cpu_n': len(c_vals),
            'cpu_median_ms': c_med,
            'cpu_mean_ms': statistics.mean(c_vals),
            'cpu_cv_pct': c_cv,
            'htp_n': len(h_vals),
            'htp_median_ms': h_med,
            'htp_mean_ms': statistics.mean(h_vals),
            'htp_cv_pct': h_cv,
            'M_H_ms': m_h_ms,
            'htp_plus_M_H_ms': h_med + m_h_ms,
            'paired_cpu_wins': wins,
            'paired_total': n_rounds,
            'paired_ratio_median': med_ratio,
        })

    with (run_dir / 'summary-table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    summary = {
        'raw_rows': len(rows),
        'kept_rows': len(kept),
        'excluded_rows': len(excluded),
        'median_cv_pct': statistics.median(cvs),
        'max_cv_pct': max(cvs),
        'M_H_ms': m_h_ms,
        'pure_first_htp_win_P': next((r['P'] for r in table if r['htp_median_ms'] < r['cpu_median_ms']), None),
        'plus_M_first_htp_win_P': next((r['P'] for r in table if r['htp_plus_M_H_ms'] < r['cpu_median_ms']), None),
    }
    (run_dir / 'summary-cleaned.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--m-h-ms', type=float, default=DEFAULT_M_H_MS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dir.resolve(), args.m_h_ms), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
