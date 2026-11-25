#!/usr/bin/env python3
"""汇总并分析 1k/2k/4k 三路 Prefill (独立CPU/共享CPU/独立HTP) 与 M(H) 实测结果（含 Q3+1.5*IQR 异常剔除）。"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/mainline-eval/prefill-3way-h1k2k4k-8b'
ARMS = ('independent_cpu', 'shared_cpu', 'independent_htp')


def quantile(ordered: list[float], q: float) -> float:
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def clean_upper_iqr(rows: list[dict], value_key: str, group_keys: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    kept, excluded = [], []
    for key, grp in sorted(groups.items()):
        vals = sorted(float(r[value_key]) for r in grp)
        q1, q3 = quantile(vals, 0.25), quantile(vals, 0.75)
        threshold = q3 + 1.5 * (q3 - q1)
        grp_kept = [r for r in grp if float(r[value_key]) <= threshold]
        if len(grp_kept) < 2:
            grp_kept = grp
        kept.extend(grp_kept)
        for r in grp:
            if r not in grp_kept:
                excluded.append(dict(r, threshold_ms=threshold, exclusion_rule=f'{value_key} > Q3 + 1.5*IQR'))
    return kept, excluded


def summarize(run_dir: Path) -> dict:
    raw_prefill = list(csv.DictReader((run_dir / 'stage-times.csv').open()))
    raw_handoff = list(csv.DictReader((run_dir / 'handoff-times.csv').open()))

    for r in raw_prefill:
        for k in ('rep', 'threads', 'H', 'P', 'top_token', 'swap_kib', 'rss_kib', 'policy7_before_khz', 'policy7_after_khz'):
            r[k] = int(r[k])
        for k in ('restore_ms', 'prefill_ms'):
            r[k] = float(r[k])
    for r in raw_handoff:
        for k in ('rep', 'H', 'P', 'total_tokens', 'handoff_bytes', 'top_token', 'swap_kib'):
            r[k] = int(r[k])
        for k in ('restore_ms', 'htp_prefill_ms', 'handoff_ms'):
            r[k] = float(r[k])

    kept_pre, excl_pre = clean_upper_iqr(raw_prefill, 'prefill_ms', ('arm', 'H', 'P'))
    kept_hnd, excl_hnd = clean_upper_iqr(raw_handoff, 'handoff_ms', ('H', 'P'))

    for name, rows in (('cleaned-stage-times.csv', kept_pre), ('excluded-stage-times.csv', excl_pre),
                       ('cleaned-handoff-times.csv', kept_hnd), ('excluded-handoff-times.csv', excl_hnd)):
        if rows:
            with (run_dir / name).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    # M(H) summaries
    handoff_summary = {}
    for h in sorted({r['H'] for r in kept_hnd}):
        by_p = {}
        for p in sorted({r['P'] for r in kept_hnd if r['H'] == h}):
            vals = [r['handoff_ms'] for r in kept_hnd if r['H'] == h and r['P'] == p]
            by_p[str(p)] = {
                'n': len(vals),
                'mean_ms': statistics.mean(vals),
                'median_ms': statistics.median(vals),
                'stdev_ms': statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        # Exact H (P=0) and pooled all-P medians
        all_vals = [r['handoff_ms'] for r in kept_hnd if r['H'] == h]
        handoff_summary[str(h)] = {
            'by_P': by_p,
            'exact_H_median_ms': by_p.get('0', {}).get('median_ms', statistics.median(all_vals)),
            'exact_H_mean_ms': by_p.get('0', {}).get('mean_ms', statistics.mean(all_vals)),
        }

    # Linear fit M(total_tokens) = a + b * total_tokens on cleaned handoff rows
    xs = [r['total_tokens'] for r in kept_hnd]
    ys = [r['handoff_ms'] for r in kept_hnd]
    if xs and max(xs) - min(xs) >= 512:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
        intercept = my - slope * mx
    else:
        # Single-H narrow P span fallback: anchor at measured exact-H median using clean 1k/2k/4k slope
        h0 = sorted(handoff_summary)[0]
        slope = 0.06026176115224677
        intercept = handoff_summary[h0]['exact_H_median_ms'] - slope * int(h0)

    # Top-token consistency per (rep, H, P) across arms
    token_mismatches = []
    by_cell_rep = {}
    for r in raw_prefill:
        by_cell_rep.setdefault((r['rep'], r['H'], r['P']), {})[r['arm']] = r['top_token']
    for (rep, h, p), arm_toks in sorted(by_cell_rep.items()):
        if len(set(arm_toks.values())) > 1:
            token_mismatches.append({'rep': rep, 'H': h, 'P': p, 'top_tokens': arm_toks})

    # Per (H, P) comparison table
    table_rows = []
    for h in sorted({r['H'] for r in kept_pre}):
        m_h_med = handoff_summary[str(h)]['exact_H_median_ms']
        for p in sorted({r['P'] for r in kept_pre if r['H'] == h}):
            row = {'H': h, 'P': p, 'M_H_median_ms': m_h_med, 'M_HP_linear_ms': intercept + slope * (h + p)}
            for arm in ARMS:
                vals = [r['prefill_ms'] for r in kept_pre if r['arm'] == arm and r['H'] == h and r['P'] == p]
                raw_vals = [r['prefill_ms'] for r in raw_prefill if r['arm'] == arm and r['H'] == h and r['P'] == p]
                rst_vals = [r['restore_ms'] for r in kept_pre if r['arm'] == arm and r['H'] == h and r['P'] == p]
                row[f'{arm}_n'] = len(vals)
                row[f'{arm}_raw_n'] = len(raw_vals)
                row[f'{arm}_mean_ms'] = statistics.mean(vals)
                row[f'{arm}_median_ms'] = statistics.median(vals)
                row[f'{arm}_restore_median_ms'] = statistics.median(rst_vals)
            row['htp_plus_M_H_ms'] = row['independent_htp_median_ms'] + m_h_med
            row['htp_plus_M_HP_ms'] = row['independent_htp_median_ms'] + row['M_HP_linear_ms']
            table_rows.append(row)

    with (run_dir / 'summary-table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0]))
        w.writeheader()
        w.writerows(table_rows)

    summary = {
        'raw_prefill_rows': len(raw_prefill),
        'kept_prefill_rows': len(kept_pre),
        'excluded_prefill_rows': len(excl_pre),
        'raw_handoff_rows': len(raw_handoff),
        'kept_handoff_rows': len(kept_hnd),
        'excluded_handoff_rows': len(excl_hnd),
        'top_token_mismatches': len(token_mismatches),
        'top_token_mismatch_samples': token_mismatches[:20],
        'handoff': handoff_summary,
        'handoff_linear_fit': {'intercept_ms': intercept, 'slope_ms_per_token': slope},
        'crossings': {},
    }
    for h in sorted({r['H'] for r in table_rows}):
        sub = [r for r in table_rows if r['H'] == h]
        summary['crossings'][str(h)] = {
            'ind_cpu_vs_ind_htp_pure_first_htp_win_P': next((r['P'] for r in sub if r['independent_htp_median_ms'] < r['independent_cpu_median_ms']), None),
            'shr_cpu_vs_ind_htp_pure_first_htp_win_P': next((r['P'] for r in sub if r['independent_htp_median_ms'] < r['shared_cpu_median_ms']), None),
            'ind_cpu_vs_htp_plus_M_first_htp_win_P': next((r['P'] for r in sub if r['htp_plus_M_HP_ms'] < r['independent_cpu_median_ms']), None),
            'shr_cpu_vs_htp_plus_M_first_htp_win_P': next((r['P'] for r in sub if r['htp_plus_M_HP_ms'] < r['shared_cpu_median_ms']), None),
        }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    out = summarize(args.dir.resolve())
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
