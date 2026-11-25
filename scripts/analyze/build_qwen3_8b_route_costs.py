#!/usr/bin/env python3
"""从干净 6 线程共享权重配对、跨 H (1k/2k/4k) 与扩展大 P 实测生成 Qwen3-8B WLS 2D 路由成本表。"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFILL_DIR = ROOT / 'results/mainline-eval/prefill-shared-16-48-h4096-8b'
DEFAULT_CLEAN_DIR = ROOT / 'results/mainline-eval/prefill-3way-h1k2k4k-8b-clean'
DEFAULT_EXT_DIR = ROOT / 'results/mainline-eval/prefill-3way-h4096-8b-extended'
DEFAULT_OUT = ROOT / 'configs/qwen3-8b-route-costs.json'


def ols_fit(xs: list[float], ys: list[float]) -> dict:
    assert len(xs) == len(ys) and len(xs) >= 2
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    preds = [intercept + slope * x for x in xs]
    errs = [y - p for y, p in zip(ys, preds)]
    ss_res = sum(e * e for e in errs)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = math.sqrt(ss_res / len(xs))
    max_abs = max(abs(e) for e in errs)
    assert math.isfinite(intercept) and math.isfinite(slope) and intercept >= 0 and slope >= 0
    return {
        'intercept_ms': intercept,
        'slope_ms_per_token': slope,
        'n': len(xs),
        'rmse_ms': rmse,
        'r2': r2,
        'max_abs_error_ms': max_abs,
    }


def _solve_3x3(A: list[list[float]], b: list[float]) -> list[float]:
    M = [A[i][:] + [b[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(3):
            if r != col:
                factor = M[r][col] / M[col][col]
                for c in range(col, 4):
                    M[r][c] -= factor * M[col][c]
    return [M[i][3] / M[i][i] for i in range(3)]


def _fit_2d_ols(pts: list[tuple[int, int, float]]) -> list[float]:
    X = [[1.0, float(p), (float(h) / 1000.0) * float(p)] for h, p, _ in pts]
    ys = [float(y) for _, _, y in pts]
    A = [[sum(X[k][i] * X[k][j] for k in range(len(pts))) for j in range(3)] for i in range(3)]
    b = [sum(X[k][i] * ys[k] for k in range(len(pts))) for i in range(3)]
    return _solve_3x3(A, b)


def _wls_1d(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    slope = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sum(
        w * (x - mx) ** 2 for w, x in zip(ws, xs)
    )
    return my - slope * mx, slope


def _eval_term(term: dict, p: float, h: float) -> float:
    return (
        term['intercept_ms']
        + term['slope_ms_per_token'] * p
        + term.get('hp_slope_ms_per_1k_tokens', 0.0) * (h / 1000.0) * p
    )


def build_cost_table(prefill_dir: Path, clean_dir: Path, ext_dir: Path) -> dict:
    rows16 = list(csv.DictReader((prefill_dir / 'summary-table.csv').open()))
    clean_stage = list(csv.DictReader((clean_dir / 'cleaned-stage-times.csv').open()))
    ext_stage = list(csv.DictReader((ext_dir / 'cleaned-stage-times.csv').open()))

    def _group_means(rows, arm):
        grouped = {}
        for r in rows:
            if r['arm'] == arm:
                grouped.setdefault((int(r['H']), int(r['P'])), []).append(float(r['prefill_ms']))
        return {k: sum(v) / len(v) for k, v in grouped.items()}

    htp_ext_means = _group_means(ext_stage, 'independent_htp')

    # Cross-H attention slope w_hp from unified clean + extended runs (exact same single-backend binary across 1k/2k/4k).
    cpu_clean_raw = [(int(r['H']), int(r['P']), float(r['prefill_ms'])) for r in clean_stage if r['arm'] == 'shared_cpu']
    cpu_whp = _fit_2d_ols(cpu_clean_raw)[2]

    htp_same_raw = [
        (int(r['H']), int(r['P']), float(r['prefill_ms']))
        for r in (clean_stage + ext_stage)
        if r['arm'] == 'independent_htp'
    ]
    htp_whp = _fit_2d_ols(htp_same_raw)[2]

    # CPU (6T shared): relative-error WLS (1/y^2) on decision-sensitive paired H=4096 (P=16..48) anchored with w_hp.
    cpu_1648 = [(float(int(r['P'])), float(r['cpu_median_ms'])) for r in rows16]
    cpu_w0, cpu_s4k = _wls_1d(
        [p for p, _ in cpu_1648],
        [y for _, y in cpu_1648],
        [1.0 / (y * y) for _, y in cpu_1648],
    )
    cpu_wp = cpu_s4k - cpu_whp * 4.096
    assert cpu_w0 >= 0 and cpu_wp >= 0 and cpu_whp >= 0

    # HTP (4T host): relative-error WLS (1/y^2) over post-HMX paired H=4096 (P=32..48) + large-P extended (512..2048).
    htp_1648_p32 = [(float(int(r['P'])), float(r['htp_median_ms'])) for r in rows16 if int(r['P']) >= 32]
    htp_big = [(float(p), htp_ext_means[(4096, p)]) for p in (512, 1024, 2048)]
    htp_pts = [(p, y, 1.0) for p, y in htp_1648_p32] + [(p, y, 2.0) for p, y in htp_big]
    htp_w0, htp_s4k = _wls_1d(
        [p for p, _, _ in htp_pts],
        [y for _, y, _ in htp_pts],
        [mult / (y * y) for _, y, mult in htp_pts],
    )
    htp_wp = htp_s4k - htp_whp * 4.096
    assert htp_w0 >= 0 and htp_wp >= 0 and htp_whp >= 0

    rows_clean = list(csv.DictReader((clean_dir / 'summary-table.csv').open()))
    hs_clean = [float(int(r['H'])) for r in rows_clean]
    cpu_rsts = [float(r['shared_cpu_restore_median_ms']) for r in rows_clean]
    htp_rsts = [float(r['independent_htp_restore_median_ms']) for r in rows_clean]

    hnd_rows = list(csv.DictReader((clean_dir / 'cleaned-handoff-times.csv').open()))
    hnd_ext = list(csv.DictReader((ext_dir / 'cleaned-handoff-times.csv').open()))
    hnd_xs = [float(int(r['total_tokens'])) for r in (hnd_rows + hnd_ext)]
    hnd_ys = [float(r['handoff_ms']) for r in (hnd_rows + hnd_ext)]

    cpu_term = {
        'intercept_ms': cpu_w0,
        'slope_ms_per_token': cpu_wp,
        'hp_slope_ms_per_1k_tokens': cpu_whp,
        'weight': '1/y^2',
        'slope_at_4096_ms_per_token': cpu_s4k,
        'n': len(cpu_1648),
    }
    htp_term = {
        'intercept_ms': htp_w0,
        'slope_ms_per_token': htp_wp,
        'hp_slope_ms_per_1k_tokens': htp_whp,
        'weight': '1/y^2 (big-P balanced 2x)',
        'slope_at_4096_ms_per_token': htp_s4k,
        'n': len(htp_pts),
        'h4096_large_p_errors_pct': {
            str(int(p)): round((_eval_term({'intercept_ms': htp_w0, 'slope_ms_per_token': htp_wp, 'hp_slope_ms_per_1k_tokens': htp_whp}, p, 4096) - actual) / actual * 100.0, 2)
            for p, actual in htp_big
        },
    }

    model = {
        'kind': 'linear_hp_v1',
        'model_id': 'qwen3-8b-q4_0-6t-wls-2d',
        'routing_p_min': 16,
        'h_min': 1024,
        'h_max': 4096,
        'p_min': 16,
        'p_max': 2048,
        'formula': 'CC=R(H)+C(P,H); HC=N(P,H)+M(H+P); common CPU decode excluded',
        'preparation': 'always prepare; ready only; CPU-to-HTP preparation hidden by actual tool gap',
        'extrapolation': '2D WLS (1, P, H/1000*P) across H=1k..4k, P=16..2048; every extrapolated decision logged',
        'cpu_threads': 6,
        'htp_host_threads': 4,
        'cpu_restore': ols_fit(hs_clean, cpu_rsts),
        'cpu_to_htp': ols_fit(hs_clean, htp_rsts),
        'cpu_prefill': cpu_term,
        'htp_prefill': htp_term,
        'htp_to_cpu': ols_fit(hnd_xs, hnd_ys),
        'provenance': {
            'prefill_source': str(prefill_dir.relative_to(ROOT)),
            'cross_h_and_handoff_source': str(clean_dir.relative_to(ROOT)),
            'extended_p_source': str(ext_dir.relative_to(ROOT)),
            'build_id': 'prefill-3way-h1k2k4k-c9f439f08c814ea3b9814c0cff54eb19',
            'git_commit': '76b095695a70d10e2dbb4a68fc705e39c010cbd9',
        },
    }
    return {'model': model}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefill-dir', type=Path, default=DEFAULT_PREFILL_DIR)
    parser.add_argument('--clean-dir', type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument('--ext-dir', type=Path, default=DEFAULT_EXT_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build_cost_table(args.prefill_dir.resolve(), args.clean_dir.resolve(), args.ext_dir.resolve())
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(f'Wrote {args.output}')
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
