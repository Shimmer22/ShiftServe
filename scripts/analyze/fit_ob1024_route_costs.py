#!/usr/bin/env python3
"""Fit constrained 2D relative-error WLS route cost table from ob1024 2D calibration grid
and run dual-end acceptance validation (calendar T1-T5 + short-P crossovers + ToolSandbox offline CC hits)."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.scheduling.kv_reduce_cost import plan_joint_route_kv

DEFAULT_CAL_DIR = ROOT / "results/mainline-eval/20261125-ob1024-2d-cal-6t"
DEFAULT_INFER_HC_DIR = ROOT / "results/mainline-eval/20261126-ob1024-infer-hc-cal-6t"
DEFAULT_BASE_COSTS = ROOT / "configs/qwen3-8b-route-costs.json"
DEFAULT_COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
DEFAULT_KV_COSTS = ROOT / "results/mainline-eval/t6-8b-table1-revised/t5-kv-priors.json"
DEFAULT_DT_CONFIRM = ROOT / "results/mainline-eval/20261125-dt-option-a-confirm/summary.json"
DEFAULT_OUT = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"


def fit_1d_nonneg_ols(xs: list[float], ys: list[float]) -> dict:
    """Fit y = max(0, a + b*x) via OLS with non-negative intercept/slope clamping."""
    n = len(xs)
    assert n >= 2
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    candidates = [(my, 0.0), (0.0, max(0.0, sum(x * y for x, y in zip(xs, ys)) / max(1e-9, sum(x * x for x in xs))))]
    if sxx > 1e-9:
        b = sxy / sxx
        a = my - b * mx
        if a >= 0 and b >= 0:
            candidates.append((a, b))
    a, b = min(candidates, key=lambda ab: sum((ab[0] + ab[1] * x - y) ** 2 for x, y in zip(xs, ys)))
    errs = [y - (a + b * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sum(e * e for e in errs) / ss_tot if ss_tot > 0 else 1.0
    return {
        "intercept_ms": max(0.0, a),
        "slope_ms_per_token": max(0.0, b),
        "n": n,
        "rmse_ms": rmse,
        "r2": r2,
        "max_abs_error_ms": max(abs(e) for e in errs),
    }


def eval_linear(term: dict, x: float, h: float = 0.0) -> float:
    return max(
        0.0,
        float(term["intercept_ms"])
        + float(term["slope_ms_per_token"]) * float(x)
        + float(term.get("hp_slope_ms_per_1k_tokens") or 0.0) * (float(h) / 1000.0) * float(x),
    )


def _solve_dense(A: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    if n == 0:
        return []
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col:
                factor = M[r][col] / M[col][col]
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def solve_bounded_wls_3d(
    rows: list[tuple[int, int, float, float]],
    lower_bounds: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Exact active-set solver for min sum w_k*(y_k - (b0 + b1*P + b2*(H/1000)*P))^2 s.t. b >= lower_bounds.
    rows: list of (H, P, y, weight).
    """
    best_beta: list[float] | None = None
    best_loss = float("inf")
    features = [[1.0, float(p), (float(h) / 1000.0) * float(p)] for h, p, _, _ in rows]

    for active_mask in itertools.product((False, True), repeat=3):
        free_idx = [i for i in range(3) if not active_mask[i]]
        beta = [lower_bounds[i] if active_mask[i] else 0.0 for i in range(3)]
        if free_idx:
            m = len(free_idx)
            A = [[0.0] * m for _ in range(m)]
            b_vec = [0.0] * m
            for (h, p, y, w), x in zip(rows, features):
                y_adj = y - sum(x[i] * beta[i] for i in range(3) if active_mask[i])
                for ai, fi in enumerate(free_idx):
                    b_vec[ai] += w * x[fi] * y_adj
                    for aj, fj in enumerate(free_idx):
                        A[ai][aj] += w * x[fi] * x[fj]
            sol = _solve_dense(A, b_vec)
            if sol is None:
                continue
            for ai, fi in enumerate(free_idx):
                beta[fi] = sol[ai]

        if any(beta[i] < lower_bounds[i] - 1e-9 for i in range(3)):
            continue
        beta = [max(lower_bounds[i], beta[i]) for i in range(3)]
        loss = sum(
            w * (y - (beta[0] * x[0] + beta[1] * x[1] + beta[2] * x[2])) ** 2
            for (_, _, y, w), x in zip(rows, features)
        )
        if loss < best_loss:
            best_loss = loss
            best_beta = beta

    assert best_beta is not None, "No feasible bounded WLS solution found"
    return best_beta[0], best_beta[1], best_beta[2]


def make_term(
    rows: list[tuple[int, int, float, float]],
    lower_bounds: tuple[float, float, float],
    weight_desc: str,
) -> dict:
    a, b, c = solve_bounded_wls_3d(rows, lower_bounds)
    rel_errs = []
    for h, p, y, _ in rows:
        pred = a + b * p + c * (h / 1000.0) * p
        rel_errs.append(abs(pred - y) / y * 100.0)
    return {
        "intercept_ms": a,
        "slope_ms_per_token": b,
        "hp_slope_ms_per_1k_tokens": c,
        "weight": weight_desc,
        "bounds": {
            "intercept_min_ms": lower_bounds[0],
            "slope_min_ms_per_token": lower_bounds[1],
            "hp_slope_min_ms_per_1k_tokens": lower_bounds[2],
        },
        "n": len(rows),
        "mape_pct": sum(rel_errs) / len(rel_errs),
        "max_abs_rel_err_pct": max(rel_errs),
        "samples": [{"H": h, "P": p, "median_ms": y} for h, p, y, _ in rows],
    }


def eval_calendar(htp_term: dict, dt_summary: dict) -> dict:
    out = {}
    for arm in ("shared_static_hc", "full_htp_default_ob1024"):
        turns = []
        for t in dt_summary["arms"][arm]["turns"][1:]:
            h, p, actual = int(t["H"]), int(t["P"]), float(t["prompt_ms"])
            pred = eval_linear(htp_term, p, h)
            err_pct = (pred - actual) / actual * 100.0
            turns.append({
                "turn": t["turn"],
                "H": h,
                "P": p,
                "actual_ms": round(actual, 2),
                "pred_ms": round(pred, 2),
                "err_pct": round(err_pct, 2),
            })
        out[arm] = {
            "max_abs_err_pct": round(max(abs(x["err_pct"]) for x in turns), 2),
            "turns": turns,
        }
    return out


def eval_short_p_crossovers(
    grid_rows: list[dict],
    base_model: dict,
    cpu_term: dict,
    htp_term: dict,
) -> dict:
    short_rows = [r for r in grid_rows if r["grid"] == "short"]
    by_h: dict[int, list[dict]] = {}
    for r in short_rows:
        by_h.setdefault(int(r["H"]), []).append(r)

    per_h = {}
    correct = 0
    total = 0
    for h, grp in sorted(by_h.items()):
        grp = sorted(grp, key=lambda r: int(r["P"]))
        r_h = eval_linear(base_model["cpu_restore"], h)
        pts = []
        for r in grp:
            p = int(r["P"])
            m_h = eval_linear(base_model["htp_to_cpu"], h + p)
            cc_meas = r_h + float(r["cpu_prefill_median_ms"])
            hc_meas = float(r["htp_prefill_median_ms"]) + m_h
            cc_pred = r_h + eval_linear(cpu_term, p, h)
            hc_pred = eval_linear(htp_term, p, h) + m_h
            meas_route = "htp" if hc_meas < cc_meas else "cpu"
            pred_route = "htp" if hc_pred < cc_pred else "cpu"
            if meas_route == pred_route:
                correct += 1
            total += 1
            pts.append({
                "P": p,
                "cpu_pre_meas": round(float(r["cpu_prefill_median_ms"]), 1),
                "htp_pre_meas": round(float(r["htp_prefill_median_ms"]), 1),
                "cc_meas": round(cc_meas, 1),
                "hc_meas": round(hc_meas, 1),
                "cc_pred": round(cc_pred, 1),
                "hc_pred": round(hc_pred, 1),
                "meas_route": meas_route,
                "pred_route": pred_route,
            })

        def _interp_cross(key_a: str, key_b: str) -> float | None:
            for a, b in zip(pts, pts[1:]):
                da = a[key_a] - a[key_b]
                db = b[key_a] - b[key_b]
                if da <= 0 and db > 0:
                    frac = (-da) / (db - da)
                    return round(a["P"] + frac * (b["P"] - a["P"]), 2)
            return None

        per_h[str(h)] = {
            "meas_first_htp_P": next((x["P"] for x in pts if x["meas_route"] == "htp"), None),
            "pred_first_htp_P": next((x["P"] for x in pts if x["pred_route"] == "htp"), None),
            "meas_route_crossing_P": _interp_cross("cc_meas", "hc_meas"),
            "pred_route_crossing_P": _interp_cross("cc_pred", "hc_pred"),
            "meas_pure_prefill_first_htp_P": next((x["P"] for x in pts if x["htp_pre_meas"] < x["cpu_pre_meas"]), None),
            "points": pts,
        }
    return {
        "short_p_route_accuracy": f"{correct}/{total}",
        "short_p_route_accuracy_pct": round(correct / total * 100.0, 2),
        "by_H": per_h,
    }


def eval_cohort_offline(route_model: dict, kv_model: dict, fixtures_path: Path) -> dict:
    corpus = json.loads(fixtures_path.read_text())
    out = {}
    for kv_enabled in (False, True):
        cc_cnt = 0
        hc_cnt = 0
        hc_compact_cnt = 0
        per_case = []
        for case in corpus["cases"]:
            c_cc = 0
            c_hc = 0
            for t in case["turns"][1:]:
                dec = plan_joint_route_kv(
                    route_model,
                    kv_model,
                    h_eff=int(t["H"]),
                    p=int(t["P"]),
                    expected_d=int(t["D"]),
                    htp_allowed=True,
                    kv_enabled=kv_enabled,
                )
                if dec["route"] == "cpu":
                    cc_cnt += 1
                    c_cc += 1
                else:
                    hc_cnt += 1
                    c_hc += 1
                    if dec["kv_plan"] and dec["kv_plan"].get("admit"):
                        hc_compact_cnt += 1
            per_case.append({"id": case["id"], "warm_turns": len(case["turns"]) - 1, "cc": c_cc, "hc": c_hc})
        out[f"kv_{'on' if kv_enabled else 'off'}"] = {
            "warm_turns": cc_cnt + hc_cnt,
            "cc_turns": cc_cnt,
            "hc_turns": hc_cnt,
            "hc_compact_turns": hc_compact_cnt,
            "per_case": per_case,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal-dir", type=Path, default=DEFAULT_CAL_DIR)
    parser.add_argument("--infer-hc-dir", type=Path, default=DEFAULT_INFER_HC_DIR)
    parser.add_argument("--base-costs", type=Path, default=DEFAULT_BASE_COSTS)
    parser.add_argument("--cold-costs", type=Path, default=DEFAULT_COLD_COSTS)
    parser.add_argument("--kv-costs", type=Path, default=DEFAULT_KV_COSTS)
    parser.add_argument("--dt-confirm", type=Path, default=DEFAULT_DT_CONFIRM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cal_dir = args.cal_dir.resolve()
    infer_dir = args.infer_hc_dir.resolve()
    grid_rows = list(csv.DictReader((cal_dir / "grid-summary.csv").open()))
    base_cfg = json.loads(args.base_costs.resolve().read_text())
    base_model = base_cfg["model"]
    dt_summary = json.loads(args.dt_confirm.resolve().read_text())
    kv_model = json.loads(args.kv_costs.resolve().read_text())
    for k in ("htp_prior_tax", "evict"):
        kv_model[k].setdefault("slope_ms_per_token", 0.0)

    cpu_rows = [
        (int(r["H"]), int(r["P"]), float(r["cpu_prefill_median_ms"]), 1.0 / (float(r["cpu_prefill_median_ms"]) ** 2))
        for r in grid_rows
        if int(r["P"]) <= 256
    ]
    cpu_term = make_term(cpu_rows, (0.0, 40.0, 1.0), "1/y^2 (P<=256)")

    infer_csv = infer_dir / "infer-hc-samples.csv"
    if infer_csv.is_file():
        infer_rows = list(csv.DictReader(infer_csv.open()))
        h_rows = (
            [(int(r["H"]), int(r["P"]), float(r["htp_prefill_median_ms"]), 1.0 / (float(r["htp_prefill_median_ms"]) ** 2)) for r in grid_rows]
            + [(int(r["H"]), int(r["P"]), float(r["htp_prefill_ms"]), 1.0 / (float(r["htp_prefill_ms"]) ** 2)) for r in infer_rows]
        )
        htp_term = make_term(h_rows, (0.0, 12.0, 1.0), "1/y^2 (36-pt stage_sweep + 52-pt infer hc_full)")
        cpu_to_htp_term = fit_1d_nonneg_ols(
            [float(r["H"]) for r in infer_rows],
            [float(r["prepare_ms"]) for r in infer_rows],
        )
        htp_to_cpu_term = fit_1d_nonneg_ols(
            [float(int(r["H"]) + int(r["P"])) for r in infer_rows],
            [float(r["handoff_ms"]) for r in infer_rows],
        )
        model_id = "qwen3-8b-q4_0-6t-ob1024-wls-2d-infer-hc"
        selected_name = "combined_stage1_plus_infer_hc_bigcore1_plus_cpu_p_le_256"
        provenance_src = str(infer_dir.relative_to(ROOT))
    else:
        h_rows = [
            (int(r["H"]), int(r["P"]), float(r["htp_prefill_median_ms"]), 1.0 / (float(r["htp_prefill_median_ms"]) ** 2))
            for r in grid_rows
        ]
        htp_term = make_term(h_rows, (0.0, 12.0, 1.0), "1/y^2 (36-pt 2D grid)")
        cpu_to_htp_term = base_model["cpu_to_htp"]
        htp_to_cpu_term = base_model["htp_to_cpu"]
        model_id = "qwen3-8b-q4_0-6t-ob1024-wls-2d-optA-cpu256"
        selected_name = "optionA_full36_htp_plus_cpu_p_le_256"
        provenance_src = str(cal_dir.relative_to(ROOT))

    cal_eval = eval_calendar(htp_term, dt_summary)
    cross_eval = eval_short_p_crossovers(grid_rows, base_model, cpu_term, htp_term)

    new_model = {
        "kind": "linear_hp_v1",
        "model_id": model_id,
        "routing_p_min": 16,
        "h_min": 512,
        "h_max": 4096,
        "p_min": 16,
        "p_max": 1024,
        "formula": "CC=R(H)+C(P,H); HC=N(P,H)+M(H+P); common CPU decode excluded",
        "preparation": "always prepare; ready only; CPU-to-HTP preparation hidden by actual tool gap",
        "extrapolation": "2D WLS (1, P, H/1000*P) across H=512..4096, P=16..1024 with physical slope bounds",
        "cpu_threads": 6,
        "htp_host_threads": 4,
        "opbatch": 1024,
        "opqueue": 16,
        "htp_big_cores": True,
        "cpu_restore": base_model["cpu_restore"],
        "cpu_to_htp": cpu_to_htp_term,
        "cpu_prefill": cpu_term,
        "htp_prefill": htp_term,
        "htp_to_cpu": htp_to_cpu_term,
        "provenance": {
            "calibration_dir": str(cal_dir.relative_to(ROOT)),
            "htp_infer_calibration_dir": provenance_src,
            "handoff_and_restore_source": str(args.base_costs.relative_to(ROOT)),
            "cold_costs_verified": str(args.cold_costs.relative_to(ROOT)),
            "build_id": "t6-final-ob1024-7b317742c4334e85b3bd409e2a221faf",
            "git_commit": "730ad251fcbe519de639a9af418705f66d992fed",
            "selected_candidate": selected_name,
        },
    }

    args.output.resolve().write_text(json.dumps({"model": new_model}, indent=2, ensure_ascii=False) + "\n")

    offline_eval = {
        "toolsandbox_10": eval_cohort_offline(new_model, kv_model, ROOT / "results/20261124-toolsandbox-10cases/fixtures.json"),
        "droidtask_10": eval_cohort_offline(new_model, kv_model, ROOT / "results/20261124-droidtask-10cases/fixtures.json"),
        "aclow_10": eval_cohort_offline(new_model, kv_model, ROOT / "results/20261124-aclow-10cases/fixtures.json"),
    }

    report = {
        "selected_candidate": selected_name,
        "cpu_prefill": {k: v for k, v in cpu_term.items() if k != "samples"},
        "htp_prefill": {k: v for k, v in htp_term.items() if k != "samples"},
        "cpu_to_htp": cpu_to_htp_term,
        "htp_to_cpu": htp_to_cpu_term,
        "calendar_t1_t5": cal_eval,
        "short_p_crossovers": cross_eval,
        "offline_routing": offline_eval,
    }
    (cal_dir / "validation-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
