"""Fit and evaluate joint KV-Reduce admission cost tables from quantitative curve scans."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]


def _affine_fit_nonneg(points):
    """Fit non-negative affine y = max(0, a + b*x) via OLS with boundary clamp."""
    n = len(points)
    if n == 0:
        return {"intercept_ms": 0.0, "slope_ms_per_token": 0.0}
    if n == 1 or len({x for x, _ in points}) < 2:
        return {"intercept_ms": max(0.0, sum(y for _, y in points) / n), "slope_ms_per_token": 0.0}
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    candidates = [(sy / n, 0.0), (0.0, max(0.0, sxy / sxx if sxx else 0.0))]
    denom = n * sxx - sx * sx
    if denom != 0:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        if intercept >= 0 and slope >= 0:
            candidates.append((intercept, slope))
    a, b = min(candidates, key=lambda pair: sum((pair[0] + pair[1] * x - y) ** 2 for x, y in points))
    return {"intercept_ms": max(0.0, a), "slope_ms_per_token": max(0.0, b)}


def fit_kv_cost_model(curve_summary: dict, model_id: str = "qwen3-1.7b-q4_0") -> dict:
    h_rows = curve_summary["h_sweep"]
    k_rows = curve_summary["k_sweep"]
    p_rows = curve_summary["p_sweep"]
    cpu_probes = curve_summary.get("cpu_probes", [])

    # 1. Decode ms/tok vs n_kv (combining full points from h_sweep + reduced points from k_sweep)
    dec_points = []
    for r in h_rows:
        if "hc_nocb" in r:
            dec_points.append((r["hc_nocb"]["decode_n_kv"], r["hc_nocb"]["decode_ms_per_tok"]))
    for r in k_rows:
        dec_points.append((r["decode_n_kv"], r["decode_ms_per_tok"]))
    decode_fit = _affine_fit_nonneg(dec_points)

    # 2. Handoff ms vs n_kv (for HC HTP->CPU transfer)
    hnd_points = []
    for r in h_rows:
        if "hc_nocb" in r:
            hnd_points.append((r["hc_nocb"]["decode_n_kv"], r["hc_nocb"]["handoff_ms"]))
    for r in k_rows:
        hnd_points.append((r["decode_n_kv"], r["handoff_ms"]))
    handoff_fit = _affine_fit_nonneg(hnd_points)

    # 3. Evict + compact ms vs H_total
    evict_points = [(r["H"] + r["P"], r["hc_prior"]["evict_ms"]) for r in h_rows if "hc_prior" in r and r["H"] >= 1024]
    evict_fit = _affine_fit_nonneg(evict_points)

    # 4. cb_eval tax on HTP vs P (across H >= 2048 where collection is meaningful)
    htp_tax_points = [(r["P"], max(0.0, r["cb_eval_tax_ms"])) for r in p_rows]
    # Add H>=2048 points at P=256 from h_sweep
    h_tax_256 = [max(0.0, r["cb_eval_tax_ms"]) for r in h_rows if r["H"] >= 2048 and "cb_eval_tax_ms" in r]
    if h_tax_256:
        htp_tax_points.append((256, statistics.mean(h_tax_256)))
    htp_prior_tax_fit = _affine_fit_nonneg(htp_tax_points)

    # 5. cb_eval tax on CPU from cpu_probes (paired turn1 H=4020, 8116 where kv_reduce ran)
    cpu_by_kv = {}
    for blk in cpu_probes:
        kv = blk["meta"]["kv_reduce"]
        for row in blk["rows"]:
            if row["h_target"] >= 4096:
                cpu_by_kv.setdefault((kv, row["h_target"]), []).append(row["turn1_prompt_ms"])
    cpu_taxes = []
    for ht in (4096, 8192):
        if (0, ht) in cpu_by_kv and (1, ht) in cpu_by_kv:
            cpu_taxes.append(max(0.0, statistics.mean(cpu_by_kv[(1, ht)]) - statistics.mean(cpu_by_kv[(0, ht)])))
    cpu_prior_tax_ms = statistics.mean(cpu_taxes) if cpu_taxes else 610.0

    # 6. Prefill + restore models (for joint CC vs HC awareness)
    cpu_restore_fit = _affine_fit_nonneg([(r["H"], r["cpu"]["restore_ms"]) for r in h_rows if "cpu" in r])
    cpu_prefill_h4k_fit = _affine_fit_nonneg(
        [(r["P"], r["cpu_prefill_ms"]) for r in p_rows if r.get("cpu_prefill_ms") is not None]
        + [(256, next(r["cpu"]["prefill_ms"] for r in h_rows if r["H"] == 4096))]
    )
    htp_prefill_h4k_fit = _affine_fit_nonneg(
        [(r["P"], r["htp_nocb_prefill_ms"]) for r in p_rows]
        + [(256, next(r["hc_nocb"]["prefill_ms"] for r in h_rows if r["H"] == 4096))]
    )

    model = {
        "kind": "kv_reduce_cost_v1",
        "model_id": model_id,
        "source_curve": curve_summary.get("curve_dir", ""),
        "expected_decode_tokens": 96,
        "quality_k_min": 1536,
        "quality_keep_ratio_min": 0.25,
        "align_pad": 256,
        "min_evict_savings_tokens": 512,
        "min_gain_ms": 100.0,
        "candidates": [1536, 2048, 2560, 3072, 4096],
        "decode_per_tok": decode_fit,
        "handoff": handoff_fit,
        "evict": evict_fit,
        "htp_prior_tax": htp_prior_tax_fit,
        "cpu_prior_tax": {"intercept_ms": cpu_prior_tax_ms, "slope_ms_per_token": 0.0},
        "cpu_restore": cpu_restore_fit,
        "cpu_prefill_p_at_4k": cpu_prefill_h4k_fit,
        "htp_prefill_p_at_4k": htp_prefill_h4k_fit,
    }
    return model


def _eval_line(fit: dict, x: float) -> float:
    return max(0.0, fit["intercept_ms"] + fit["slope_ms_per_token"] * x)


def _pad_align(n: int, align: int) -> int:
    if align <= 1:
        return n
    return ((n + align - 1) // align) * align


def plan_kv_reduce(
    model: dict,
    h: int,
    p: int,
    expected_d: int,
    is_htp: bool,
    h_logical: int | None = None,
    pinned_tokens: int = 0,
) -> dict:
    total = h + p
    logical_h = max(h, h_logical) if (h_logical is not None and h_logical >= 0) else h
    logical_total = logical_h + p
    k_min = int(model.get("quality_k_min", 1536))
    ratio_min = float(model.get("quality_keep_ratio_min", 0.25))
    align = int(model.get("align_pad", 256))
    min_evict = int(model.get("min_evict_savings_tokens", 512))
    min_gain = float(model.get("min_gain_ms", 100.0))
    use_hnd_saving = bool(is_htp and model.get("handoff_saving_enabled", False))

    k_floor = max(k_min, pinned_tokens, _pad_align(int(math.ceil(ratio_min * logical_total)), align))
    if total <= k_floor + min_evict:
        return {
            "admit": False,
            "keep_tokens": total,
            "expected_gain_ms": 0.0,
            "reason": "below_quality_or_evict_floor",
            "k_floor": k_floor,
            "H_total": total,
        }

    candidates = sorted({
        c for c in list(model.get("candidates", [1536, 2048, 2560, 3072, 4096])) + [k_floor + off for off in (0, 512, 1024, 2048)]
        if c >= k_floor and c + min_evict <= total
    })
    if not candidates and k_floor + min_evict <= total:
        candidates = [k_floor]
    if not candidates:
        return {
            "admit": False,
            "keep_tokens": total,
            "expected_gain_ms": 0.0,
            "reason": "no_valid_keep_candidate",
            "k_floor": k_floor,
            "H_total": total,
        }

    tax_ms = _eval_line(model["htp_prior_tax"], p) if is_htp else _eval_line(model["cpu_prior_tax"], p)
    evict_ms = _eval_line(model["evict"], total)
    dec_full = _eval_line(model["decode_per_tok"], total)
    hnd_full = _eval_line(model["handoff"], total) if use_hnd_saving else 0.0

    best = None
    for k in candidates:
        dec_k = _eval_line(model["decode_per_tok"], k)
        hnd_k = _eval_line(model["handoff"], k) if use_hnd_saving else 0.0
        hnd_saving = hnd_full - hnd_k
        dec_saving = expected_d * (dec_full - dec_k)
        gain = hnd_saving + dec_saving - tax_ms - evict_ms
        if best is None or gain > best["expected_gain_ms"]:
            best = {
                "keep_tokens": k,
                "expected_gain_ms": gain,
                "handoff_saving_ms": hnd_saving,
                "decode_saving_ms": dec_saving,
                "prior_tax_ms": tax_ms,
                "evict_ms": evict_ms,
                "k_floor": k_floor,
                "H_total": total,
            }

    assert best is not None
    admit = best["expected_gain_ms"] > min_gain
    best["admit"] = admit
    best["reason"] = "cost_table_admitted" if admit else "negative_net_gain"
    return best


def _eval_linear(term: dict, x: float, h_eff: float = 0.0) -> float:
    return max(
        0.0,
        float(term["intercept_ms"])
        + float(term["slope_ms_per_token"]) * float(x)
        + float(term.get("hp_slope_ms_per_1k_tokens") or 0.0) * (float(h_eff) / 1000.0) * float(x),
    )


def plan_joint_route_kv(
    route_model: dict | None,
    kv_model: dict | None,
    h_eff: int,
    p: int,
    expected_d: int,
    htp_allowed: bool = True,
    kv_enabled: bool = True,
    default_route: str = "cpu",
    h_logical: int | None = None,
    pinned_tokens: int = 0,
) -> dict:
    """Jointly minimize total predicted latency over (backend in {cpu, htp}, KV reduce K*)."""
    if h_eff < 0 or p < 0 or expected_d < 0 or pinned_tokens < 0:
        raise ValueError("negative token count")
    total = h_eff + p
    kv_valid = bool(kv_model and kv_model.get("kind") == "kv_reduce_cost_v1")
    route_valid = bool(route_model and route_model.get("kind") == "linear_hp_v1")

    cc_ms = 0.0
    hc_ms = 0.0
    extrapolated = False
    if route_valid:
        cc_ms = _eval_linear(route_model["cpu_restore"], h_eff) + _eval_linear(route_model["cpu_prefill"], p, h_eff)
        hnd_full = _eval_linear(route_model["htp_to_cpu"], total)
        hc_ms = _eval_linear(route_model["htp_prefill"], p, h_eff) + hnd_full
        p_max = int(route_model.get("p_max", 0))
        extrapolated = (
            h_eff < int(route_model.get("h_min", 0))
            or total > int(route_model.get("h_max", 0))
            or p < int(route_model.get("p_min", 0))
            or (p_max > 0 and p > p_max)
        )
    elif kv_valid:
        hnd_full = _eval_linear(kv_model["handoff"], total)
        hc_ms = hnd_full

    dec_full_cost = float(expected_d) * _eval_linear(kv_model["decode_per_tok"], total) if kv_valid else 0.0
    cpu_no_kv_total = cc_ms + dec_full_cost
    htp_no_kv_total = hc_ms + dec_full_cost
    cpu_best_total = cpu_no_kv_total
    htp_best_total = htp_no_kv_total
    cpu_kv_plan = None
    htp_kv_plan = None

    if kv_enabled and kv_valid:
        cpu_kv_plan = plan_kv_reduce(kv_model, h_eff, p, expected_d, is_htp=False, h_logical=h_logical, pinned_tokens=pinned_tokens)
        if cpu_kv_plan["admit"]:
            cpu_best_total = cpu_no_kv_total - cpu_kv_plan["expected_gain_ms"]
        htp_kv_plan = plan_kv_reduce(kv_model, h_eff, p, expected_d, is_htp=True, h_logical=h_logical, pinned_tokens=pinned_tokens)
        if htp_kv_plan["admit"]:
            htp_best_total = htp_no_kv_total - htp_kv_plan["expected_gain_ms"]

    def _result(route: str, reason: str, kv_plan: dict | None, selected_total: float) -> dict:
        return {
            "route": route,
            "reason": reason,
            "extrapolated": extrapolated,
            "h_eff": h_eff,
            "p": p,
            "expected_d": expected_d,
            "cc_ms": cc_ms,
            "hc_ms": hc_ms,
            "cpu_no_kv_total_ms": cpu_no_kv_total,
            "htp_no_kv_total_ms": htp_no_kv_total,
            "cpu_best_total_ms": cpu_best_total,
            "htp_best_total_ms": htp_best_total,
            "selected_total_ms": selected_total,
            "kv_plan": kv_plan,
        }

    if not htp_allowed:
        return _result(
            default_route,
            "fixed_backend",
            htp_kv_plan if default_route == "htp" else cpu_kv_plan,
            htp_best_total if default_route == "htp" else cpu_best_total,
        )

    if route_valid and p < int(route_model.get("routing_p_min", 16)):
        return _result("cpu", "below_routing_p_min", cpu_kv_plan, cpu_best_total)

    if htp_best_total < cpu_best_total:
        reason = "joint_route_kv_cost" if (kv_enabled and kv_valid and htp_kv_plan and htp_kv_plan["admit"]) else "linear_hp_cost"
        return _result("htp", reason, htp_kv_plan, htp_best_total)

    reason = "joint_route_kv_cost" if (kv_enabled and kv_valid and cpu_kv_plan and cpu_kv_plan["admit"]) else "linear_hp_cost"
    return _result("cpu", reason, cpu_kv_plan, cpu_best_total)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve-summary", type=Path, default=ROOT / "results/20261115-qwen3-1_7b-kv-curve/summary.json")
    p.add_argument("--model-id", default="qwen3-1.7b-q4_0")
    p.add_argument("--output", type=Path, default=ROOT / "configs/qwen3-1_7b-kv-reduce-cost.json")
    args = p.parse_args()

    summary = json.loads(args.curve_summary.resolve().read_text())
    model = fit_kv_cost_model(summary, model_id=args.model_id)
    args.output.resolve().write_text(json.dumps(model, indent=2) + "\n")
    print("Wrote cost table:", args.output)
    for is_htp in [False, True]:
        label = "HC" if is_htp else "CPU"
        print(f"=== Admission sanity ({label}, D_exp={model['expected_decode_tokens']}) ===")
        for h, pval, d in [(512, 256, 96), (1024, 256, 96), (2048, 256, 96), (4096, 256, 32), (4096, 256, 96), (6144, 256, 96), (8192, 256, 96), (7960, 470, 688)]:
            dec = plan_kv_reduce(model, h, pval, d, is_htp)
            print(f"  H={h:4d} P={pval:3d} D={d:3d} -> admit={str(dec['admit']):5s} K={dec['keep_tokens']:4d} gain={dec['expected_gain_ms']:7.1f}ms ({dec['reason']})")


if __name__ == "__main__":
    main()
