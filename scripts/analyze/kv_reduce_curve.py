"""Analyze Qwen3-1.7B KV Reduce time, history-length, keep-budget, and quality curves."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def ols_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return {"intercept": ys[0] if ys else 0.0, "slope": 0.0, "r2": 0.0}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return {"intercept": intercept, "slope": slope, "r2": r2}


def check_boundary_quality(boundary: int, text: str, cpu_fg: dict | None):
    checks = {}
    if boundary == 0:
        checks = {
            "email": "sophia.martin4832@example.com" in text,
            "address": "760 Elm" in text and "77034" in text,
            "orders": "#W1603792" in text and "#W1092119" in text,
            "card": "3292" in text,
        }
    elif boundary == 1:
        checks = {
            "order_W6397299": "#W6397299" in text,
            "next_order_W6231698": "#W6231698" in text,
            "tool_call": "<tool_call>" in text and "get_order_details" in text,
        }
    elif boundary == 2:
        checks = {
            "flight_HAT023": "HAT023" in text,
            "price_or_tool": ("364" in text) or ("search_direct_flight" in text),
        }
    elif boundary == 3:
        checks = {
            "reservation_Q0ZF0J": "Q0ZF0J" in text,
            "tool_call": "<tool_call>" in text and "get_reservation_details" in text,
        }
    elif boundary == 4:
        checks = {
            "phone": "555-123-2002" in text,
            "plan": "P1001" in text,
        }
    elif boundary == 5:
        checks = {
            "mms_or_roaming": ("roaming" in text.lower() or "mms" in text.lower()) and "line" in text.lower(),
        }
    passed = all(checks.values()) if checks else True
    return {
        "checks": checks,
        "all_key_facts_passed": passed,
        "text_preview": text[:180],
    }


def load_baseline_freegen(baseline_campaign: Path):
    fgs = {}
    m = baseline_campaign / "measurements"
    if not m.exists():
        return fgs
    for b in range(6):
        p = m / f"baseline_a-ready-r00-b{b:02d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d["response"]["results"]:
            if r["condition"] == "cpu" and r.get("free_generation"):
                fgs[b] = r["free_generation"]
    return fgs


def summarize(curve_dir: Path, baseline_campaign: Path | None = None):
    meas_dir = curve_dir / "measurements"
    grid_rows = []
    quality_rows = []

    baseline_fgs = load_baseline_freegen(baseline_campaign) if baseline_campaign else {}

    for path in sorted(meas_dir.glob("*.json")):
        if path.name.endswith(".raw.json"):
            continue
        rec = json.loads(path.read_text())
        meta = rec["meta"]
        kind = meta["kind"]
        if kind == "cpu_probe":
            continue
        resp = rec["response"]
        for row in resp["results"]:
            base = {
                "kind": kind,
                "service": meta["service"],
                "round": meta["round"],
                "condition": row["condition"],
                "route": row["route"],
                "H": row["H"],
                "P": row["P"],
                "D": row["D"],
                "kv_keep": meta.get("kv_keep", row["H"] + row["P"]),
                "kv_tail": meta.get("kv_tail", 0),
                "decode_n_kv": row["decode_n_kv"],
                "restore_ms": row["restore_ms"],
                "prepare_ms": row["prepare_ms"],
                "prefill_ms": row["prefill_ms"],
                "evict_ms": row.get("evict_ms", 0.0),
                "handoff_ms": row["handoff_ms"],
                "handoff_bytes": row.get("handoff_bytes", 0),
                "decode_ms": row["decode_ms"],
                "decode_ms_per_tok": row["decode_ms"] / row["D"] if row["D"] else 0.0,
                "post_return_ms": row["post_return_ms"],
                "vm_rss_kib": row["resources_after"]["VmRSS_kib"],
                "vm_swap_kib": row["resources_after"]["VmSwap_kib"],
            }
            if kind == "quality_k":
                b = meta["boundary"]
                fg = row["free_generation"]
                cpu_fg = baseline_fgs.get(b)
                qcheck = check_boundary_quality(b, fg["text"], cpu_fg)
                ratio = row["decode_n_kv"] / float(row["H"] + row["P"])
                base.update({
                    "boundary": b,
                    "domain": meta["domain"],
                    "keep_ratio": round(ratio, 4),
                    "top1_match_cpu": (fg["top1_first"] == cpu_fg["top1_first"]) if cpu_fg else None,
                    "lcp_tokens_cpu": common_prefix_len(fg["tokens"], cpu_fg["tokens"]) if cpu_fg else None,
                    **qcheck,
                })
                quality_rows.append(base)
            else:
                grid_rows.append(base)

    # Write CSVs
    if grid_rows:
        with (curve_dir / "grid_timings.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
            w.writeheader()
            w.writerows(grid_rows)
    if quality_rows:
        qflat = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in quality_rows]
        with (curve_dir / "quality_k_curve.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qflat[0].keys()))
            w.writeheader()
            w.writerows(qflat)

    # Summarize H-sweep (P=256)
    h_summary = []
    hs = sorted({r["H"] for r in grid_rows if r["kind"] == "h_sweep"})
    for h in hs:
        cpu_r = [r for r in grid_rows if r["kind"] == "h_sweep" and r["H"] == h and r["condition"] == "cpu"]
        hc0_r = [r for r in grid_rows if r["kind"] == "h_sweep" and r["H"] == h and r["service"] == "baseline_a" and r["condition"] == "route"]
        hc1_r = [r for r in grid_rows if r["kind"] == "h_sweep" and r["H"] == h and r["service"] == "prior_b" and r["condition"] == "route"]
        entry = {"H": h, "P": hc0_r[0]["P"] if hc0_r else 256}
        if cpu_r:
            entry["cpu"] = {
                "restore_ms": statistics.mean(r["restore_ms"] for r in cpu_r),
                "prefill_ms": statistics.mean(r["prefill_ms"] for r in cpu_r),
                "decode_ms_per_tok": statistics.mean(r["decode_ms_per_tok"] for r in cpu_r),
            }
        if hc0_r:
            entry["hc_nocb"] = {
                "prepare_ms": statistics.mean(r["prepare_ms"] for r in hc0_r),
                "prefill_ms": statistics.mean(r["prefill_ms"] for r in hc0_r),
                "handoff_ms": statistics.mean(r["handoff_ms"] for r in hc0_r),
                "handoff_bytes": hc0_r[0]["handoff_bytes"],
                "decode_n_kv": hc0_r[0]["decode_n_kv"],
                "decode_ms_per_tok": statistics.mean(r["decode_ms_per_tok"] for r in hc0_r),
            }
        if hc1_r:
            entry["hc_prior"] = {
                "prefill_ms": statistics.mean(r["prefill_ms"] for r in hc1_r),
                "evict_ms": statistics.mean(r["evict_ms"] for r in hc1_r),
                "handoff_ms": statistics.mean(r["handoff_ms"] for r in hc1_r),
                "handoff_bytes": hc1_r[0]["handoff_bytes"],
                "decode_n_kv": hc1_r[0]["decode_n_kv"],
                "decode_ms_per_tok": statistics.mean(r["decode_ms_per_tok"] for r in hc1_r),
            }
        if hc0_r and hc1_r:
            tax = entry["hc_prior"]["prefill_ms"] - entry["hc_nocb"]["prefill_ms"]
            ev = entry["hc_prior"]["evict_ms"]
            hnd_save = entry["hc_nocb"]["handoff_ms"] - entry["hc_prior"]["handoff_ms"]
            dec_save_tok = entry["hc_nocb"]["decode_ms_per_tok"] - entry["hc_prior"]["decode_ms_per_tok"]
            overhead = tax + ev - hnd_save
            entry["cb_eval_tax_ms"] = tax
            entry["handoff_saving_ms"] = hnd_save
            entry["decode_saving_ms_per_tok"] = dec_save_tok
            entry["breakeven_D_hc"] = (
                0.0 if overhead <= 0 else (overhead / dec_save_tok if dec_save_tok > 0 else None)
            )
        h_summary.append(entry)

    # Summarize P-sweep (H=4096)
    p_summary = []
    for pval in sorted({r["P"] for r in grid_rows if r["kind"] == "p_sweep"}):
        cpu_r = [r for r in grid_rows if r["kind"] == "p_sweep" and r["P"] == pval and r["condition"] == "cpu"]
        hc0_r = [r for r in grid_rows if r["kind"] == "p_sweep" and r["P"] == pval and r["service"] == "baseline_a" and r["condition"] == "route"]
        hc1_r = [r for r in grid_rows if r["kind"] == "p_sweep" and r["P"] == pval and r["service"] == "prior_b" and r["condition"] == "route"]
        if hc0_r and hc1_r:
            p0 = statistics.mean(r["prefill_ms"] for r in hc0_r)
            p1 = statistics.mean(r["prefill_ms"] for r in hc1_r)
            p_summary.append({
                "H": 4096,
                "P": pval,
                "cpu_prefill_ms": statistics.mean(r["prefill_ms"] for r in cpu_r) if cpu_r else None,
                "htp_nocb_prefill_ms": p0,
                "htp_prior_prefill_ms": p1,
                "cb_eval_tax_ms": p1 - p0,
                "cb_eval_tax_pct": (p1 - p0) / p0 * 100.0 if p0 > 0 else 0.0,
            })

    # Summarize tail-sweep
    tail_summary = []
    for tv in sorted({r["kv_tail"] for r in grid_rows if r["kind"] == "tail_sweep"}):
        tr = [r for r in grid_rows if r["kind"] == "tail_sweep" and r["kv_tail"] == tv]
        tail_summary.append({
            "tail_tokens": tv,
            "prefill_ms": statistics.mean(r["prefill_ms"] for r in tr),
            "evict_ms": statistics.mean(r["evict_ms"] for r in tr),
        })

    # Summarize K-sweep (H=8192, P=256, D=64)
    k_summary = []
    for kv in sorted({r["kv_keep"] for r in grid_rows if r["kind"] == "k_sweep"}):
        kr = [r for r in grid_rows if r["kind"] == "k_sweep" and r["kv_keep"] == kv]
        k_summary.append({
            "kv_keep": kv,
            "decode_n_kv": kr[0]["decode_n_kv"],
            "evict_ms": statistics.mean(r["evict_ms"] for r in kr),
            "handoff_bytes": kr[0]["handoff_bytes"],
            "handoff_ms": statistics.mean(r["handoff_ms"] for r in kr),
            "decode_ms": statistics.mean(r["decode_ms"] for r in kr),
            "decode_ms_per_tok": statistics.mean(r["decode_ms_per_tok"] for r in kr),
            "prefill_ms": statistics.mean(r["prefill_ms"] for r in kr),
        })

    # Fit decode_ms_per_tok(n_kv) & handoff_ms(bytes)
    dec_pts = [(r["decode_n_kv"], r["decode_ms_per_tok"]) for r in grid_rows if r["kind"] in ("k_sweep", "h_sweep") and r["condition"] == "route"]
    dec_fit = ols_fit([x[0] for x in dec_pts], [x[1] for x in dec_pts]) if dec_pts else {}
    hnd_pts = [(r["handoff_bytes"] / (1024 * 1024), r["handoff_ms"]) for r in grid_rows if r["condition"] == "route" and r["handoff_bytes"] > 0]
    hnd_fit = ols_fit([x[0] for x in hnd_pts], [x[1] for x in hnd_pts]) if hnd_pts else {}

    # Summarize CPU probes if present
    cpu_probes = []
    for path in sorted(meas_dir.glob("cpu-probe-*.json")):
        rec = json.loads(path.read_text())
        cpu_probes.append(rec)

    summary = {
        "curve_dir": str(curve_dir.relative_to(ROOT)),
        "grid_count": len(grid_rows),
        "quality_count": len(quality_rows),
        "h_sweep": h_summary,
        "p_sweep": p_summary,
        "tail_sweep": tail_summary,
        "k_sweep": k_summary,
        "fits": {
            "decode_ms_per_tok_vs_n_kv": dec_fit,
            "handoff_ms_vs_mib": hnd_fit,
        },
        "quality_k_curve": quality_rows,
        "cpu_probes": cpu_probes,
    }
    (curve_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("curve_dir", type=Path)
    p.add_argument("--baseline-campaign", type=Path, default=ROOT / "results/20261115-tau2-qwen3-1_7b-hc-kv/campaign")
    args = p.parse_args()
    s = summarize(args.curve_dir.resolve(), args.baseline_campaign.resolve())
    print("H-sweep summary:")
    for row in s["h_sweep"]:
        print(" ", json.dumps(row))
    print("K-sweep (H=8192):")
    for row in s["k_sweep"]:
        print(" ", json.dumps(row))
    print("Fits:", json.dumps(s["fits"]))


if __name__ == "__main__":
    main()
