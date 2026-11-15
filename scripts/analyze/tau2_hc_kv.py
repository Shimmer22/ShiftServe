"""Analyze Qwen3-1.7B tau2-bench HC + KV Reduce campaign measurements."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def flatten_measurements(campaign_dir: Path):
    rows = []
    free_gens = {}
    for raw_path in sorted((campaign_dir / "measurements").glob("*.json")):
        if raw_path.name.endswith(".raw.json"):
            continue
        meta = json.loads(raw_path.read_text())
        req_meta = meta["meta"]
        resp = meta["response"]
        for r in resp["results"]:
            cond = r["condition"]
            variant = (
                "cpu_baseline"
                if (req_meta["service"] == "baseline_a" and cond == "cpu")
                else req_meta["variant_name"]
            )
            flat = {
                "service": req_meta["service"],
                "variant": variant,
                "gap_mode": req_meta["gap_mode"],
                "boundary": req_meta["boundary"],
                "domain": req_meta["domain"],
                "task_id": req_meta["task_id"],
                "round": req_meta["round"],
                "condition": cond,
                "route": r["route"],
                "availability": r["availability"],
                "H": r["H"],
                "P": r["P"],
                "D": r["D"],
                "decode_n_kv": r.get("decode_n_kv"),
                "requested_gap_ms": r["requested_gap_ms"],
                "actual_gap_ms": r["actual_gap_ms"],
                "prepare_ms": r["prepare_ms"],
                "wait_ms": r["wait_ms"],
                "restore_ms": r["restore_ms"],
                "prefill_ms": r["prefill_ms"],
                "evict_ms": r.get("evict_ms", 0.0),
                "handoff_ms": r["handoff_ms"],
                "handoff_bytes": r.get("handoff_bytes", 0),
                "decode_ms": r["decode_ms"],
                "decode_ms_per_tok": r["decode_ms"] / r["D"] if r["D"] else 0.0,
                "post_return_ms": r["post_return_ms"],
                "vm_rss_kib": r["resources_after"]["VmRSS_kib"],
                "vm_swap_kib": r["resources_after"]["VmSwap_kib"],
                "major_faults_delta": r["resources_after"]["major_faults"] - r["resources_before"]["major_faults"],
            }
            rows.append(flat)
            if r.get("free_generation") and req_meta["gap_mode"] == "ready":
                free_gens[(req_meta["boundary"], variant)] = r["free_generation"]
    return rows, free_gens


def summarize(campaign_dir: Path):
    rows, free_gens = flatten_measurements(campaign_dir)
    if not rows:
        raise RuntimeError(f"No measurements found under {campaign_dir / 'measurements'}")

    ready_rows = [r for r in rows if r["gap_mode"] == "ready"]
    real_gap_rows = [r for r in rows if r["gap_mode"] == "real_gap"]

    variants = ["cpu_baseline", "hc_baseline", "hc_prefill_prior", "hc_recent"]
    totals = {}
    for v in variants:
        v_rows = [r for r in ready_rows if r["variant"] == v]
        if not v_rows:
            continue
        rounds = sorted({r["round"] for r in v_rows})
        by_round = [
            sum(r["post_return_ms"] for r in v_rows if r["round"] == rnd) / 1000.0
            for rnd in rounds
        ]
        by_round_dec = [
            sum(r["decode_ms"] for r in v_rows if r["round"] == rnd) / 1000.0
            for rnd in rounds
        ]
        by_round_pre = [
            sum(r["prefill_ms"] for r in v_rows if r["round"] == rnd) / 1000.0
            for rnd in rounds
        ]
        by_round_hnd = [
            sum(r["handoff_ms"] for r in v_rows if r["round"] == rnd) / 1000.0
            for rnd in rounds
        ]
        totals[v] = {
            "rounds": len(rounds),
            "post_return_mean_s": statistics.mean(by_round),
            "prefill_mean_s": statistics.mean(by_round_pre),
            "handoff_mean_s": statistics.mean(by_round_hnd),
            "decode_mean_s": statistics.mean(by_round_dec),
            "mean_handoff_bytes": statistics.mean(r["handoff_bytes"] for r in v_rows),
            "mean_decode_n_kv": statistics.mean(r["decode_n_kv"] for r in v_rows),
            "max_vm_rss_kib": max(r["vm_rss_kib"] for r in v_rows),
            "max_vm_swap_kib": max(r["vm_swap_kib"] for r in v_rows),
            "total_major_faults_delta": sum(r["major_faults_delta"] for r in v_rows),
        }

    per_boundary = []
    boundaries = sorted({r["boundary"] for r in ready_rows})
    for b in boundaries:
        entry = {"boundary": b}
        b_sample = next(r for r in ready_rows if r["boundary"] == b)
        entry.update({"domain": b_sample["domain"], "H": b_sample["H"], "P": b_sample["P"], "D": b_sample["D"]})
        for v in variants:
            vr = [r for r in ready_rows if r["boundary"] == b and r["variant"] == v]
            if not vr:
                continue
            entry[v] = {
                "post_return_ms": statistics.mean(r["post_return_ms"] for r in vr),
                "prefill_ms": statistics.mean(r["prefill_ms"] for r in vr),
                "evict_ms": statistics.mean(r["evict_ms"] for r in vr),
                "handoff_ms": statistics.mean(r["handoff_ms"] for r in vr),
                "handoff_bytes": vr[0]["handoff_bytes"],
                "decode_ms": statistics.mean(r["decode_ms"] for r in vr),
                "decode_ms_per_tok": statistics.mean(r["decode_ms_per_tok"] for r in vr),
                "decode_n_kv": vr[0]["decode_n_kv"],
            }
        # Quality comparison against cpu_baseline
        cpu_fg = free_gens.get((b, "cpu_baseline"))
        if cpu_fg:
            q = {}
            for v in ["hc_baseline", "hc_prefill_prior", "hc_recent"]:
                fg = free_gens.get((b, v))
                if not fg:
                    continue
                q[v] = {
                    "top1_match": fg["top1_first"] == cpu_fg["top1_first"],
                    "exact_match": fg["tokens"] == cpu_fg["tokens"],
                    "lcp_tokens": common_prefix_len(fg["tokens"], cpu_fg["tokens"]),
                    "total_tokens": len(cpu_fg["tokens"]),
                    "text_preview": fg["text"][:160],
                }
            entry["free_generation_vs_cpu"] = q
        per_boundary.append(entry)

    real_gap_summary = []
    for r in sorted(real_gap_rows, key=lambda x: (x["boundary"], x["round"])):
        real_gap_summary.append({
            "boundary": r["boundary"],
            "requested_gap_ms": r["requested_gap_ms"],
            "prepare_ms": r["prepare_ms"],
            "wait_ms": r["wait_ms"],
            "prefill_ms": r["prefill_ms"],
            "evict_ms": r["evict_ms"],
            "handoff_ms": r["handoff_ms"],
            "decode_ms": r["decode_ms"],
            "post_return_ms": r["post_return_ms"],
        })

    # Write CSV
    csv_path = campaign_dir / "timings.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "campaign_dir": str(campaign_dir.relative_to(ROOT)),
        "measurements_count": len(rows),
        "totals_ready": totals,
        "per_boundary_ready": per_boundary,
        "real_gap_hc_prefill_prior": real_gap_summary,
    }
    if "hc_baseline" in totals and "hc_prefill_prior" in totals:
        summary["hc_kv_vs_hc_baseline"] = {
            "post_return_speedup": totals["hc_baseline"]["post_return_mean_s"] / totals["hc_prefill_prior"]["post_return_mean_s"],
            "decode_speedup": totals["hc_baseline"]["decode_mean_s"] / totals["hc_prefill_prior"]["decode_mean_s"],
            "handoff_speedup": totals["hc_baseline"]["handoff_mean_s"] / totals["hc_prefill_prior"]["handoff_mean_s"],
            "handoff_bytes_ratio": totals["hc_baseline"]["mean_handoff_bytes"] / totals["hc_prefill_prior"]["mean_handoff_bytes"],
        }
    if "cpu_baseline" in totals and "hc_prefill_prior" in totals:
        summary["hc_kv_vs_cpu_baseline"] = {
            "post_return_speedup": totals["cpu_baseline"]["post_return_mean_s"] / totals["hc_prefill_prior"]["post_return_mean_s"],
            "decode_speedup": totals["cpu_baseline"]["decode_mean_s"] / totals["hc_prefill_prior"]["decode_mean_s"],
        }

    (campaign_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign_dir", type=Path)
    args = p.parse_args()
    summary = summarize(args.campaign_dir.resolve())
    print(json.dumps(summary["totals_ready"], indent=2))
    if "hc_kv_vs_hc_baseline" in summary:
        print("HC+PrefillPrior vs HC Baseline:", json.dumps(summary["hc_kv_vs_hc_baseline"]))
    if "hc_kv_vs_cpu_baseline" in summary:
        print("HC+PrefillPrior vs CPU Baseline:", json.dumps(summary["hc_kv_vs_cpu_baseline"]))


if __name__ == "__main__":
    main()
