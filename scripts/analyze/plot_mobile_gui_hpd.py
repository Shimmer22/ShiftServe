#!/usr/bin/env python3
"""Plot publication-grade (H, P, D) distribution figure for Mobile GUI Agents
(DroidTask Full 158 tasks + AndroidControl-Low 120 sampled tasks)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analyze"))
from mobile_gui_hpd_and_freeze import quantile, simulate_phone_prior


def summarize_cohort(name: str, subset: list[dict]) -> dict:
    n_turns = [len(c["turns"]) for c in subset]
    max_logs = [max(t["logical"] for t in c["turns"]) for c in subset]
    all_h = [t["H"] for c in subset for t in c["turns"]]
    warm_h = [t["H"] for c in subset for t in c["turns"][1:]]
    all_p = [t["P"] for c in subset for t in c["turns"]]
    warm_p = [t["P"] for c in subset for t in c["turns"][1:]]
    all_d = [t["D"] for c in subset for t in c["turns"]]
    s50 = [simulate_phone_prior(c, ratio=0.5) for c in subset]
    s75 = [simulate_phone_prior(c, ratio=0.75) for c in subset]
    return {
        "cohort": name,
        "tasks": len(subset),
        "total_turns": len(all_p),
        "warm_turns": len(warm_p),
        "turns_per_task": {"min": min(n_turns), "p50": quantile(n_turns, 0.5), "max": max(n_turns)},
        "max_logical_tokens": {
            "min": min(max_logs), "p25": quantile(max_logs, 0.25),
            "p50": quantile(max_logs, 0.5), "p75": quantile(max_logs, 0.75), "max": max(max_logs),
        },
        "H_warm_turns": {
            "min": min(warm_h), "p25": quantile(warm_h, 0.25),
            "p50": quantile(warm_h, 0.5), "p75": quantile(warm_h, 0.75), "max": max(warm_h),
        },
        "P_all_turns": {
            "min": min(all_p), "p25": quantile(all_p, 0.25),
            "p50": quantile(all_p, 0.5), "p75": quantile(all_p, 0.75), "max": max(all_p),
            "le_32_count": sum(1 for x in all_p if x <= 32),
            "le_32_pct": sum(1 for x in all_p if x <= 32) / len(all_p) * 100.0,
            "le_64_count": sum(1 for x in all_p if x <= 64),
            "le_64_pct": sum(1 for x in all_p if x <= 64) / len(all_p) * 100.0,
        },
        "P_warm_turns": {
            "min": min(warm_p), "p25": quantile(warm_p, 0.25),
            "p50": quantile(warm_p, 0.5), "p75": quantile(warm_p, 0.75), "max": max(warm_p),
        },
        "D_pure_action": {
            "min": min(all_d), "p25": quantile(all_d, 0.25),
            "p50": quantile(all_d, 0.5), "p75": quantile(all_d, 0.75), "max": max(all_d),
        },
        "phone_9472_prefill_prior_est": {
            "ratio_0.50": {
                "avg_triggers_per_task": sum(x[0] for x in s50) / len(s50),
                "tasks_ge_2_triggers": sum(1 for x in s50 if x[0] >= 2),
                "tasks_fit_9472": sum(1 for x in s50 if x[1]),
            },
            "ratio_0.75": {
                "avg_triggers_per_task": sum(x[0] for x in s75) / len(s75),
                "tasks_ge_2_triggers": sum(1 for x in s75 if x[0] >= 2),
                "tasks_fit_9472": sum(1 for x in s75 if x[1]),
            },
        },
    }


def plot_hpd(dt_rows: list[dict], ac_rows: list[dict], frozen_ids: set[str], out_dir: Path) -> dict:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    c_dt = "#1f77b4"
    c_ac = "#d95f02"
    c_frz = "#111111"

    dt_p = np.array([t["P"] for r in dt_rows for t in r["turns"]])
    ac_p = np.array([t["P"] for r in ac_rows for t in r["turns"]])
    dt_d = np.array([t["D"] for r in dt_rows for t in r["turns"]])
    ac_d = np.array([t["D"] for r in ac_rows for t in r["turns"]])
    dt_h_warm = np.array([t["H"] for r in dt_rows for t in r["turns"][1:]])
    ac_h_warm = np.array([t["H"] for r in ac_rows for t in r["turns"][1:]])
    dt_p_warm = np.array([t["P"] for r in dt_rows for t in r["turns"][1:]])
    ac_p_warm = np.array([t["P"] for r in ac_rows for t in r["turns"][1:]])
    dt_maxl = np.sort([max(t["logical"] for t in r["turns"]) for r in dt_rows])
    ac_maxl = np.sort([max(t["logical"] for t in r["turns"]) for r in ac_rows])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), dpi=300)
    fig.subplots_adjust(hspace=0.32, wspace=0.24, top=0.90, bottom=0.08, left=0.07, right=0.97)
    fig.suptitle(
        "Pure-Action (H, P, D) Distributions — Qwen3-8B Exact Tokenization\n"
        f"DroidTask Full ({len(dt_rows)} tasks, {len(dt_p)} turns) vs. "
        f"AndroidControl-Low ({len(ac_rows)} tasks, {len(ac_p)} turns)",
        fontsize=12.5, fontweight="bold", y=0.975,
    )

    # Panel (a): New Input P (log x histogram)
    ax = axes[0, 0]
    bins_p = np.logspace(np.log10(20), np.log10(3500), 32)
    ax.hist(dt_p, bins=bins_p, weights=np.ones_like(dt_p) / len(dt_p) * 100,
            alpha=0.55, color=c_dt, edgecolor="white", linewidth=0.5,
            label=f"DroidTask (p50={int(np.median(dt_p))}, min={dt_p.min()})")
    ax.hist(ac_p, bins=bins_p, weights=np.ones_like(ac_p) / len(ac_p) * 100,
            alpha=0.50, color=c_ac, edgecolor="white", linewidth=0.5,
            label=f"AC-Low 120 (p50={int(np.median(ac_p))}, min={ac_p.min()})")
    ax.axvline(32, color="#c0392b", linestyle="--", linewidth=1.4, label="P = 32 (0.0% turns)")
    ax.axvline(64, color="#7f8c8d", linestyle=":", linewidth=1.3,
               label=f"P = 64 (AC <=64: {np.sum(ac_p <= 64)}/{len(ac_p)})")
    ax.set_xscale("log")
    ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Incremental Prefill Tokens P (log scale)")
    ax.set_ylabel("Turn Share (%)")
    ax.set_title("(a) Per-Turn New Input Length (P)")
    ax.set_ylim(0, 27.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    # Panel (b): Episode Max Logical Length CDF
    ax = axes[0, 1]
    ax.step(dt_maxl, np.arange(1, len(dt_maxl) + 1) / len(dt_maxl) * 100,
            where="post", color=c_dt, linewidth=2.0,
            label=f"DroidTask 158 (p50={int(np.median(dt_maxl))}, max={dt_maxl.max()})")
    ax.step(ac_maxl, np.arange(1, len(ac_maxl) + 1) / len(ac_maxl) * 100,
            where="post", color=c_ac, linewidth=2.0,
            label=f"AC-Low 120 (p50={int(np.median(ac_maxl))}, max={ac_maxl.max()})")
    ax.axvspan(4500, 8000, color="#2ecc71", alpha=0.10, label="Phone Frozen Window (4.5k–8k)")
    ax.axvline(9472, color="#8e44ad", linestyle="--", linewidth=1.4, label="Phone n_ctx = 9,472")
    ax.set_xlabel("Max Cumulative Logical Tokens per Episode")
    ax.set_ylabel("Cumulative Episode Share (%)")
    ax.set_title("(b) Episode Peak Context Length CDF")
    ax.set_xlim(0, max(ac_maxl.max(), dt_maxl.max()) * 1.04)
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    # Panel (c): Pure-Action Decode D distribution
    ax = axes[1, 0]
    bins_d = np.arange(5, max(dt_d.max(), ac_d.max()) + 3) - 0.5
    ax.hist(dt_d, bins=bins_d, weights=np.ones_like(dt_d) / len(dt_d) * 100,
            alpha=0.60, color=c_dt, edgecolor="white", linewidth=0.5,
            label=f"DroidTask (p50={int(np.median(dt_d))}, range {dt_d.min()}–{dt_d.max()})")
    ax.hist(ac_d, bins=bins_d, weights=np.ones_like(ac_d) / len(ac_d) * 100,
            alpha=0.55, color=c_ac, edgecolor="white", linewidth=0.5,
            label=f"AC-Low 120 (p50={int(np.median(ac_d))}, range {ac_d.min()}–{ac_d.max()})")
    ax.set_xlabel("Pure-Action Output Tokens D (No <think>)")
    ax.set_ylabel("Turn Share (%)")
    ax.set_title("(c) Pure-Action Decode Length (D)")
    ax.set_ylim(0, 72)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    # Panel (d): Joint (H, P) on warm turns (turn >= 1) + frozen replay overlay
    ax = axes[1, 1]
    ax.scatter(dt_h_warm, dt_p_warm, s=13, alpha=0.45, color=c_dt, edgecolors="none",
               label=f"DroidTask Warm Turns (n={len(dt_h_warm)})")
    ax.scatter(ac_h_warm, ac_p_warm, s=13, alpha=0.40, color=c_ac, edgecolors="none",
               label=f"AC-Low 120 Warm Turns (n={len(ac_h_warm)})")

    frz_h = [t["H"] for r in dt_rows + ac_rows if r["id"] in frozen_ids for t in r["turns"][1:]]
    frz_p = [t["P"] for r in dt_rows + ac_rows if r["id"] in frozen_ids for t in r["turns"][1:]]
    if frz_h:
        ax.scatter(frz_h, frz_p, s=26, facecolors="none", edgecolors=c_frz, linewidths=0.9,
                   label=f"Frozen 8 Traces Warm Turns (n={len(frz_h)})")

    ax.axhline(32, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(9472, color="#8e44ad", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yscale("log")
    ax.set_yticks([32, 64, 128, 256, 512, 1024, 2048])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Anchored History Tokens H (Warm Turns)")
    ax.set_ylabel("Incremental Input P (log scale)")
    ax.set_title("(d) Joint (H, P) Regime Map (Warm Turns)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    png_path = out_dir / "mobile_gui_full_hpd_distribution.png"
    pdf_path = out_dir / "mobile_gui_full_hpd_distribution.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "cohorts": [
            summarize_cohort("droidtask_all_158", dt_rows),
            summarize_cohort("ac_low_120", ac_rows),
        ],
        "figures": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
    }
    (out_dir / "ac120_and_dt158_hpd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Export CSV for AC-120 + DT-158 combined Fig. 1 dataset
    csv_path = out_dir / "ac120_and_dt158_turn_hpd.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "case_id", "turn_index", "pin_tokens", "H", "P", "D", "prompt_tokens", "logical_tokens"])
        for r in dt_rows + ac_rows:
            for t in r["turns"]:
                w.writerow([r["benchmark"], r["id"], t["turn"], r["pin"], t["H"], t["P"], t["D"], t["prompt_tokens"], t["logical"]])

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dt-json", type=Path, default=ROOT / "results/mainline-eval/mobile-gui-hpd/mobile-gui-hpd-exact.json")
    p.add_argument("--ac120-json", type=Path, default=ROOT / "results/mainline-eval/mobile-gui-hpd/ac120-hpd-exact.json")
    p.add_argument("--inventory", type=Path, default=ROOT / "results/mainline-eval/mobile-gui-hpd/inventory.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/mainline-eval/mobile-gui-hpd")
    args = p.parse_args()

    dt_rows = [r for r in json.loads(args.dt_json.read_text()) if r["benchmark"] == "droidtask"]
    ac_rows = json.loads(args.ac120_json.read_text())
    frozen_ids = {c["id"] for c in json.loads(args.inventory.read_text())["cases"]} if args.inventory.exists() else set()
    summary = plot_hpd(dt_rows, ac_rows, frozen_ids, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
