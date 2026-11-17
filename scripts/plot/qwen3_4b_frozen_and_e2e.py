#!/usr/bin/env python3
"""Plot Qwen3-4B-Q4_0 paired frozen-replay and tau2 E2E joint M1+M2 summary figures."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = ROOT / "results/20261117-qwen3-4b-frozen-replay"
E2E_DIR = ROOT / "results/20261116-tau2-qwen3-4b-joint-v2-e2e"
OUT_DIR = FROZEN_DIR / "figures"

CONDS = ["cpu_baseline", "cpu_adaptive_compact", "route_m1_only", "route_m1_m2_compact"]
LABELS = ["CPU 基线", "CPU + KV 压缩", "仅 M1 路由", "M1 + M2 联合"]
COLORS = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]


def setup_style() -> None:
    arial_unicode = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if arial_unicode.exists():
        font_manager.fontManager.addfont(str(arial_unicode))
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["font.size"] = 10


def load_e2e_metrics() -> dict:
    out = {}
    for c in CONDS:
        tdir = E2E_DIR / "conditions" / c / "tasks"
        tfiles = [p for p in sorted(tdir.glob("*.json")) if ".failed-" not in p.name]
        rew = sum(json.loads(p.read_text()).get("reward", 0) or 0 for p in tfiles)
        w_tot, w_prm = [], []
        for p in tfiles:
            d = json.loads(p.read_text())
            for call in d.get("calls", []):
                if not call.get("is_agent"):
                    continue
                t = call["timings"]
                if t.get("H_anchor", 0) > 0:
                    w_tot.append(t["total_ms"] / 1000.0)
                    w_prm.append(t["prompt_ms"] / 1000.0)
        out[c] = {
            "reward": rew,
            "n_tasks": len(tfiles),
            "warm_total_s": statistics.mean(w_tot),
            "warm_prompt_s": statistics.mean(w_prm),
        }
    return out


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(p.read_text()) for p in sorted((FROZEN_DIR / "samples").glob("*.json"))]
    summary = json.loads((FROZEN_DIR / "summary.json").read_text())["summary"]
    e2e = load_e2e_metrics()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

    # Panel (a): Paired frozen replay per-sample normalized to cpu_baseline (%)
    ax = axes[0]
    x = np.arange(len(samples))
    width = 0.20
    s_labels = [f"P={s['sample']['P']}\n(H={s['sample']['H']//1000}.{s['sample']['H']%1000//100}k)" for s in samples]
    for idx, (c, lbl, col) in enumerate(zip(CONDS, LABELS, COLORS)):
        vals = [
            s["conditions"][c]["post_return_ms"] / s["conditions"]["cpu_baseline"]["post_return_ms"] * 100.0
            for s in samples
        ]
        ax.bar(x + (idx - 1.5) * width, vals, width, label=lbl, color=col, alpha=0.92)
    ax.axhline(100.0, color="#333333", ls="--", lw=1.0)
    ax.set_xticks(x, s_labels, fontsize=8.5)
    ax.set_ylabel("相对 CPU 基线耗时 (%)")
    ax.set_title("(a) 冻结成对回放：6 样本单轮相对耗时", fontweight="bold")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=8, loc="upper right")

    # Panel (b): Stage breakdown (mean across 6 frozen samples, seconds)
    ax = axes[1]
    bx = np.arange(len(CONDS))
    stages = [
        ("avg_restore_ms", "Restore", "#8c564b"),
        ("avg_prefill_ms", "Prefill", "#4c72b0"),
        ("avg_evict_ms", "KV 裁剪", "#9467bd"),
        ("avg_handoff_ms", "Handoff", "#e377c2"),
        ("avg_decode_ms", "Decode", "#55a868"),
    ]
    bottom = np.zeros(len(CONDS))
    for key, sname, scol in stages:
        vals = np.array([summary[c][key] / 1000.0 for c in CONDS])
        ax.bar(bx, vals, 0.52, bottom=bottom, label=sname, color=scol, alpha=0.9)
        bottom += vals
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 1.5, f"{tot:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(bx, LABELS, fontsize=8.8)
    ax.set_ylabel("平均响应耗时 (s)")
    ax.set_title("(b) 冻结回放均值分解 (6 样本)", fontweight="bold")
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.legend(fontsize=8, loc="upper right")

    # Panel (c): Live tau2 E2E warm turn latency & pass count
    ax = axes[2]
    ex = np.arange(len(CONDS))
    wt_vals = [e2e[c]["warm_total_s"] for c in CONDS]
    wp_vals = [e2e[c]["warm_prompt_s"] for c in CONDS]
    bars = ax.bar(ex - 0.17, wt_vals, 0.34, label="Warm 整轮均值 (s)", color="#4c72b0", alpha=0.9)
    ax.bar(ex + 0.17, wp_vals, 0.34, label="Warm Prefill 均值 (s)", color="#dd8452", alpha=0.9)
    for i, c in enumerate(CONDS):
        ax.text(i - 0.17, wt_vals[i] + 1.0, f"{wt_vals[i]:.1f}s\n({int(e2e[c]['reward'])}/{e2e[c]['n_tasks']})",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        ax.text(i + 0.17, wp_vals[i] + 1.0, f"{wp_vals[i]:.1f}s", ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(ex, LABELS, fontsize=8.8)
    ax.set_ylabel("平均耗时 (s)")
    ax.set_title("(c) τ²-bench 端到端 Warm 轮延迟与通过题数", fontweight="bold")
    ax.set_ylim(0, max(wt_vals) * 1.22)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Qwen3-4B-Q4_0：M1（Tool-Gap 选路）与 M2（自适应 KV 压缩）联合验证汇总", fontsize=12.5, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig-qwen3-4b-joint-summary.{ext}", dpi=200)
    plt.close(fig)
    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
