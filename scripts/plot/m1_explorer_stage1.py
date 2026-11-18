#!/usr/bin/env python3
"""Plot Stage 1 results: H=4096 fine-grained truth vs M1-Explorer, multi-H scaling, and Oracle convergence."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from a2o.scheduling.m1_explorer import M1Explorer

TRUTH_H4096 = ROOT / "results/20261117-m1-explorer-truth-h4096/truth.json"
COMP_H4096 = ROOT / "results/20261117-m1-explorer-truth-h4096/compare.json"
MULTI_H_RAW = ROOT / "results/20261117-m1-explorer-large-campaign/oracle-grid/raw_multi_h_truth.json"
TINY_PROBES = ROOT / "results/20261117-m1-explorer-bootstrap/tiny_probes.json"
OUT_DIR = ROOT / "results/20261117-m1-explorer-large-campaign/figures"


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


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h4096_truth = json.loads(TRUTH_H4096.read_text())
    h4096_comp = json.loads(COMP_H4096.read_text())
    multi_h = json.loads(MULTI_H_RAW.read_text())
    tiny_probes = json.loads(TINY_PROBES.read_text())["probes"]
    all_truth = list(multi_h) + list(h4096_truth)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.3), constrained_layout=True)

    # Panel (a): H=4096 fine-grained truth vs M1-Explorer learned curves
    ax = axes[0]
    ps = [x["P"] for x in h4096_truth]
    cpu_t = [x["cpu_total_ms"] / 1000.0 for x in h4096_truth]
    htp_t = [x["htp_total_ms"] / 1000.0 for x in h4096_truth]
    exp_cpu = [x["cc_pred_ms"] / 1000.0 for x in h4096_comp["explorer_final_curve"]]
    exp_htp = [x["hc_pred_ms"] / 1000.0 for x in h4096_comp["explorer_final_curve"]]

    ax.plot(ps, cpu_t, "o-", color="#2b5c8f", lw=1.8, ms=4.5, label="CPU 真值 (含恢复)")
    ax.plot(ps, htp_t, "s-", color="#d95f02", lw=1.8, ms=4.5, label="HTP 真值 (含搬回)")
    ax.plot(ps, exp_cpu, "--", color="#6baed6", lw=1.8, label="M1-Explorer 学得 CPU")
    ax.plot(ps, exp_htp, "--", color="#fd8d3c", lw=1.8, label="M1-Explorer 学得 HTP")
    ax.axvline(36, color="#666666", ls=":", lw=1.1)
    ax.text(38, max(cpu_t) * 0.90, "交叉区 P≈36–40", fontsize=8.2, color="#444444")
    ax.set_xlabel("新增 token 数 P")
    ax.set_ylabel("总耗时 (s)")
    ax.set_title("(a) H=4096 细粒度真值 vs M1-Explorer 拟合", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left")

    # Panel (b): Multi-H scaling (H=2048, 6144, 9216)
    ax = axes[1]
    styles = {
        2048: ("#6baed6", "#fdbe85", "o"),
        6144: ("#2b5c8f", "#d95f02", "s"),
        9216: ("#08306b", "#a63603", "^"),
    }
    for h_val, (c_col, h_col, mk) in styles.items():
        sub = sorted([r for r in multi_h if r["H"] == h_val], key=lambda z: z["P"])
        xp = [r["P"] for r in sub]
        ax.plot(xp, [r["cpu_total_ms"] / 1000.0 for r in sub], f"{mk}-", color=c_col, lw=1.5, ms=4, label=f"CPU H={h_val}")
        ax.plot(xp, [r["htp_total_ms"] / 1000.0 for r in sub], f"{mk}--", color=h_col, lw=1.7, ms=4, label=f"HTP H={h_val}")
    ax.set_xlabel("新增 token 数 P")
    ax.set_ylabel("总耗时 (s)")
    ax.set_title("(b) 跨历史深度真值对比 (2k / 6k / 9k)", fontweight="bold")
    ax.legend(fontsize=7.4, ncol=2, loc="upper left")

    # Panel (c): Completed real tasks (0..9) -> Fixed multi-H testbed Oracle closeness
    ax = axes[2]
    tdir = ROOT / "results/20261116-tau2-qwen3-4b-joint-v2-e2e/conditions/cpu_baseline/tasks"
    real_tasks = []
    for fp in sorted(tdir.glob("*.json")):
        if ".failed-" in fp.name:
            continue
        td = json.loads(fp.read_text())
        turns = [
            (int(c["timings"]["H_anchor"]), int(c["timings"]["P_evaluated"]))
            for c in td.get("calls", [])
            if c.get("is_agent") and c["timings"].get("H_anchor", 0) > 0 and c["timings"].get("P_evaluated", 0) > 0
        ]
        if turns:
            real_tasks.append((td["key"], turns))

    # Build continuous true-latency surface over all 49 real measurements for task-stream feedback
    env = M1Explorer(online_update_enabled=True, exploration_enabled=False)
    for pt in all_truth:
        env.observe(h=int(pt["H"]), p=int(pt["P"]), route="cpu",
                    restore_ms=float(pt["cpu_restore_ms"]), prefill_ms=float(pt["cpu_prefill_ms"]), force=True)
        env.observe(h=int(pt["H"]), p=int(pt["P"]), route="htp",
                    prefill_ms=float(pt["htp_prefill_ms"]), handoff_ms=float(pt["htp_to_cpu_ms"]), force=True)

    def eval_on_fixed_truth(router: M1Explorer) -> float:
        tot_oracle = 0.0
        tot_model = 0.0
        for pt in all_truth:
            h, p = int(pt["H"]), int(pt["P"])
            cc, hc = float(pt["cpu_total_ms"]), float(pt["htp_total_ms"])
            tot_oracle += min(cc, hc) if p >= 32 else cc
            dec = router.select_route(h, p, htp_ready=True)
            tot_model += hc if dec.route == "htp" else cc
        return tot_oracle / max(1.0, tot_model) * 100.0

    routers = {
        "static_tiny": M1Explorer(online_update_enabled=False, exploration_enabled=False),
        "online_greedy": M1Explorer(beta=0.0, online_update_enabled=True, exploration_enabled=False),
        "m1_explorer": M1Explorer(beta=1.5, max_explore_penalty_ms=4000.0, online_update_enabled=True, exploration_enabled=True),
    }
    for r in routers.values():
        r.bootstrap_probes(tiny_probes)

    xs = list(range(len(real_tasks) + 1))
    scores = {k: [eval_on_fixed_truth(r)] for k, r in routers.items()}
    for _, turns in real_tasks:
        for h, p in turns:
            for name, r in routers.items():
                dec = r.select_route(h, p, htp_ready=True)
                if dec.route == "htp":
                    hp_obs, _ = env.htp_prefill.predict(r._prefill_feat(h, p))
                    ho_obs, _ = env.htp_to_cpu.predict([1.0, float(h + p)])
                    r.observe(h=h, p=p, route="htp", prefill_ms=hp_obs, handoff_ms=ho_obs)
                else:
                    rst_obs, _ = env.cpu_restore.predict([1.0, float(h)])
                    cp_obs, _ = env.cpu_prefill.predict(r._prefill_feat(h, p))
                    r.observe(h=h, p=p, route="cpu", restore_ms=rst_obs, prefill_ms=cp_obs)
        for name, r in routers.items():
            scores[name].append(eval_on_fixed_truth(r))

    ax.axhline(100.0, color="#222222", ls="--", lw=1.2, label="Oracle 上界 (100%)")
    ax.plot(xs, scores["m1_explorer"], "o-", color="#c44e52", lw=2.3, ms=5,
            label=f"M1-Explorer ({scores['m1_explorer'][-1]:.1f}%)")
    ax.plot(xs, scores["online_greedy"], "s-.", color="#55a868", lw=1.8, ms=4.5,
            label=f"纯贪心在线 ({scores['online_greedy'][-1]:.1f}%)")
    ax.plot(xs, scores["static_tiny"], "^:", color="#4c72b0", lw=1.8, ms=4.5,
            label=f"静态小点不更新 ({scores['static_tiny'][-1]:.1f}%)")
    ax.set_xticks(xs)
    ax.set_xlabel("已完成的真实任务数 (k)")
    ax.set_ylabel("固定全区间接近 Oracle 程度 (%)")
    ax.set_title("(c) 随任务写入真实数据：逼近 Oracle 曲线", fontweight="bold")
    ax.set_ylim(80, 102)
    ax.legend(fontsize=8.2, loc="lower right")

    fig.suptitle("M1-Explorer 阶段 1：真值扫描与从零逼近 Oracle 收敛验证", fontsize=12.8, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig-stage1-oracle-and-convergence.{ext}", dpi=200)
    plt.close(fig)
    print("Wrote Stage 1 figures to", OUT_DIR)


if __name__ == "__main__":
    main()
