#!/usr/bin/env python3
"""Plot Step 1 KV Reduce curves, Step 2 cost boundary, and Step 3 tau2 E2E comparison."""
import argparse
import collections
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.scheduling.kv_reduce_cost import plan_kv_reduce


def _setup_style():
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


def plot_step1_curves(curve_sum: dict, cost_tbl: dict, out_dir: Path):
    h_rows = curve_sum["h_sweep"]
    k_rows = curve_sum["k_sweep"]
    p_rows = curve_sum["p_sweep"]
    cpu_probes = curve_sum["cpu_probes"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), constrained_layout=True)

    # Panel (a): Decode latency vs KV context size
    ax = axes[0, 0]
    hs = [r["H"] + r["P"] for r in h_rows]
    ax.plot(hs, [r["cpu"]["decode_ms_per_tok"] for r in h_rows], "o-", color="#2b5c8f", lw=2, label="CPU baseline (无裁剪, n_kv=H+P)")
    ax.plot([r["hc_nocb"]["decode_n_kv"] for r in h_rows], [r["hc_nocb"]["decode_ms_per_tok"] for r in h_rows], "s--", color="#6b8e23", lw=1.6, label="HC nocb (无裁剪直传)")
    ax.plot(hs, [r["hc_prior"]["decode_ms_per_tok"] for r in h_rows], "^-", color="#d95f02", lw=2, label="HC prior 裁剪后 (固定压至 K=1024)")
    ks = [r["decode_n_kv"] for r in k_rows]
    ax.plot(ks, [r["decode_ms_per_tok"] for r in k_rows], "D:", color="#7570b3", lw=1.8, label="k_sweep (按 K 扫描, H≈8192)")
    xs_fit = np.linspace(512, 8600, 100)
    dec_cfg = cost_tbl["decode_per_tok"]
    ax.plot(xs_fit, dec_cfg["intercept_ms"] + dec_cfg["slope_ms_per_token"] * xs_fit, "--", color="#444444", alpha=0.6, lw=1.2,
            label=f"成本表拟合: {dec_cfg['intercept_ms']:.1f} + {dec_cfg['slope_ms_per_token']:.5f}·n_kv")
    ax.set_title("(a) Decode 耗时随上下文/保留长度变化 (ms/tok)", fontweight="bold")
    ax.set_xlabel("上下文长度 (H+P) 或保留 KV 槽位数 K (tokens)")
    ax.set_ylabel("Decode 单 token 耗时 (ms/tok)")
    ax.legend(fontsize=8.3, loc="upper left")

    # Panel (b): Handoff time vs H / K
    ax = axes[0, 1]
    ax.plot(hs, [r["hc_nocb"]["handoff_ms"] for r in h_rows], "s--", color="#6b8e23", lw=2, label="HC nocb 回迁耗时 (n_kv=H+P)")
    ax.plot(hs, [r["hc_prior"]["handoff_ms"] for r in h_rows], "^-", color="#d95f02", lw=2, label="HC prior 裁剪后回迁 (K=1024)")
    ax.plot(ks, [r["handoff_ms"] for r in k_rows], "D:", color="#7570b3", lw=1.8, label="k_sweep 按保留档 K 回迁耗时")
    ho_cfg = cost_tbl["handoff"]
    ax.plot(xs_fit, ho_cfg["intercept_ms"] + ho_cfg["slope_ms_per_token"] * xs_fit, "--", color="#444444", alpha=0.6, lw=1.2,
            label=f"拟合: {ho_cfg['intercept_ms']:.1f} + {ho_cfg['slope_ms_per_token']:.4f}·n_kv")
    ax.set_title("(b) HTP→CPU KV 状态回迁 (Handoff) 耗时", fontweight="bold")
    ax.set_xlabel("有效传输 KV 长度 / 对应总长 (tokens)")
    ax.set_ylabel("Handoff 耗时 (ms)")
    ax.legend(fontsize=8.3, loc="upper left")

    # Panel (c): PrefillPrior cb_eval tax on HTP & CPU
    ax = axes[1, 0]
    h_vals = [r["H"] for r in h_rows]
    ax.plot(h_vals, [r["cb_eval_tax_ms"] for r in h_rows], "o-", color="#c51b7d", lw=2, label="HTP prior 增税 vs H (P=256)")
    # CPU probe differences
    cpu_by_kv = {}
    for grp in cpu_probes:
        by_h = collections.defaultdict(list)
        for row in grp["rows"]:
            by_h[row["h_target"]].append(row)
        cpu_by_kv[grp["meta"]["kv_reduce"]] = {h: sum(x["turn1_prompt_ms"] for x in rs) / len(rs) for h, rs in by_h.items()}
    cpu_hs = sorted(cpu_by_kv[0].keys())
    cpu_taxes = [cpu_by_kv[1][h] - cpu_by_kv[0][h] for h in cpu_hs]
    ax.plot(cpu_hs, cpu_taxes, "s-", color="#1b9e77", lw=2, label="CPU 回调先验税 vs H (turn1, P=30)")
    for h_pt, tx_pt in zip(cpu_hs, cpu_taxes):
        ax.annotate(f"{tx_pt:+.0f} ms", (h_pt, tx_pt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#1b9e77")
    ax.axhline(0, color="#666666", lw=1, ls=":")
    ax.axvspan(400, 2200, color="#ffcccc", alpha=0.30, label="H≤2048 死区 (节省不足以抵消回调税)")
    # Inset or secondary annotation for p_sweep
    p_desc = "P 扫描 (H=4096 HTP):\n" + "\n".join(f"P={r['P']}: +{r['cb_eval_tax_ms']:.0f} ms" for r in p_rows)
    ax.text(0.97, 0.06, p_desc, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#d0c090", alpha=0.9))
    ax.set_title("(c) PrefillPrior 回调税 (cb_eval_tax) 与死区来源", fontweight="bold")
    ax.set_xlabel("历史长度 H (tokens)")
    ax.set_ylabel("额外 Prefill 耗时税 (ms)")
    ax.legend(fontsize=8.3, loc="upper left")

    # Panel (d): Decision boundary phase diagram in (H, D) plane
    ax = axes[1, 1]
    # Compute exact minimal D where net_gain >= 100 ms for CPU and HTP
    h_grid = np.arange(1024, 9217, 128)
    cpu_thresh_d, htp_thresh_d = [], []
    for h_val in h_grid:
        d_c = np.nan
        d_h = np.nan
        for d_test in range(1, 500):
            if np.isnan(d_c) and plan_kv_reduce(cost_tbl, int(h_val - 256), 256, d_test, is_htp=False)["admit"]:
                d_c = d_test
            if np.isnan(d_h) and plan_kv_reduce(cost_tbl, int(h_val - 256), 256, d_test, is_htp=True)["admit"]:
                d_h = d_test
            if not np.isnan(d_c) and not np.isnan(d_h):
                break
        cpu_thresh_d.append(d_c)
        htp_thresh_d.append(d_h)

    ax.axvspan(1024, 2048, color="#f4cccc", alpha=0.45, label="质量/最小删减下界拦截区 (below floor)")
    ax.plot(h_grid, cpu_thresh_d, "-", color="#2b5c8f", lw=2.4, label="CPU 准入边界 (净收益 ≥ 100 ms)")
    ax.plot(h_grid, htp_thresh_d, "--", color="#d95f02", lw=2.2, label="HTP→CPU 准入边界 (P=256, 含回迁省时)")
    ax.fill_between(h_grid, cpu_thresh_d, 256, where=~np.isnan(cpu_thresh_d), color="#cce5ff", alpha=0.38, label="CPU KV Reduce 净收益准入区")
    ax.axhline(96, color="#2ca02c", lw=1.6, ls="-.", label="实验默认 expected_decode_tokens D=96")
    for h_mk, c_mk, txt, xyoff in [
        (2279, "#b2182b", "2k: 拒绝(-421ms)", (-10, -18)),
        (4329, "#1b7837", "4k: 准入(+1030ms)", (0, -16)),
        (6379, "#1b7837", "6k: 准入(+2301ms)", (0, 10)),
    ]:
        ax.scatter([h_mk], [96], color=c_mk, s=55, zorder=5)
        ax.annotate(txt, (h_mk, 96), textcoords="offset points", xytext=xyoff, ha="center", fontsize=8.2, color=c_mk, fontweight="bold")
    ax.set_xlim(1024, 9000)
    ax.set_ylim(0, 240)
    ax.set_title("(d) kv_reduce_cost_v1 成本表决策相图 (H_total vs D)", fontweight="bold")
    ax.set_xlabel("总上下文长度 H_total = H_eff + P (tokens)")
    ax.set_ylabel("预期生成长度 D (tokens)")
    ax.legend(fontsize=8.0, loc="upper right")

    fig.suptitle("Qwen3-1.7B 机制 2 第 1–2 步：KV Reduce 时间-历史曲线与成本准入边界", fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1-time-deadzone-curves.{ext}", dpi=200)
    plt.close(fig)


def plot_quality_curve(curve_sum: dict, out_dir: Path):
    q_rows = curve_sum["quality_k_curve"]
    by_k = collections.defaultdict(list)
    for r in q_rows:
        by_k[r["kv_keep"]].append(r)

    ks = sorted(by_k.keys())
    pass_rates = [sum(1 for r in by_k[k] if r["all_key_facts_passed"]) / len(by_k[k]) * 100.0 for k in ks]
    mean_lcp = [sum(r["lcp_tokens_cpu"] for r in by_k[k]) / len(by_k[k]) for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    # Panel (a): Quality vs K
    l1 = ax1.plot(ks, pass_rates, "o-", color="#2b5c8f", lw=2.4, ms=7, label="关键事实通过率 (%) [左轴]")
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("保留 KV 预算 K (tokens)")
    ax1.set_ylabel("τ²-bench 关键事实通过率 (%)", color="#2b5c8f")
    ax1.axvline(1536, color="#d95f02", ls="--", lw=1.8, label="拟合安全下界 K_min = 1536")
    for x, y in zip(ks, pass_rates):
        ax1.annotate(f"{int(round(y/100*6))}/6", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.8, color="#2b5c8f", fontweight="bold")

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.grid(False)
    l2 = ax1_r.plot(ks, mean_lcp, "s--", color="#1b9e77", lw=2.0, ms=6, label="Greedy 与 CPU 对齐前缀长度 LCP [右轴]")
    ax1_r.set_ylabel("平均连续匹配 Token 数 (LCP)", color="#1b9e77")
    ax1_r.set_ylim(0, 36)
    lines = l1 + l2 + [ax1.get_lines()[1]]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=8.4)
    ax1.set_title("(a) 质量退化拐点：保留长度 K vs 通过率与 LCP", fontweight="bold")

    # Panel (b): Keep ratio vs decode speed (Pareto of all 36 boundary points)
    pass_x = [r["keep_ratio"] for r in q_rows if r["all_key_facts_passed"]]
    pass_y = [r["decode_ms_per_tok"] for r in q_rows if r["all_key_facts_passed"]]
    fail_x = [r["keep_ratio"] for r in q_rows if not r["all_key_facts_passed"]]
    fail_y = [r["decode_ms_per_tok"] for r in q_rows if not r["all_key_facts_passed"]]
    ax2.scatter(pass_x, pass_y, c="#1b7837", s=65, alpha=0.82, edgecolors="white", lw=0.6, label=f"关键事实通过 ({len(pass_x)}/36)")
    ax2.scatter(fail_x, fail_y, c="#d73027", marker="X", s=85, alpha=0.9, label=f"关键事实缺失 ({len(fail_x)}/36)")
    for r in q_rows:
        if not r["all_key_facts_passed"] and r["keep_ratio"] >= 0.12:
            off_y = -11 if (r["boundary"] == 2 and r["kv_keep"] == 1024) or (r["boundary"] == 0 and r["kv_keep"] == 3072) else 4
            ax2.annotate(f"b{r['boundary']} K={r['kv_keep']}", (r["keep_ratio"], r["decode_ms_per_tok"]),
                         textcoords="offset points", xytext=(6, off_y), fontsize=7.8, color="#a50f15")
    ax2.axvline(0.25, color="#d95f02", ls="--", lw=1.8, label="拟合最小保留比例 r_min = 0.25")
    ax2.axvspan(0, 0.25, color="#ffcccc", alpha=0.22)
    ax2.set_title("(b) 36 组边界质量-速度分布 (保留比例 vs Decode 耗时)", fontweight="bold")
    ax2.set_xlabel("KV 保留比例 K / (H+P)")
    ax2.set_ylabel("Decode 单 token 耗时 (ms/tok)")
    ax2.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Qwen3-1.7B KV Reduce 质量拐点标定（6 个 τ²-bench 边界 × 6 档 K）", fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2-quality-budget-pareto.{ext}", dpi=200)
    plt.close(fig)


def plot_e2e_three_conditions(e2e_sum: dict, out_dir: Path):
    conds = ["cpu_baseline", "cpu_adaptive_full", "cpu_adaptive_compact"]
    short_labels = ["cpu_baseline\n(kv=0 无裁剪)", "cpu_adaptive_full\n(成本准入, 锚点不缩)", "cpu_adaptive_compact\n(成本准入 + 紧凑锚点)"]

    # Common completed subset across all 3 conditions
    by_task = collections.defaultdict(dict)
    for c in conds:
        for t in e2e_sum["conditions"][c]["tasks"]:
            by_task[(t["domain"], t["task_id"])][c] = t
    common_keys = [k for k, mp in by_task.items() if all(mp[c]["complete"] for c in conds)]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True)

    # Panel (a): Common 5 tasks Agent service stage breakdown
    ax = axes[0]
    x = np.arange(len(conds))
    rst, prm, kvr, dec, oth, tots = [], [], [], [], [], []
    for c in conds:
        sub = [by_task[k][c] for k in common_keys]
        s_rst = sum(t["restore_s"] for t in sub)
        s_prm = sum(t["prompt_s"] for t in sub)
        s_kvr = sum(t["kv_reduce_s"] for t in sub)
        s_dec = sum(t["decode_s"] for t in sub)
        s_tot = sum(t["agent_service_s"] for t in sub)
        s_oth = max(0.0, s_tot - (s_rst + s_prm + s_kvr + s_dec))
        rst.append(s_rst); prm.append(s_prm); kvr.append(s_kvr); dec.append(s_dec); oth.append(s_oth); tots.append(s_tot)

    w = 0.52
    b_rst = np.array(rst)
    b_prm = np.array(prm)
    b_kvr = np.array(kvr)
    b_dec = np.array(dec)
    b_oth = np.array(oth)
    ax.bar(x, b_prm, w, label="Prefill (+cb_eval 回调税)", color="#4c72b0")
    ax.bar(x, b_dec, w, bottom=b_prm, label="Decode", color="#55a868")
    ax.bar(x, b_rst, w, bottom=b_prm + b_dec, label="Restore (快照恢复)", color="#c44e52")
    ax.bar(x, b_kvr, w, bottom=b_prm + b_dec + b_rst, label="KV Reduce (驱逐+紧凑)", color="#dd8452")
    ax.bar(x, b_oth, w, bottom=b_prm + b_dec + b_rst + b_kvr, label="其他(快照保存/分词)", color="#999999")
    for idx, tot in enumerate(tots):
        sp = tots[0] / tot
        ax.text(idx, tot + 14, f"{tot:.1f}s\n({sp:.2f}×)", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax.set_xticks(x, short_labels, fontsize=8.3)
    ax.set_ylim(0, max(tots) * 1.18)
    ax.set_ylabel("Agent 服务端累计耗时 (s)")
    ax.set_title(f"(a) 严格公共 {len(common_keys)} 题分阶段耗时拆解", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper right")

    # Panel (b): Long-turn admission rate & Peak memory (VmHWM)
    ax = axes[1]
    hwm_gib = [int(e2e_sum["conditions"][c]["phone_memory"]["after"]["VmHWM"].split()[0]) / (1024 * 1024) for c in conds]
    admit_pct = []
    for c in conds:
        lt = e2e_sum["conditions"][c]["long_turns_ge4k"]
        admit_pct.append(100.0 * lt["admitted_count"] / lt["count"] if lt["count"] else 0.0)
    bar_w = 0.35
    b1 = ax.bar(x - bar_w / 2, hwm_gib, bar_w, color="#2b5c8f", label="手机内存峰值 VmHWM (GiB) [左]")
    ax.set_ylim(0, 8.2)
    ax.set_ylabel("手机内存峰值 VmHWM (GiB)", color="#2b5c8f")
    for bar, val in zip(b1, hwm_gib):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.2f}\nGiB", ha="center", va="bottom", fontsize=8.2, color="#2b5c8f", fontweight="bold")

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.grid(False)
    b2 = ax_r.bar(x + bar_w / 2, admit_pct, bar_w, color="#dd8452", label="长轮(≥4k)触发 KV Reduce 比例 (%) [右]")
    ax_r.set_ylim(0, 122)
    ax_r.set_ylabel("长轮触发 KV Reduce 比例 (%)", color="#b35806")
    for bar, val, c in zip(b2, admit_pct, conds):
        lt = e2e_sum["conditions"][c]["long_turns_ge4k"]
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{lt['admitted_count']}/{lt['count']}\n({val:.0f}%)",
                  ha="center", va="bottom", fontsize=8.0, color="#b35806", fontweight="bold")
    ax.set_xticks(x, short_labels, fontsize=8.3)
    ax.set_title("(b) 内存峰值与重裁剪频率（9 题全集）", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper left")

    # Panel (c): Logical H_anchor vs Physical H_effective on tasks
    ax = axes[2]
    tasks_compact = e2e_sum["conditions"]["cpu_adaptive_compact"]["tasks"]
    tasks_full = e2e_sum["conditions"]["cpu_adaptive_full"]["tasks"]
    t_names = [
        f"{t['domain'][:3]}:{t['task_id'][:8]}" if not t["task_id"].startswith("[") else f"tel:{idx-5}"
        for idx, t in enumerate(tasks_compact)
    ]
    tx = np.arange(len(tasks_compact))
    h_anc = [t["max_H_anchor"] for t in tasks_compact]
    h_eff = [t["max_H_effective"] for t in tasks_compact]
    ax.plot(tx, h_anc, "o--", color="#666666", lw=1.8, label="compact_anchor 逻辑位置 max_H_anchor")
    ax.plot(tx, h_eff, "s-", color="#1b7837", lw=2.3, label="compact_anchor 物理活跃 KV max_H_effective")
    # Mark telecom:break_apn overflow on full_anchor
    full_t7 = tasks_full[7]
    ax.scatter([7], [full_t7["max_prompt_tokens"]], color="#d73027", marker="X", s=110, zorder=6,
               label=f"full_anchor 第16轮溢出 (11371+768 > 12288)")
    ax.axhline(12288, color="#d73027", ls=":", lw=1.4, label="context_size = 12288 上限")
    ax.set_xticks(tx, t_names, rotation=30, ha="right", fontsize=8.2)
    ax.set_ylabel("Anchor Token / KV 槽位数")
    ax.set_ylim(0, 16200)
    ax.set_title("(c) compact_anchor 防上下文溢出效果（9 题）", fontweight="bold")
    ax.legend(fontsize=7.6, loc="upper left")

    fig.suptitle("Qwen3-1.7B τ²-bench 三条件端到端对比（9 题实测，含 5 题公共严格对齐）", fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3-tau2-e2e-three-conditions.{ext}", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve-summary", type=Path, default=ROOT / "results/20261115-qwen3-1_7b-kv-curve/summary.json")
    p.add_argument("--cost-config", type=Path, default=ROOT / "configs/qwen3-1_7b-kv-reduce-cost.json")
    p.add_argument("--e2e-summary", type=Path, default=ROOT / "results/20261115-tau2-qwen3-1_7b-adaptive-kv-e2e/summary.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/20261115-qwen3-1_7b-kv-curve/figures")
    args = p.parse_args()

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve_sum = json.loads(args.curve_summary.read_text())
    cost_tbl = json.loads(args.cost_config.read_text())
    e2e_sum = json.loads(args.e2e_summary.read_text())

    plot_step1_curves(curve_sum, cost_tbl, args.out_dir)
    plot_quality_curve(curve_sum, args.out_dir)
    plot_e2e_three_conditions(e2e_sum, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
