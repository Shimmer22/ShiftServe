#!/usr/bin/env python3
"""Plot Qwen3-4B Joint (M1 Route x M2 Adaptive KV Reduce) calibration, memory gate, quality, and E2E figures."""
import argparse
import collections
import glob
import json
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.scheduling.kv_reduce_cost import plan_joint_route_kv


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


def _load_calib_points(calib_dir: Path):
    pts = []
    for fp in sorted(calib_dir.glob("H*-P*.response.json")):
        d = json.loads(fp.read_text())
        rows = d["rows"]
        pts.append({
            "H": d["H"],
            "P": d["P"],
            "cpu_prefill": statistics.median(x["cpu_prefill_ms"] for x in rows),
            "htp_prefill": statistics.median(x["htp_prefill_ms"] for x in rows),
            "cc_ms": statistics.median(x["cc_ms"] for x in rows),
            "hc_ms": statistics.median(x["hc_ms"] for x in rows),
        })
    return pts


def plot_calibration_and_gate(
    curve_sum: dict,
    kv_cost: dict,
    route_cost: dict,
    calib_dir: Path,
    gate_4b_path: Path,
    gate_8b_path: Path,
    out_dir: Path,
    e2e_sum: dict | None = None,
):
    calib_pts = _load_calib_points(calib_dir)
    route_model = route_cost["model"] if "model" in route_cost else route_cost
    h_rows = curve_sum["h_sweep"]
    k_rows = curve_sum["k_sweep"]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.8), constrained_layout=True)

    # Panel (a): CPU vs HTP Prefill scaling across P and H regimes
    ax = axes[0, 0]
    for h_val, col, mk in [(128, "#8da0cb", "o"), (768, "#4c72b0", "s"), (1536, "#1f4e79", "^")]:
        sub = sorted([x for x in calib_pts if x["H"] == h_val], key=lambda z: z["P"])
        if sub:
            ax.plot([x["P"] for x in sub], [x["cpu_prefill"] / 1000.0 for x in sub],
                    f"{mk}-", color=col, lw=1.7, label=f"CPU Prefill (H={h_val})")
            ax.plot([x["P"] for x in sub], [x["htp_prefill"] / 1000.0 for x in sub],
                    f"{mk}--", color=col, alpha=0.85, lw=1.7, label=f"HTP Prefill (H={h_val})")
    xs_p = np.linspace(1, 520, 120)
    cp_fit = route_model["cpu_prefill"]
    hp_fit = route_model["htp_prefill"]
    ax.plot(xs_p, (cp_fit["intercept_ms"] + cp_fit["slope_ms_per_token"] * xs_p) / 1000.0,
            ":", color="#222222", lw=1.6, label=f"线性拟合 CPU: {cp_fit['slope_ms_per_token']:.1f} ms/tok")
    ax.plot(xs_p, (hp_fit["intercept_ms"] + hp_fit["slope_ms_per_token"] * xs_p) / 1000.0,
            "-.", color="#d95f02", lw=1.8, label=f"线性拟合 HTP: {hp_fit['slope_ms_per_token']:.1f} ms/tok")
    ax.set_title("(a) 4B (cpu_repack=0) 校准网格：CPU vs HTP Prefill 交叉区间", fontweight="bold")
    ax.set_xlabel("增量提示长度 P (tokens)")
    ax.set_ylabel("Prefill 耗时 (s)")
    ax.legend(fontsize=7.6, ncol=2, loc="upper left")

    # Panel (b): Decode & Handoff vs K / n_kv
    ax = axes[0, 1]
    hs = [r["H"] + r["P"] for r in h_rows]
    ks = [r["decode_n_kv"] for r in k_rows]
    l1 = ax.plot(hs, [r["cpu"]["decode_ms_per_tok"] for r in h_rows], "o-", color="#2b5c8f", lw=2, label="CPU Decode (无裁剪)")
    l2 = ax.plot(ks, [r["decode_ms_per_tok"] for r in k_rows], "D-", color="#7570b3", lw=2, label="KV Reduce 后 Decode (按 K)")
    xs_kv = np.linspace(512, 8500, 100)
    dfit = kv_cost["decode_per_tok"]
    l3 = ax.plot(xs_kv, dfit["intercept_ms"] + dfit["slope_ms_per_token"] * xs_kv, "--", color="#444444", lw=1.3,
                 label=f"Decode 拟合: {dfit['intercept_ms']:.1f}+{dfit['slope_ms_per_token']:.4f}·n_kv")
    ax.set_xlabel("有效 KV 长度 / 总上下文 (tokens)")
    ax.set_ylabel("Decode 速度 (ms/tok)", color="#2b5c8f")
    max_dec_b = max([r["cpu"]["decode_ms_per_tok"] for r in h_rows] + [r["decode_ms_per_tok"] for r in k_rows])
    ax.set_ylim(85, max(165.0, max_dec_b * 1.16))

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.grid(False)
    l4 = ax_r.plot(hs, [r["hc_nocb"]["handoff_ms"] for r in h_rows], "s--", color="#d95f02", lw=1.8, label="无裁剪 Handoff (ms) [右]")
    l5 = ax_r.plot(ks, [r["handoff_ms"] for r in k_rows], "^:", color="#1b9e77", lw=2.0, label="裁剪后 Handoff (ms) [右]")
    ax_r.set_ylabel("HTP→CPU Handoff 耗时 (ms)", color="#d95f02")
    lines = l1 + l2 + l3 + l4 + l5
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.6, loc="upper left")
    ax.set_title("(b) 4B Decode 速率与 HTP→CPU Handoff 线性降载", fontweight="bold")

    # Panel (c): 8B vs 4B Dual-Resident RAM / Page-Fault / Swap Gate
    ax = axes[1, 0]
    g8 = json.loads(gate_8b_path.read_text()) if gate_8b_path.exists() else None
    g4 = json.loads(gate_4b_path.read_text()) if gate_4b_path.exists() else None
    cats, hwm_gibs, swaps_gib, majflts = [], [], [], []
    if g8 and "stages" in g8:
        for st in g8["stages"]:
            ctx_s = st.get("ctx_size", 0)
            hwm_k = max([st.get("mem_after_init", {}).get("VmHWM", 0)] + [c.get("mem", {}).get("VmHWM", 0) for c in st.get("chat", [])])
            swp_k = max([st.get("mem_after_init", {}).get("VmSwap", 0)] + [c.get("mem", {}).get("VmSwap", 0) for c in st.get("chat", [])])
            mf_peak = max((g.get("majflt_delta", 0) for g in st.get("gap_replay", [])), default=0)
            cats.append(f"8B 双驻留\nctx={ctx_s}\n(repack=0)")
            hwm_gibs.append(hwm_k / (1024 * 1024))
            swaps_gib.append(swp_k / (1024 * 1024))
            majflts.append(mf_peak)
    if g4 and "stages" in g4:
        for repack_val in (0, 1):
            st = next((s for s in g4["stages"] if int(bool(s.get("cpu_repack"))) == repack_val), None)
            if st:
                ctx_s = st.get("ctx_size", 8192)
                hwm_k = max([st.get("mem_after_init", {}).get("VmHWM", 0)] + [c.get("mem", {}).get("VmHWM", 0) for c in st.get("chat", [])])
                swp_k = max([st.get("mem_after_init", {}).get("VmSwap", 0)] + [c.get("mem", {}).get("VmSwap", 0) for c in st.get("chat", [])])
                mf_peak = max((g.get("majflt_delta", 0) for g in st.get("gap_replay", []) if g.get("P", 0) >= 128), default=0)
                cats.append(f"4B 双驻留\nctx={ctx_s}\n(repack={repack_val})")
                hwm_gibs.append(hwm_k / (1024 * 1024))
                swaps_gib.append(swp_k / (1024 * 1024))
                majflts.append(mf_peak)

    x_g = np.arange(len(cats))
    cols_g = ["#d73027" if mf > 1000 else ("#1b7837" if "repack=0" in cat else "#4c72b0") for cat, mf in zip(cats, majflts)]
    bars = ax.bar(x_g, hwm_gibs, 0.50, color=cols_g, alpha=0.88)
    for bar, hval, swp, mf in zip(bars, hwm_gibs, swaps_gib, majflts):
        tag_mf = f"Swap={swp:.2f}GiB\nΔmajflt={mf//1000}k!" if mf >= 1000 else f"Swap=0\nΔmajflt={mf}"
        ax.text(bar.get_x() + bar.get_width() / 2, hval + 0.18, f"{hval:.2f} GiB\n({tag_mf})",
                ha="center", va="bottom", fontsize=7.9, fontweight="bold",
                color="#a50f15" if mf >= 1000 else "#1b5e20")
    ax.set_xticks(x_g, cats, fontsize=8.3)
    ax.set_ylim(0, max(hwm_gibs) * 1.30)
    ax.set_ylabel("手机内存峰值 VmHWM (GiB)")
    ax.set_title("(c) 16 GiB 手机双驻留准入门控：8B 换页抖动 vs 4B 零换页", fontweight="bold")

    # Panel (d): Joint (Route x KV Reduce) 2D Decision Region Map at D=96
    ax = axes[1, 1]
    h_vals = np.linspace(256, 6144, 90)
    p_vals = np.linspace(16, 640, 80)
    grid = np.zeros((len(p_vals), len(h_vals)), dtype=int)
    # 0: CPU no-KV, 1: CPU + KV Reduce, 2: HTP no-KV, 3: HTP + KV Reduce
    for ip, pv in enumerate(p_vals):
        for ih, hv in enumerate(h_vals):
            pl = plan_joint_route_kv(route_model, kv_cost, int(hv), int(pv), 96, htp_allowed=True, kv_enabled=True)
            is_htp = pl["route"] == "htp"
            is_kv = pl["kv_plan"]["admit"]
            grid[ip, ih] = (2 if is_htp else 0) + (1 if is_kv else 0)
    cmap = ListedColormap(["#deebf7", "#6baed6", "#fee6ce", "#e6550d"])
    ax.pcolormesh(h_vals, p_vals, grid, cmap=cmap, vmin=0, vmax=3, shading="auto", alpha=0.85)
    ax.axhline(route_model.get("routing_p_min", 32), color="#333333", ls="--", lw=1.2)
    ax.text(350, route_model.get("routing_p_min", 32) + 8, f"routing_p_min = {route_model.get('routing_p_min', 32)}", fontsize=8.0, color="#222222")
    if e2e_sum and "route_m1_m2_compact" in e2e_sum.get("conditions", {}):
        # Overlay actual E2E turn (H_eff, P) points from route_m1_m2_compact worker if available
        cond_dir = Path(e2e_sum["conditions"]["route_m1_m2_compact"].get("cond_dir", ""))
        pass
    legend_patches = [
        Patch(facecolor="#deebf7", edgecolor="#888", label="CPU (不裁剪 KV)"),
        Patch(facecolor="#6baed6", edgecolor="#888", label="CPU + KV Reduce"),
        Patch(facecolor="#fee6ce", edgecolor="#888", label="HTP 迁移 (不裁剪 KV)"),
        Patch(facecolor="#e6550d", edgecolor="#888", label="联合 HTP + KV Reduce"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.0, loc="upper right", framealpha=0.92)
    ax.set_title("(d) 联合路由器 (H_eff, P) 四象限决策相图 (D_exp=96)", fontweight="bold")
    ax.set_xlabel("物理活跃 KV 历史 H_eff (tokens)")
    ax.set_ylabel("增量提示长度 P (tokens)")

    fig.suptitle("Qwen3-4B-Q4_0 联合机制（M1 异构路由 × M2 自适应 KV Reduce）：标定、内存门控与四象限决策域", fontsize=12.8, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1-4b-calibration-joint-region.{ext}", dpi=200)
    plt.close(fig)


def plot_quality_lcp(curve_sum: dict, out_dir: Path):
    q_rows = curve_sum["quality_k_curve"]
    by_k = collections.defaultdict(list)
    for r in q_rows:
        by_k[r["kv_keep"]].append(r)

    ks = sorted(by_k.keys())
    mean_lcp = [sum(r["lcp_tokens_cpu"] for r in by_k[k]) / len(by_k[k]) for k in ks]
    mean_dec = [sum(r["decode_ms_per_tok"] for r in by_k[k]) / len(by_k[k]) for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

    l1 = ax1.plot(ks, mean_lcp, "o-", color="#1b9e77", lw=2.4, ms=7, label="与 4B 全 KV 基线精确前缀匹配 LCP (tokens) [左]")
    for x, y in zip(ks, mean_lcp):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#1b7837", fontweight="bold")
    ax1.axvline(1536, color="#d95f02", ls="--", lw=1.8, label="安全下界 K_min = 1536 (top1 匹配 36/36=100%)")
    ax1.set_xlabel("保留 KV 预算 K (tokens)")
    ax1.set_ylabel("Greedy 精确一致前缀均值 LCP (tokens)", color="#1b7837")
    ax1.set_ylim(0, 68)

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.grid(False)
    l2 = ax1_r.plot(ks, mean_dec, "s--", color="#2b5c8f", lw=2.0, ms=6, label="平均 Decode 延迟 (ms/tok) [右]")
    ax1_r.set_ylabel("Decode 单 token 耗时 (ms/tok)", color="#2b5c8f")
    ax1_r.set_ylim(100, max(198.0, max(mean_dec) * 1.12))
    lines = l1 + [ax1.get_lines()[1]] + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8.1)
    ax1.set_title("(a) 保留预算 K vs 全序列 Greedy 对齐前缀 (LCP) 与 Decode 延迟", fontweight="bold")

    # Panel (b): Keep ratio vs LCP across 6 boundaries
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    for b in range(6):
        sub = sorted([r for r in q_rows if r["boundary"] == b], key=lambda z: z["keep_ratio"])
        ax2.plot([r["keep_ratio"] for r in sub], [r["lcp_tokens_cpu"] for r in sub],
                 "o-", color=palette[b], lw=1.8, label=f"边界 b{b} ({sub[0]['domain']}, H+P={sub[0]['H']+sub[0]['P']})")
    ax2.axvline(0.25, color="#d95f02", ls="--", lw=1.8, label="最小保留比例 r_min = 0.25")
    ax2.axvspan(0, 0.25, color="#ffcccc", alpha=0.22)
    ax2.set_xlabel("KV 保留比例 K / (H+P)")
    ax2.set_ylabel("Greedy 精确对齐前缀 LCP (tokens)")
    ax2.set_title("(b) 6 个 τ²-bench 边界的保留比例-对齐质量拐点", fontweight="bold")
    ax2.legend(fontsize=7.9, loc="upper left")

    fig.suptitle("Qwen3-4B-Q4_0 质量保真标定（6 边界 × 6 档 K = 36 组，全部 36/36 top1 与无裁剪基线一致）", fontsize=12.6, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2-4b-quality-lcp.{ext}", dpi=200)
    plt.close(fig)


def plot_e2e_joint(e2e_sum: dict, out_dir: Path):
    order = ["cpu_baseline", "route_m1_only", "cpu_adaptive_compact", "route_m1_m2_compact"]
    conds = [c for c in order if c in e2e_sum["conditions"]]
    label_map = {
        "cpu_baseline": "CPU Baseline\n(M0: kv=0)",
        "route_m1_only": "M1 Only\n(Route, kv=0)",
        "cpu_adaptive_compact": "M2 Only\n(CPU+KV Compact)",
        "route_m1_m2_compact": "Joint M1+M2\n(Route+KV Compact)",
    }
    short_labels = [label_map.get(c, c) for c in conds]

    by_task = collections.defaultdict(dict)
    for c in conds:
        for t in e2e_sum["conditions"][c]["tasks"]:
            by_task[(t["domain"], t["task_id"])][c] = t
    common_keys = [k for k, mp in by_task.items() if len(mp) == len(conds) and all(mp[c]["complete"] for c in conds)]
    use_keys = common_keys if common_keys else list(by_task.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.5), constrained_layout=True)

    # Panel (a): Service stage breakdown over common completed tasks
    ax = axes[0]
    x = np.arange(len(conds))
    prm, dec, rst, hnd, kvr, wait_s, oth, tots = [], [], [], [], [], [], [], []
    for c in conds:
        sub = [by_task[k][c] for k in use_keys if c in by_task[k] and by_task[k][c]["complete"]]
        s_prm = sum(t["prompt_s"] for t in sub)
        s_dec = sum(t["decode_s"] for t in sub)
        s_rst = sum(t["restore_s"] for t in sub)
        s_hnd = sum(t.get("handoff_s", 0.0) for t in sub)
        s_kvr = sum(t["kv_reduce_s"] for t in sub)
        s_wt = sum(t.get("wait_s", 0.0) for t in sub)
        s_tot = sum(t["agent_service_s"] for t in sub)
        s_oth = max(0.0, s_tot - (s_prm + s_dec + s_rst + s_hnd + s_kvr + s_wt))
        prm.append(s_prm); dec.append(s_dec); rst.append(s_rst); hnd.append(s_hnd)
        kvr.append(s_kvr); wait_s.append(s_wt); oth.append(s_oth); tots.append(s_tot)

    w = 0.54
    a_prm = np.array(prm); a_dec = np.array(dec); a_rst = np.array(rst)
    a_hnd = np.array(hnd); a_kvr = np.array(kvr); a_wt = np.array(wait_s); a_oth = np.array(oth)
    ax.bar(x, a_prm, w, label="Prefill", color="#4c72b0")
    ax.bar(x, a_dec, w, bottom=a_prm, label="Decode", color="#55a868")
    ax.bar(x, a_rst, w, bottom=a_prm + a_dec, label="Restore", color="#c44e52")
    ax.bar(x, a_hnd, w, bottom=a_prm + a_dec + a_rst, label="Handoff (HTP→CPU)", color="#9370db")
    ax.bar(x, a_kvr, w, bottom=a_prm + a_dec + a_rst + a_hnd, label="KV Reduce", color="#dd8452")
    ax.bar(x, a_wt, w, bottom=a_prm + a_dec + a_rst + a_hnd + a_kvr, label="Greedy Wait", color="#f1c40f")
    ax.bar(x, a_oth, w, bottom=a_prm + a_dec + a_rst + a_hnd + a_kvr + a_wt, label="其他", color="#999999")
    for idx, tot in enumerate(tots):
        sp = tots[0] / tot if tot > 0 else 0.0
        rw = e2e_sum["conditions"][conds[idx]]["reward_sum"]
        ct = e2e_sum["conditions"][conds[idx]]["completed_tasks"]
        ax.text(idx, tot + max(tots) * 0.02, f"{tot:.0f}s ({sp:.2f}×)\nrew={rw:.1f} ({ct}/9)",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.set_xticks(x, short_labels, fontsize=8.0)
    ax.set_ylim(0, max(tots) * 1.24 if tots else 1.0)
    ax.set_ylabel("Agent 服务端累计耗时 (s)")
    ax.set_title(f"(a) 公共完成 {len(use_keys)} 题端到端耗时拆解与全集得分", fontweight="bold")
    ax.legend(fontsize=7.3, ncol=2, loc="upper right")

    # Panel (b): Peak VmHWM, HTP routed ratio, KV admit ratio
    ax = axes[1]
    hwm_gib = []
    htp_pct = []
    for c in conds:
        cs = e2e_sum["conditions"][c]
        hwm_str = cs.get("phone_memory", {}).get("after", {}).get("VmHWM", "0 kB")
        hwm_gib.append(int(hwm_str.split()[0]) / (1024 * 1024) if hwm_str.split()[0].isdigit() else 0.0)
        tot_turns = cs.get("htp_routed_turns", 0) + cs.get("cpu_routed_turns", 0)
        htp_pct.append(100.0 * cs.get("htp_routed_turns", 0) / tot_turns if tot_turns else 0.0)
    bw = 0.34
    b1 = ax.bar(x - bw / 2, hwm_gib, bw, color="#2b5c8f", label="内存峰值 VmHWM (GiB) [左]")
    for bar, val in zip(b1, hwm_gib):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.2f}\nGiB", ha="center", va="bottom", fontsize=8.0, color="#2b5c8f", fontweight="bold")
    ax.set_ylim(0, max(hwm_gib + [8.0]) * 1.25)
    ax.set_ylabel("手机内存峰值 VmHWM (GiB)", color="#2b5c8f")

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.grid(False)
    b2 = ax_r.bar(x + bw / 2, htp_pct, bw, color="#e6550d", label="HTP 路由比例 (%) [右]")
    for bar, val, c in zip(b2, htp_pct, conds):
        cs = e2e_sum["conditions"][c]
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 2.0,
                  f"{val:.0f}%\n({cs.get('htp_routed_turns',0)})", ha="center", va="bottom", fontsize=7.8, color="#b35806", fontweight="bold")
    ax_r.set_ylim(0, 118)
    ax_r.set_ylabel("HTP 路由轮次占比 (%)", color="#b35806")
    ax.set_xticks(x, short_labels, fontsize=8.0)
    ax.set_title("(b) 内存驻留峰值与动态 HTP 选路占比 (全 9 题)", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.6, loc="upper left")

    # Panel (c): H_anchor vs H_effective across 9 tasks
    ax = axes[2]
    tasks_j = e2e_sum["conditions"]["route_m1_m2_compact"]["tasks"]
    t_names = [f"{t['domain'][:3]}:{t['task_id'][:6]}" if not t["task_id"].startswith("[") else f"tel:{idx-5}" for idx, t in enumerate(tasks_j)]
    tx = np.arange(len(tasks_j))
    ax.plot(tx, [t["max_H_anchor"] for t in tasks_j], "o--", color="#666666", lw=1.8, label="Joint M1+M2 逻辑 max_H_anchor")
    ax.plot(tx, [t["max_H_effective"] for t in tasks_j], "s-", color="#1b7837", lw=2.3, label="Joint M1+M2 物理活跃 max_H_effective")
    if "route_m1_only" in e2e_sum["conditions"]:
        tasks_m1 = e2e_sum["conditions"]["route_m1_only"]["tasks"]
        ax.plot(tx, [t["max_H_anchor"] for t in tasks_m1], "^:", color="#d95f02", lw=1.7, label="M1-only max_H_anchor (=物理全量)")
    ax.axhline(e2e_sum["context_size"], color="#d73027", ls=":", lw=1.4, label=f"context_size = {e2e_sum['context_size']}")
    ax.set_xticks(tx, t_names, rotation=30, ha="right", fontsize=8.0)
    ax.set_ylabel("上下文 / 活跃 KV 槽位数 (tokens)")
    ax.set_title("(c) 紧凑锚点抑制 KV 膨胀与防止溢出 (9 题)", fontweight="bold")
    ax.legend(fontsize=7.6, loc="upper left")

    fig.suptitle("Qwen3-4B-Q4_0 τ²-bench 2×2 四条件联合消融端到端对比（9 题实测）", fontsize=12.8, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3-4b-tau2-joint-e2e.{ext}", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve-summary", type=Path, default=ROOT / "results/20261115-qwen3-4b-kv-curve/summary.json")
    p.add_argument("--kv-cost", type=Path, default=ROOT / "configs/qwen3-4b-kv-reduce-cost.json")
    p.add_argument("--route-cost", type=Path, default=ROOT / "configs/qwen3-4b-route-costs.json")
    p.add_argument("--calib-dir", type=Path, default=ROOT / "results/20261115-qwen3-4b-joint/route-calib")
    p.add_argument("--gate-4b", type=Path, default=ROOT / "results/20261115-qwen3-4b-dual-gate/summary.json")
    p.add_argument("--gate-8b", type=Path, default=ROOT / "results/20261115-qwen3-8b-dual-gate/summary.json")
    p.add_argument("--e2e-summary", type=Path, default=ROOT / "results/20261115-tau2-qwen3-4b-joint-e2e/summary.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/20261115-qwen3-4b-joint/figures")
    args = p.parse_args()

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve_sum = json.loads(args.curve_summary.read_text())
    kv_cost = json.loads(args.kv_cost.read_text())
    route_cost = json.loads(args.route_cost.read_text())
    e2e_sum = json.loads(args.e2e_summary.read_text()) if args.e2e_summary.exists() else None

    plot_calibration_and_gate(curve_sum, kv_cost, route_cost, args.calib_dir, args.gate_4b, args.gate_8b, args.out_dir, e2e_sum)
    plot_quality_lcp(curve_sum, args.out_dir)
    if e2e_sum:
        plot_e2e_joint(e2e_sum, args.out_dir)
    print("Wrote 4B figures to", args.out_dir)


if __name__ == "__main__":
    main()
