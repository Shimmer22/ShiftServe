#!/usr/bin/env python3
"""EXP-007: A2O 空档内 KV 整理 + 搬移到 NPU 就绪布局的耗时上界（本机 + 可选手机 memcpy）。

测什么（非真实 GUI-KV prune 算法）:
  - 体量估算: Qwen2.5-VL-3B LLM KV = 2(K+V)*n_layer*n_kv_head*head_dim*2B(f16)*n_tok
    与 EXP-006 日志一致: ctx4096 → HTP0 KV 144 MiB。
  - organize 上界: 按 token 截断保留、按层选择拷贝、整段字节 memcpy（模拟 prune/merge 的拷贝上界）。
  - move 上界: host memcpy / 页对齐拷贝到「NPU 就绪」目标 buffer（同尺寸；非 Hexagon DMA）。

不测: 真实 prune 分数、真实 Hexagon DMA/RPC、与整段 NPU 推理时间对比。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Qwen2.5-VL-3B (from EXP-006 vision_htp0 log / GGUF meta)
N_LAYER = 36
N_HEAD = 16
N_KV_HEAD = 2
N_EMBD = 2048
HEAD_DIM = N_EMBD // N_HEAD  # 128
DTYPE_BYTES = 2  # f16 KV

# GUI multi-step history scenarios (tokens kept after hypothetical prune)
SCENARIOS = [
    # name, n_tok_src, keep_frac (organize truncates to keep_frac * n_tok_src)
    ("gui_1step_M", 600, 1.0),          # ~560 img + ~40 text
    ("gui_3step_keep_last", 1800, 1 / 3),  # keep ~1 step
    ("gui_5step_keep_2", 3000, 2 / 5),
    ("ctx_512_full", 512, 1.0),
    ("ctx_1024_full", 1024, 1.0),
    ("ctx_2048_full", 2048, 1.0),
    ("ctx_4096_full", 4096, 1.0),  # matches EXP-006 HTP0 KV=144MiB
]

# A2O budgets from EXP-005 a2o_summary_table.md (ms)
A2O_BUDGETS = [
    ("UI-TARS-desktop (ADB)", 1345.4),
    ("Mobile-Agent-v1", 2546.6),
    ("Mobile-Agent-v3", 3546.6),
    ("AndroidLab", 5703.6),
    ("Mobile-Agent-v2/E", 6546.6),
    ("AppAgent", 12703.6),
    ("AndroidWorld M3A (wait only)", 2000.0),
]


def kv_bytes(n_tok: int, n_layer: int = N_LAYER) -> int:
    # 2 (K+V) * layers * kv_heads * head_dim * dtype
    return int(2 * n_layer * N_KV_HEAD * HEAD_DIM * DTYPE_BYTES * n_tok)


def kv_mib(n_tok: int, n_layer: int = N_LAYER) -> float:
    return kv_bytes(n_tok, n_layer) / (1024.0 * 1024.0)


def timed_median_ms(fn, repeats: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(xs, dtype=float)
    return {
        "n": int(len(a)),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def page_aligned_views(nbytes: int):
    psz = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    raw_src = np.empty(nbytes + psz, dtype=np.uint8)
    raw_dst = np.empty(nbytes + psz, dtype=np.uint8)
    off_s = (-raw_src.ctypes.data) % psz
    off_d = (-raw_dst.ctypes.data) % psz
    src = raw_src[off_s : off_s + nbytes]
    dst = raw_dst[off_d : off_d + nbytes]
    src.fill(0xA5)
    dst.fill(0)
    return src, dst, raw_src, raw_dst, psz


def bench_host(out_dir: Path, repeats: int) -> list[dict]:
    rows = []
    for name, n_tok, keep_frac in SCENARIOS:
        n_keep = max(1, int(round(n_tok * keep_frac)))
        src_b = kv_bytes(n_tok)
        keep_b = kv_bytes(n_keep)
        # per-layer slice size for layer-select organize
        per_layer_src = src_b // N_LAYER
        per_layer_keep = keep_b // N_LAYER
        keep_layers = N_LAYER  # select-all layers but truncate seq; also half-layer variant below

        src, dst, raw_src, raw_dst, psz = page_aligned_views(src_b)
        # target "NPU-ready" buffer same size as kept KV (page aligned)
        npu_raw = np.empty(keep_b + psz, dtype=np.uint8)
        off_n = (-npu_raw.ctypes.data) % psz
        npu = npu_raw[off_n : off_n + keep_b]
        npu.fill(0)

        # --- organize ops (upper bounds) ---
        def op_memcpy_full():
            np.copyto(dst, src)

        def op_trunc_token_copy():
            # keep first n_keep tokens: contiguous prefix of each layer's K/V
            # layout assumed [layer][kv][tok][dim] flattened → per-layer contiguous prefix
            for L in range(N_LAYER):
                s0 = L * per_layer_src
                d0 = L * per_layer_keep
                dst[d0 : d0 + per_layer_keep] = src[s0 : s0 + per_layer_keep]

        def op_layer_select_half():
            # keep even layers only, full seq (bytes = src_b/2) into dst prefix
            half = src_b // 2
            o = 0
            for L in range(0, N_LAYER, 2):
                s0 = L * per_layer_src
                dst[o : o + per_layer_src] = src[s0 : s0 + per_layer_src]
                o += per_layer_src
            assert o == half

        def op_move_page_aligned():
            # after organize, move kept bytes into NPU-ready buffer
            np.copyto(npu, dst[:keep_b])

        stats_full = timed_median_ms(op_memcpy_full, repeats)
        stats_trunc = timed_median_ms(op_trunc_token_copy, repeats)
        stats_layer = timed_median_ms(op_layer_select_half, repeats)
        # ensure dst has kept content for move
        op_trunc_token_copy()
        stats_move = timed_median_ms(op_move_page_aligned, repeats)

        # prep total upper = trunc organize + move (pragmatic path)
        prep_median = stats_trunc["median_ms"] + stats_move["median_ms"]
        prep_p90 = stats_trunc["p90_ms"] + stats_move["p90_ms"]

        row = {
            "device": "mac_host",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "scenario": name,
            "n_tok_src": n_tok,
            "n_tok_keep": n_keep,
            "keep_frac": keep_frac,
            "src_bytes": src_b,
            "keep_bytes": keep_b,
            "src_mib": round(src_b / (1024 * 1024), 3),
            "keep_mib": round(keep_b / (1024 * 1024), 3),
            "page_size": psz,
            "hexagon_dma": False,
            "organize_memcpy_full_median_ms": round(stats_full["median_ms"], 4),
            "organize_trunc_median_ms": round(stats_trunc["median_ms"], 4),
            "organize_trunc_p90_ms": round(stats_trunc["p90_ms"], 4),
            "organize_layer_half_median_ms": round(stats_layer["median_ms"], 4),
            "move_page_aligned_median_ms": round(stats_move["median_ms"], 4),
            "move_page_aligned_p90_ms": round(stats_move["p90_ms"], 4),
            "prep_trunc_plus_move_median_ms": round(prep_median, 4),
            "prep_trunc_plus_move_p90_ms": round(prep_p90, 4),
            "stats_full": stats_full,
            "stats_trunc": stats_trunc,
            "stats_layer": stats_layer,
            "stats_move": stats_move,
        }
        rows.append(row)
        print(
            f"[host] {name}: src={row['src_mib']}MiB keep={row['keep_mib']}MiB "
            f"trunc={row['organize_trunc_median_ms']}ms move={row['move_page_aligned_median_ms']}ms "
            f"prep={row['prep_trunc_plus_move_median_ms']}ms",
            flush=True,
        )
        # free large arrays
        del src, dst, raw_src, raw_dst, npu, npu_raw
    return rows


def phone_safe_to_run_large(adb: str, dev: str) -> tuple[bool, str]:
    """Refuse large phone buffers if llama is running or MemAvailable is low."""
    ps = subprocess.run(
        [adb, "-s", dev, "shell", "ps -A -o PID,NAME,ARGS 2>/dev/null || ps -A"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (ps.stdout or "") + (ps.stderr or "")
    has_llama = any("llama" in ln.lower() for ln in out.splitlines())
    # fallback: toybox pidof
    if not has_llama:
        for name in ("llama-mtmd-cli", "llama-cli", "llama-completion", "llama-server"):
            r = subprocess.run([adb, "-s", dev, "shell", f"pidof {name}"], capture_output=True, text=True, timeout=10)
            if (r.stdout or "").strip():
                has_llama = True
                break
    mem = subprocess.run(
        [adb, "-s", dev, "shell", "cat /proc/meminfo"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    avail_kb = None
    for ln in (mem.stdout or "").splitlines():
        if ln.startswith("MemAvailable:"):
            avail_kb = int(ln.split()[1])
            break
    avail_mib = (avail_kb or 0) / 1024.0
    note = f"llama_running={has_llama} MemAvailable_MiB={avail_mib:.1f}"
    # need headroom for 2 buffers + binary; skip >=16MiB tests if <400MiB avail or llama present
    ok_small_only = True
    ok_large = (not has_llama) and avail_mib >= 400
    return ok_large, note


def bench_phone(adb: str, dev: str, bin_host: Path, out_dir: Path, repeats: int) -> list[dict]:
    ok_large, note = phone_safe_to_run_large(adb, dev)
    remote = "/data/local/tmp/a2o_exp007_memcpy_bench"
    subprocess.run([adb, "-s", dev, "push", str(bin_host), remote], check=True, capture_output=True)
    subprocess.run([adb, "-s", dev, "shell", f"chmod 755 {remote}"], check=True, capture_output=True)

    # sizes: always try tiny; larger only if safe
    size_plan = [
        ("phone_1MiB", 1 * 1024 * 1024, 20),
        ("phone_4MiB", 4 * 1024 * 1024, 10),
        ("phone_16MiB", 16 * 1024 * 1024, 5),
    ]
    if ok_large:
        size_plan += [
            ("phone_36MiB_approx_1k_tok", kv_bytes(1024), 3),
            ("phone_72MiB_approx_2k_tok", kv_bytes(2048), 2),
            ("phone_144MiB_ctx4096", kv_bytes(4096), 1),
        ]
    rows = []
    for name, nbytes, iters in size_plan:
        # skip 16MiB+ when not ok_large
        if nbytes >= 16 * 1024 * 1024 and not ok_large:
            rows.append(
                {
                    "device": "phone_android",
                    "scenario": name,
                    "bytes": nbytes,
                    "mib": round(nbytes / (1024 * 1024), 3),
                    "skipped": True,
                    "skip_reason": note,
                    "hexagon_dma": False,
                }
            )
            print(f"[phone] SKIP {name}: {note}", flush=True)
            continue
        cmd = f"{remote} {nbytes} {iters} 1"
        p = subprocess.run(
            [adb, "-s", dev, "shell", cmd],
            capture_output=True,
            text=True,
            timeout=120,
        )
        line = (p.stdout or "").strip().splitlines()[-1] if p.stdout else ""
        err = (p.stderr or "").strip()
        parsed = {"raw": line, "stderr": err, "returncode": p.returncode}
        # parse per_ms=
        per_ms = None
        gib_s = None
        for part in line.split():
            if part.startswith("per_ms="):
                per_ms = float(part.split("=", 1)[1])
            if part.startswith("gib_s="):
                gib_s = float(part.split("=", 1)[1])
        row = {
            "device": "phone_android",
            "adb": dev,
            "scenario": name,
            "bytes": nbytes,
            "mib": round(nbytes / (1024 * 1024), 3),
            "iters": iters,
            "per_ms": per_ms,
            "gib_s": gib_s,
            "skipped": False,
            "safety_note": note,
            "hexagon_dma": False,
            "method": "userspace_memcpy_page_aligned",
            **parsed,
        }
        rows.append(row)
        print(f"[phone] {name}: {line or err}", flush=True)
    # cleanup remote binary only (no large leftover files)
    subprocess.run([adb, "-s", dev, "shell", f"rm -f {remote}"], capture_output=True)
    return rows


def write_comparison(host_rows: list[dict], phone_rows: list[dict], out_dir: Path, meta: dict):
    # pick representative host prep times
    by_sc = {r["scenario"]: r for r in host_rows}
    reps = [
        ("gui_1step_M", "单步≈600 tok 全量整理+搬移"),
        ("gui_5step_keep_2", "5 步历史截断保留约 2 步 + 搬移"),
        ("ctx_2048_full", "满 2048 tok KV 整理(截断恒等)+搬移"),
        ("ctx_4096_full", "满 4096 tok（=EXP-006 HTP0 KV 144MiB）整理+搬移"),
    ]
    lines = []
    lines.append("# EXP-007 对照表: A2O 预算 vs KV prep 耗时（上界）")
    lines.append("")
    lines.append(f"- 测量时间: {meta.get('timestamp_asia_shanghai')}")
    lines.append(f"- Host: {meta.get('host_platform')}（numpy page-aligned memcpy；**非** Hexagon DMA）")
    lines.append(
        f"- Phone: adb `{meta.get('adb')}`；用户态 memcpy microbench；"
        f" Hexagon DMA=**否**；安全注记: {meta.get('phone_safety')}"
    )
    lines.append(
        "- 模型体量公式: `2 * 36 * 2 * 128 * 2 * n_tok` bytes（f16）；"
        "ctx4096 → 144.00 MiB，与 EXP-006 `HTP0 KV buffer size = 144.00 MiB` 一致。"
    )
    lines.append(
        "- prep = `organize_trunc`（按 token 截断的逐层前缀拷贝）+ `move_page_aligned`（搬到页对齐目标 buffer）。"
        " 这是拷贝上界，**不是**真实 GUI-KV prune/merge 算法。"
    )
    lines.append("- HTP0-REPACK≈1488 MiB 是权重 repack，不是 KV；本表不拿它当搬移体量。")
    lines.append("")
    lines.append("## Host prep 摘要")
    lines.append("")
    lines.append("| 场景 | src MiB | keep MiB | trunc 中位 ms | move 中位 ms | prep 中位 ms | prep p90 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, _ in reps:
        r = by_sc[key]
        lines.append(
            f"| {key} | {r['src_mib']} | {r['keep_mib']} | "
            f"{r['organize_trunc_median_ms']} | {r['move_page_aligned_median_ms']} | "
            f"{r['prep_trunc_plus_move_median_ms']} | {r['prep_trunc_plus_move_p90_ms']} |"
        )
    lines.append("")
    lines.append("## 各框架 A2O 预算 vs prep（是否落在预算内）")
    lines.append("")
    lines.append(
        "| 框架 | A2O 预算 ms | vs gui_1step prep | vs gui_5step_keep_2 prep | vs ctx4096 prep | 备注 |"
    )
    lines.append("|---|---:|---|---|---|---|")
    p1 = by_sc["gui_1step_M"]["prep_trunc_plus_move_median_ms"]
    p5 = by_sc["gui_5step_keep_2"]["prep_trunc_plus_move_median_ms"]
    p4 = by_sc["ctx_4096_full"]["prep_trunc_plus_move_median_ms"]

    def fit(budget, prep):
        return "是" if prep <= budget else "否"

    for name, budget in A2O_BUDGETS:
        lines.append(
            f"| {name} | {budget} | {fit(budget, p1)} ({p1}ms) | {fit(budget, p5)} ({p5}ms) | "
            f"{fit(budget, p4)} ({p4}ms) | EXP-005 推算预算 |"
        )
    lines.append("")
    lines.append("## Phone memcpy（同尺寸量级探测；非 DMA）")
    lines.append("")
    if not phone_rows:
        lines.append("未跑手机侧。")
    else:
        lines.append("| 场景 | MiB | per_ms | GiB/s | skipped | 说明 |")
        lines.append("|---|---:|---:|---:|---|---|")
        for r in phone_rows:
            lines.append(
                f"| {r.get('scenario')} | {r.get('mib')} | {r.get('per_ms')} | {r.get('gib_s')} | "
                f"{r.get('skipped')} | {r.get('skip_reason') or r.get('method') or ''} |"
            )
    lines.append("")
    lines.append("## 关键路径增量（解释）")
    lines.append("")
    lines.append(
        "若把 prep 放进动作后空档：在 host 拷贝上界下，gui 单步/多步截断 prep 远小于最短 A2O（UI-TARS≈1.3s）；"
        "满 ctx4096 双倍拷贝量级仍通常远小于多数框架预算。真正上关键路径时还会叠加调度/锁/与截图并发，"
        "以及真实 prune 算分与 Hexagon 侧布局转换——那些**未测**，故本表只回答「纯整理+搬移拷贝上界 vs A2O」。"
    )
    path = out_dir / "a2o_vs_kv_prep_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--adb", default="/Users/hui/Library/Android/sdk/platform-tools/adb")
    ap.add_argument("--device", default="10.29.11.213:42629")
    ap.add_argument("--skip-phone", action="store_true")
    ap.add_argument(
        "--phone-bin",
        type=Path,
        default=Path(__file__).resolve().parent / "native" / "memcpy_bench_android",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Asia/Shanghai timestamp (box/Mac may already be set; still label)
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    host_rows = bench_host(args.out_dir, args.repeats)

    phone_rows: list[dict] = []
    phone_safety = "skipped_by_flag"
    if not args.skip_phone:
        ok_large, phone_safety = phone_safe_to_run_large(args.adb, args.device)
        phone_rows = bench_phone(args.adb, args.device, args.phone_bin, args.out_dir, args.repeats)

    meta = {
        "timestamp_asia_shanghai": ts,
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
        "model": {
            "name": "Qwen2.5-VL-3B",
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_kv_head": N_KV_HEAD,
            "n_embd": N_EMBD,
            "head_dim": HEAD_DIM,
            "kv_dtype": "f16",
            "bytes_per_token": kv_bytes(1),
            "ctx4096_mib_formula": round(kv_mib(4096), 3),
            "ctx4096_mib_exp006_log": 144.0,
        },
        "hexagon_dma": False,
        "adb": args.device,
        "phone_safety": phone_safety,
        "repeats": args.repeats,
        "a2o_budget_source": "EXP-005-a2o-fixed-wait/results/a2o_summary_table.md",
        "kv_size_source": "EXP-006-hexagon-smoke/results/vision_htp0_20261103.log (HTP0 KV 144MiB @ ctx4096)",
    }

    # flatten CSV
    csv_path = args.out_dir / "host_kv_prep_timings.csv"
    flat_keys = [
        "device",
        "scenario",
        "n_tok_src",
        "n_tok_keep",
        "keep_frac",
        "src_mib",
        "keep_mib",
        "organize_memcpy_full_median_ms",
        "organize_trunc_median_ms",
        "organize_trunc_p90_ms",
        "organize_layer_half_median_ms",
        "move_page_aligned_median_ms",
        "move_page_aligned_p90_ms",
        "prep_trunc_plus_move_median_ms",
        "prep_trunc_plus_move_p90_ms",
        "hexagon_dma",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys)
        w.writeheader()
        for r in host_rows:
            w.writerow({k: r.get(k) for k in flat_keys})

    phone_csv = args.out_dir / "phone_memcpy_timings.csv"
    if phone_rows:
        pk = [
            "device",
            "scenario",
            "mib",
            "bytes",
            "iters",
            "per_ms",
            "gib_s",
            "skipped",
            "hexagon_dma",
            "method",
            "skip_reason",
            "safety_note",
            "raw",
        ]
        with phone_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pk, extrasaction="ignore")
            w.writeheader()
            for r in phone_rows:
                w.writerow(r)

    summary = {"meta": meta, "host": host_rows, "phone": phone_rows}
    json_path = args.out_dir / "kv_prep_summary.json"
    # strip heavy nested stats duplication ok to keep
    json.dump(summary, json_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    table_path = write_comparison(host_rows, phone_rows, args.out_dir, meta)
    print("wrote", csv_path)
    print("wrote", json_path)
    print("wrote", table_path)
    if phone_rows:
        print("wrote", phone_csv)


if __name__ == "__main__":
    main()
