#!/usr/bin/env python3
"""EXP-008: GUI 历史 KV 策略 — keep(action_prev + 任务关注度加权) + history 相似度 merge。

本轮：host 侧策略 + 计时；合成 KV（Qwen2.5-VL-3B 体量公式）；NPU DMA 预热为 stub。
无真实 dump 时用合成数据，结果中标明 synthetic=true。
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Qwen2.5-VL-3B LLM KV (f16)，与 EXP-006/007 一致
N_LAYER = 36
N_HEAD = 16
N_KV_HEAD = 2
N_EMBD = 2048
HEAD_DIM = N_EMBD // N_HEAD  # 128
DTYPE_BYTES = 2  # f16
BYTES_PER_TOKEN = 2 * N_LAYER * N_KV_HEAD * HEAD_DIM * DTYPE_BYTES  # 36864

# 聚类用特征维（合成 KV 的 per-token 代理向量，非真实 head 展开；算法与计时用）
FEAT_DIM = 64

# A2O 预算，摘自 EXP-005 a2o_summary_table.md（推算）
A2O_BUDGETS = [
    ("UI-TARS-desktop (ADB)", 1345.4),
    ("Mobile-Agent-v1", 2546.6),
    ("Mobile-Agent-v3", 3546.6),
    ("AndroidLab", 5703.6),
    ("Mobile-Agent-v2/E", 6546.6),
    ("AppAgent", 12703.6),
    ("AndroidWorld M3A (wait only)", 2000.0),
]


def kv_bytes(n_tok: int) -> int:
    return int(BYTES_PER_TOKEN * n_tok)


def kv_mib(n_tok: int) -> float:
    return kv_bytes(n_tok) / (1024.0 * 1024.0)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def feature_checksum(feat: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(feat).tobytes()).hexdigest()[:16]


@dataclass
class KVSegment:
    seg_id: str
    kind: str  # task | action_prev | history
    n_tokens: int
    # (n_tokens, FEAT_DIM) 代理特征；真实 dump 时可由层均值等派生
    features: np.ndarray
    # 可选：逐 token 关注度 [0,1]；仅 task 常用
    importance: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def mean_feature(self) -> np.ndarray:
        return self.features.mean(axis=0)

    def bytes(self) -> int:
        return kv_bytes(self.n_tokens)


@dataclass
class Disposition:
    seg_id: str
    kind: str
    action: str  # keep_full | keep_partial | merge_in | merged_rep
    n_tokens_in: int
    n_tokens_out: int
    detail: dict = field(default_factory=dict)


def npu_dma_warmup_move(src: Any = None, dst: Any = None) -> dict:
    """NPU DMA / fastrpc 预热搬移 — 本轮 stub。下一步：接入 Hexagon DMA。"""
    # TODO: Hexagon DMA / fastrpc layout convert + warmup
    return {
        "status": "stub",
        "hexagon_dma": False,
        "elapsed_ms": 0.0,
        "note": "NPU DMA warmup not implemented this round",
    }


def make_synthetic_segments(
    n_history: int = 6,
    task_tokens: int = 48,
    action_tokens: int = 32,
    history_tokens: int = 80,
    seed: int = 42,
    n_task_clusters: int = 2,
) -> list[KVSegment]:
    """合成与 Qwen2.5-VL-3B token 体量一致的段列表（特征为代理，标明 synthetic）。"""
    rng = np.random.default_rng(seed)
    segs: list[KVSegment] = []

    # task: 两簇 token（高关注 / 低关注），importance 与到 task 中心的相似度相关
    centers = rng.normal(size=(n_task_clusters, FEAT_DIM)).astype(np.float32)
    centers[0] *= 1.5  # 主任务方向
    assign = rng.integers(0, n_task_clusters, size=task_tokens)
    task_feat = centers[assign] + 0.15 * rng.normal(size=(task_tokens, FEAT_DIM)).astype(np.float32)
    task_center = centers[0]
    imp = np.array(
        [max(0.0, cosine_sim(task_feat[i], task_center)) for i in range(task_tokens)],
        dtype=np.float32,
    )
    # 拉开分布：高簇抬高、低簇压低
    imp = np.where(assign == 0, np.clip(imp * 1.2 + 0.2, 0, 1), np.clip(imp * 0.5, 0, 1)).astype(
        np.float32
    )
    segs.append(
        KVSegment(
            seg_id="task",
            kind="task",
            n_tokens=task_tokens,
            features=task_feat,
            importance=imp,
            meta={"synthetic": True, "task_center_norm": float(np.linalg.norm(task_center))},
        )
    )

    # action_prev: 独立方向，必须整段保留
    act = rng.normal(size=(action_tokens, FEAT_DIM)).astype(np.float32) + np.array(
        [3.0] + [0.0] * (FEAT_DIM - 1), dtype=np.float32
    )
    segs.append(
        KVSegment(
            seg_id="action_prev",
            kind="action_prev",
            n_tokens=action_tokens,
            features=act,
            meta={"synthetic": True},
        )
    )

    # history: 若干相近簇，便于 merge
    n_hist_clusters = max(2, n_history // 3)
    hist_centers = rng.normal(size=(n_hist_clusters, FEAT_DIM)).astype(np.float32)
    for i in range(n_history):
        c = hist_centers[i % n_hist_clusters]
        noise = 0.2 * rng.normal(size=(history_tokens, FEAT_DIM)).astype(np.float32)
        feat = c + noise
        segs.append(
            KVSegment(
                seg_id=f"history_{i}",
                kind="history",
                n_tokens=history_tokens,
                features=feat,
                meta={"synthetic": True, "planted_cluster": int(i % n_hist_clusters)},
            )
        )
    return segs


def load_segments_from_npz(path: Path) -> list[KVSegment]:
    """从 npz 加载。期望 keys: seg_ids, kinds, features_{id}, 可选 importance_{id}/n_tokens_{id}。"""
    data = np.load(path, allow_pickle=True)
    seg_ids = list(data["seg_ids"])
    kinds = list(data["kinds"])
    segs = []
    for sid, kind in zip(seg_ids, kinds):
        sid = str(sid)
        kind = str(kind)
        feat = data[f"features_{sid}"]
        imp = data[f"importance_{sid}"] if f"importance_{sid}" in data.files else None
        n_tok = int(data[f"n_tokens_{sid}"]) if f"n_tokens_{sid}" in data.files else feat.shape[0]
        segs.append(
            KVSegment(
                seg_id=sid,
                kind=kind,
                n_tokens=n_tok,
                features=feat.astype(np.float32),
                importance=None if imp is None else imp.astype(np.float32),
                meta={"synthetic": False, "source": str(path)},
            )
        )
    return segs


def task_attention_scores(seg: KVSegment) -> np.ndarray:
    """任务段关注度：优先用 importance；否则与 task 均值向量的余弦相似度映射到 [0,1]。"""
    if seg.importance is not None:
        return seg.importance.astype(np.float32)
    center = seg.mean_feature()
    sims = np.array(
        [cosine_sim(seg.features[i], center) for i in range(seg.n_tokens)], dtype=np.float32
    )
    # 余弦 [-1,1] → [0,1]
    return ((sims + 1.0) * 0.5).astype(np.float32)


def keep_task_by_attention(
    seg: KVSegment,
    high_thr: float = 0.55,
    mid_thr: float = 0.35,
) -> tuple[Optional[KVSegment], Optional[KVSegment], Disposition]:
    """高分保留；中低分进 merge 池（作为 history 风格段）。"""
    scores = task_attention_scores(seg)
    high_mask = scores >= high_thr
    low_mask = ~high_mask  # 中低都进 merge 池（mid_thr 记入元数据供分析）
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    kept = None
    to_merge = None
    if n_high > 0:
        kept = KVSegment(
            seg_id=f"{seg.seg_id}_kept",
            kind="task",
            n_tokens=n_high,
            features=seg.features[high_mask],
            importance=scores[high_mask],
            meta={**seg.meta, "parent": seg.seg_id, "keep_reason": "high_attention"},
        )
    if n_low > 0:
        to_merge = KVSegment(
            seg_id=f"{seg.seg_id}_merge_pool",
            kind="history",  # 进 merge 池
            n_tokens=n_low,
            features=seg.features[low_mask],
            importance=scores[low_mask],
            meta={
                **seg.meta,
                "parent": seg.seg_id,
                "from_task_low_attention": True,
                "mid_thr": mid_thr,
                "high_thr": high_thr,
            },
        )
    disp = Disposition(
        seg_id=seg.seg_id,
        kind=seg.kind,
        action="keep_partial" if n_high and n_low else ("keep_full" if n_high else "merge_in"),
        n_tokens_in=seg.n_tokens,
        n_tokens_out=n_high,
        detail={
            "n_high": n_high,
            "n_to_merge_pool": n_low,
            "high_thr": high_thr,
            "mid_thr": mid_thr,
            "score_mean": float(scores.mean()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
        },
    )
    return kept, to_merge, disp


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 30) -> np.ndarray:
    """简单 Lloyd k-means，返回每个点的簇 id。纯 numpy。"""
    n = X.shape[0]
    k = max(1, min(k, n))
    # 初始化：随机选 k 个点
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        # assign
        # (n, k) distances
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            else:
                centers[j] = X[int(rng.integers(0, n))]
    return labels


def merge_history_segments(
    segs: list[KVSegment],
    n_clusters: Optional[int] = None,
    seed: int = 0,
) -> tuple[list[KVSegment], list[Disposition], dict]:
    """对 history 段按均值特征 k-means；每类合并为加权中心代表（token 数=成员之和的均值取整，特征=加权均值）。"""
    if not segs:
        return [], [], {"n_in": 0, "n_out": 0, "n_clusters": 0}

    X = np.stack([s.mean_feature() for s in segs], axis=0).astype(np.float64)
    weights = np.array([s.n_tokens for s in segs], dtype=np.float64)
    n = len(segs)
    if n_clusters is None:
        # 默认压到约 ceil(sqrt(n)) 或至少 1
        n_clusters = max(1, int(np.ceil(np.sqrt(n))))
    n_clusters = max(1, min(n_clusters, n))
    rng = np.random.default_rng(seed)
    labels = _kmeans(X, n_clusters, rng)

    out: list[KVSegment] = []
    disps: list[Disposition] = []
    for j in range(n_clusters):
        members = [segs[i] for i in range(n) if labels[i] == j]
        if not members:
            continue
        w = np.array([m.n_tokens for m in members], dtype=np.float64)
        # 加权中心：按 token 数加权的特征均值；代表长度取成员 token 均值（压缩）
        feats = np.stack([m.mean_feature() for m in members], axis=0)
        center = (feats * w[:, None]).sum(axis=0) / w.sum()
        # 代表段：用中心重复 n_rep 次作为特征（n_rep = 均值 token，至少 1）
        n_rep = max(1, int(round(w.mean())))
        rep_feat = np.tile(center.astype(np.float32)[None, :], (n_rep, 1))
        member_ids = [m.seg_id for m in members]
        rep_id = f"merged_{j}_" + "_".join(member_ids[:3])
        if len(member_ids) > 3:
            rep_id += f"_plus{len(member_ids)-3}"
        out.append(
            KVSegment(
                seg_id=rep_id,
                kind="history",
                n_tokens=n_rep,
                features=rep_feat,
                meta={
                    "synthetic": any(m.meta.get("synthetic") for m in members),
                    "merged_from": member_ids,
                    "tokens_in_sum": int(w.sum()),
                    "cluster": j,
                },
            )
        )
        for m in members:
            disps.append(
                Disposition(
                    seg_id=m.seg_id,
                    kind=m.kind,
                    action="merge_in",
                    n_tokens_in=m.n_tokens,
                    n_tokens_out=0,
                    detail={"cluster": j, "rep_id": rep_id, "rep_n_tokens": n_rep},
                )
            )
        disps.append(
            Disposition(
                seg_id=rep_id,
                kind="history",
                action="merged_rep",
                n_tokens_in=int(w.sum()),
                n_tokens_out=n_rep,
                detail={"cluster": j, "members": member_ids, "compression_in_cluster": float(n_rep / w.sum())},
            )
        )
    stats = {
        "n_in": n,
        "n_out": len(out),
        "n_clusters": len(out),
        "tokens_in": int(weights.sum()),
        "tokens_out": int(sum(s.n_tokens for s in out)),
    }
    return out, disps, stats


def apply_policy(
    segments: list[KVSegment],
    high_thr: float = 0.55,
    mid_thr: float = 0.35,
    n_clusters: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """主策略：action_prev 整段 keep；task 关注度加权 keep/merge；其余 history 聚类 merge。"""
    t0 = time.perf_counter()
    kept: list[KVSegment] = []
    merge_pool: list[KVSegment] = []
    dispositions: list[Disposition] = []
    action_prev_ref = None

    for seg in segments:
        if seg.kind == "action_prev":
            kept.append(seg)
            action_prev_ref = {
                "seg_id": seg.seg_id,
                "n_tokens": seg.n_tokens,
                "checksum": feature_checksum(seg.features),
                "bytes": seg.bytes(),
            }
            dispositions.append(
                Disposition(
                    seg_id=seg.seg_id,
                    kind=seg.kind,
                    action="keep_full",
                    n_tokens_in=seg.n_tokens,
                    n_tokens_out=seg.n_tokens,
                    detail={"policy": "action_prev_immutable"},
                )
            )
        elif seg.kind == "task":
            k, m, d = keep_task_by_attention(seg, high_thr=high_thr, mid_thr=mid_thr)
            dispositions.append(d)
            if k is not None:
                kept.append(k)
            if m is not None:
                merge_pool.append(m)
        else:
            merge_pool.append(seg)

    t_keep = time.perf_counter()
    merged, merge_disps, merge_stats = merge_history_segments(
        merge_pool, n_clusters=n_clusters, seed=seed
    )
    t_merge = time.perf_counter()
    dispositions.extend(merge_disps)

    out_segs = kept + merged
    # 校验 action_prev 未被改动
    ap_ok = True
    ap_detail = {}
    if action_prev_ref is not None:
        found = [s for s in out_segs if s.seg_id == action_prev_ref["seg_id"]]
        if len(found) != 1:
            ap_ok = False
            ap_detail["error"] = "action_prev missing or duplicated in output"
        else:
            s = found[0]
            ap_ok = (
                s.n_tokens == action_prev_ref["n_tokens"]
                and feature_checksum(s.features) == action_prev_ref["checksum"]
            )
            ap_detail = {
                "n_tokens_unchanged": s.n_tokens == action_prev_ref["n_tokens"],
                "checksum_unchanged": feature_checksum(s.features) == action_prev_ref["checksum"],
                "checksum": feature_checksum(s.features),
            }

    tokens_in = sum(s.n_tokens for s in segments)
    tokens_out = sum(s.n_tokens for s in out_segs)
    bytes_in = sum(s.bytes() for s in segments)
    bytes_out = sum(s.bytes() for s in out_segs)

    # 可选：把输出段特征拼成 buffer 做一次 host memcpy 计时（体量按真实 KV bytes）
    # 这里只计「整理决策+merge 计算」；另测按 keep 字节的 memcpy 作为搬移上界
    move_ms = None
    if bytes_out > 0:
        # 用 uint8 buffer 模拟按压缩后体量的 page-ish memcpy
        src = np.empty(bytes_out, dtype=np.uint8)
        dst = np.empty(bytes_out, dtype=np.uint8)
        src.fill(1)
        t_m0 = time.perf_counter()
        np.copyto(dst, src)
        move_ms = (time.perf_counter() - t_m0) * 1e3

    dma_stub = npu_dma_warmup_move()

    result = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "mib_in": bytes_in / (1024 * 1024),
        "mib_out": bytes_out / (1024 * 1024),
        "compression_ratio_tokens": float(tokens_out / tokens_in) if tokens_in else 0.0,
        "compression_ratio_bytes": float(bytes_out / bytes_in) if bytes_in else 0.0,
        "action_prev_intact": ap_ok,
        "action_prev_detail": ap_detail,
        "merge_stats": merge_stats,
        "dispositions": [asdict(d) for d in dispositions],
        "out_segment_ids": [s.seg_id for s in out_segs],
        "out_segment_kinds": [s.kind for s in out_segs],
        "out_segment_tokens": [s.n_tokens for s in out_segs],
        "timing_ms": {
            "keep_phase": (t_keep - t0) * 1e3,
            "merge_phase": (t_merge - t_keep) * 1e3,
            "policy_total": (t_merge - t0) * 1e3,
            "host_memcpy_out_bytes": move_ms,
            "npu_dma_warmup": dma_stub["elapsed_ms"],
        },
        "npu_dma": dma_stub,
        "out_segments": out_segs,  # 内部用；序列化前会剔
    }
    return result


def timed_policy(segments: list[KVSegment], repeats: int, warmup: int, **kwargs) -> dict:
    for _ in range(warmup):
        apply_policy(segments, **kwargs)
    times = []
    last = None
    for _ in range(repeats):
        # 深拷贝特征，避免原地污染（本实现不改输入，但保持干净）
        segs = [
            KVSegment(
                seg_id=s.seg_id,
                kind=s.kind,
                n_tokens=s.n_tokens,
                features=s.features.copy(),
                importance=None if s.importance is None else s.importance.copy(),
                meta=dict(s.meta),
            )
            for s in segments
        ]
        t0 = time.perf_counter()
        last = apply_policy(segs, **kwargs)
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(times, dtype=float)
    assert last is not None
    last["timing_repeats_ms"] = {
        "n": int(len(a)),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }
    return last


def compare_a2o(prep_ms: float) -> list[dict]:
    rows = []
    for name, budget in A2O_BUDGETS:
        rows.append(
            {
                "framework": name,
                "a2o_budget_ms": budget,
                "prep_ms": prep_ms,
                "fits_budget": bool(prep_ms < budget),
                "budget_source": "EXP-005-a2o-fixed-wait/results/a2o_summary_table.md",
            }
        )
    return rows


def serialize_result(result: dict) -> dict:
    out = {k: v for k, v in result.items() if k != "out_segments"}
    return out


def run_bench(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.npz:
        segments = load_segments_from_npz(Path(args.npz))
        # npz 默认视为非合成；可用 --force-synthetic 覆盖标记
        synthetic = bool(args.force_synthetic)
    else:
        segments = make_synthetic_segments(
            n_history=args.n_history,
            task_tokens=args.task_tokens,
            action_tokens=args.action_tokens,
            history_tokens=args.history_tokens,
            seed=args.seed,
        )
        synthetic = True

    result = timed_policy(
        segments,
        repeats=args.repeats,
        warmup=args.warmup,
        high_thr=args.high_thr,
        mid_thr=args.mid_thr,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    prep_ms = float(result["timing_repeats_ms"]["median_ms"])
    # prep 含策略；加上 host memcpy 中位近似（单次已在 result 里）
    memcpy_ms = result["timing_ms"].get("host_memcpy_out_bytes") or 0.0
    prep_plus_move = prep_ms + float(memcpy_ms)
    a2o = compare_a2o(prep_plus_move)

    summary = {
        "meta": {
            "timestamp_asia_shanghai": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
            "synthetic_kv": synthetic,
            "feat_dim_proxy": FEAT_DIM,
            "model_bytes_per_token": BYTES_PER_TOKEN,
            "model": {
                "name": "Qwen2.5-VL-3B",
                "n_layer": N_LAYER,
                "n_kv_head": N_KV_HEAD,
                "head_dim": HEAD_DIM,
                "kv_dtype": "f16",
                "note": "字节按真实 KV 公式计；聚类特征为 FEAT_DIM 维代理（非真实 dump）"
                if synthetic
                else "特征来自 npz",
            },
            "policy": {
                "action_prev": "keep_full immutable",
                "task": f"attention-weighted keep if score>={args.high_thr}; else merge pool",
                "history": "k-means merge to weighted centroid",
                "high_thr": args.high_thr,
                "mid_thr": args.mid_thr,
                "n_clusters": args.n_clusters,
            },
            "npu_dma": "stub",
            "a2o_budget_source": "EXP-005-a2o-fixed-wait/results/a2o_summary_table.md",
            "input_segments": [
                {
                    "seg_id": s.seg_id,
                    "kind": s.kind,
                    "n_tokens": s.n_tokens,
                    "mib": kv_mib(s.n_tokens),
                }
                for s in segments
            ],
        },
        "result": serialize_result(result),
        "a2o_comparison": a2o,
        "prep_plus_host_memcpy_ms": prep_plus_move,
    }

    json_path = out_dir / "kv_keep_merge_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # dispositions CSV
    disp_path = out_dir / "dispositions.csv"
    with open(disp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seg_id", "kind", "action", "n_tokens_in", "n_tokens_out", "detail_json"],
        )
        w.writeheader()
        for d in result["dispositions"]:
            w.writerow(
                {
                    "seg_id": d["seg_id"],
                    "kind": d["kind"],
                    "action": d["action"],
                    "n_tokens_in": d["n_tokens_in"],
                    "n_tokens_out": d["n_tokens_out"],
                    "detail_json": json.dumps(d.get("detail", {}), ensure_ascii=False),
                }
            )

    # timing CSV
    timing_path = out_dir / "timings.csv"
    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ms"])
        for k, v in result["timing_ms"].items():
            w.writerow([k, v])
        for k, v in result["timing_repeats_ms"].items():
            if k != "n":
                w.writerow([f"repeat_{k}", v])
        w.writerow(["prep_plus_host_memcpy_median_approx", prep_plus_move])

    # a2o CSV
    a2o_path = out_dir / "a2o_comparison.csv"
    with open(a2o_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["framework", "a2o_budget_ms", "prep_ms", "fits_budget", "budget_source"]
        )
        w.writeheader()
        for row in a2o:
            w.writerow(row)

    # markdown 摘要
    md_path = out_dir / "summary.md"
    lines = [
        "# EXP-008 KV keep(A)+任务关注度 + history merge — 结果摘要",
        "",
        f"- 时间: {summary['meta']['timestamp_asia_shanghai']}（Asia/Shanghai）",
        f"- **合成 KV**: `{synthetic}`（无真实 dump；特征维={FEAT_DIM} 代理；字节按 Qwen2.5-VL-3B 公式）",
        f"- tokens in→out: {result['tokens_in']} → {result['tokens_out']} "
        f"（压缩比 {result['compression_ratio_tokens']:.3f}）",
        f"- MiB in→out: {result['mib_in']:.3f} → {result['mib_out']:.3f} "
        f"（压缩比 {result['compression_ratio_bytes']:.3f}）",
        f"- action_prev 未破坏: `{result['action_prev_intact']}`",
        f"- 策略耗时中位: {result['timing_repeats_ms']['median_ms']:.3f} ms "
        f"（p90 {result['timing_repeats_ms']['p90_ms']:.3f}）",
        f"- + host memcpy(out bytes): ≈ {prep_plus_move:.3f} ms",
        f"- NPU DMA: stub（未实现）",
        "",
        "## vs A2O 预算（EXP-005）",
        "",
        "| 框架 | 预算 ms | prep+memcpy ms | 落入空档 |",
        "|---|---:|---:|---|",
    ]
    for row in a2o:
        lines.append(
            f"| {row['framework']} | {row['a2o_budget_ms']} | {row['prep_ms']:.3f} | "
            f"{'是' if row['fits_budget'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            f"- `{json_path.name}`",
            f"- `{disp_path.name}`",
            f"- `{timing_path.name}`",
            f"- `{a2o_path.name}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("meta", "prep_plus_host_memcpy_ms") if k in summary} | {
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "compression_ratio_tokens": result["compression_ratio_tokens"],
        "mib_in": result["mib_in"],
        "mib_out": result["mib_out"],
        "action_prev_intact": result["action_prev_intact"],
        "median_ms": result["timing_repeats_ms"]["median_ms"],
        "prep_plus_host_memcpy_ms": prep_plus_move,
        "all_fit_a2o": all(r["fits_budget"] for r in a2o),
        "out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXP-008 KV keep/merge policy bench")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--npz", type=str, default=None, help="可选真实/派生 KV 段 npz")
    p.add_argument("--force-synthetic", action="store_true")
    p.add_argument("--n-history", type=int, default=6)
    p.add_argument("--task-tokens", type=int, default=48)
    p.add_argument("--action-tokens", type=int, default=32)
    p.add_argument("--history-tokens", type=int, default=80)
    p.add_argument("--high-thr", type=float, default=0.55)
    p.add_argument("--mid-thr", type=float, default=0.35)
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--warmup", type=int, default=3)
    return p


def main():
    args = build_argparser().parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
