#!/usr/bin/env python3
"""Run EXP-008 policy on real-KV npz; emit keep_mask.txt over original positions.

Merge representatives cannot be written back as real KV cells without inventing tensors;
for writeback we KEEP:
  - all action_prev positions
  - task high-attention positions
  - for each merge cluster: the member segment whose mean feature is closest to centroid
    (keep that segment's tokens; drop other members). This approximates merge as
    "cluster exemplar keep" rather than centroid rewrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow importing EXP-008 module when run from experiment tree or via --exp008
def load_exp008(exp008_script: Path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("kv_keep_merge", exp008_script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["kv_keep_merge"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True, help="segments.json with pos ranges")
    ap.add_argument("--exp008", required=True, help="path to kv_keep_merge.py")
    ap.add_argument("--out-mask", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--high-thr", type=float, default=0.55)
    ap.add_argument("--mid-thr", type=float, default=0.35)
    ap.add_argument("--n-clusters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mod = load_exp008(Path(args.exp008))
    segs = mod.load_segments_from_npz(Path(args.npz))
    result = mod.apply_policy(
        segs, high_thr=args.high_thr, mid_thr=args.mid_thr, n_clusters=args.n_clusters, seed=args.seed
    )
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    n_pos = int(meta["n_tokens_total"])
    keep = np.zeros(n_pos, dtype=np.int32)

    # Map seg_id → pos range
    ranges = {s["seg_id"]: (int(s["pos0"]), int(s["pos1"])) for s in meta["segments"]}

    # action_prev full keep
    for s in segs:
        if s.kind == "action_prev":
            a, b = ranges[s.seg_id]
            keep[a:b] = 1

    # task kept high attention: need indices from disposition / recompute
    for s in segs:
        if s.kind != "task":
            continue
        scores = mod.task_attention_scores(s)
        a, b = ranges[s.seg_id]
        # features rows correspond to positions a..b
        for i, sc in enumerate(scores):
            if sc >= args.high_thr and a + i < b:
                keep[a + i] = 1

    # history merge → keep exemplar segment per cluster (closest to centroid)
    # Re-run merge labeling
    merge_pool = []
    for s in segs:
        if s.kind == "action_prev":
            continue
        if s.kind == "task":
            scores = mod.task_attention_scores(s)
            low_idx = np.where(scores < args.high_thr)[0]
            if len(low_idx) == 0:
                continue
            merge_pool.append(
                mod.KVSegment(
                    seg_id=f"{s.seg_id}_merge_pool",
                    kind="history",
                    n_tokens=len(low_idx),
                    features=s.features[low_idx],
                    importance=scores[low_idx],
                    meta={"parent": s.seg_id, "token_indices": low_idx.tolist(), "pos0": ranges[s.seg_id][0]},
                )
            )
        else:
            merge_pool.append(s)

    if merge_pool:
        X = np.stack([s.mean_feature() for s in merge_pool], axis=0).astype(np.float64)
        n = len(merge_pool)
        k = args.n_clusters if args.n_clusters is not None else max(1, int(np.ceil(np.sqrt(n))))
        k = max(1, min(k, n))
        rng = np.random.default_rng(args.seed)
        labels = mod._kmeans(X, k, rng)
        for j in range(k):
            members = [i for i in range(n) if labels[i] == j]
            if not members:
                continue
            # centroid
            w = np.array([merge_pool[i].n_tokens for i in members], dtype=np.float64)
            feats = np.stack([merge_pool[i].mean_feature() for i in members], axis=0)
            center = (feats * w[:, None]).sum(axis=0) / w.sum()
            # pick exemplar
            dists = ((feats - center) ** 2).sum(axis=1)
            ei = members[int(dists.argmin())]
            ex = merge_pool[ei]
            if "token_indices" in ex.meta:
                # task low-attention subset
                pos0 = int(ex.meta["pos0"])
                for ti in ex.meta["token_indices"]:
                    keep[pos0 + int(ti)] = 1
            else:
                a, b = ranges[ex.seg_id]
                keep[a:b] = 1

    # Assert action_prev intact
    ap_ok = True
    for s in segs:
        if s.kind == "action_prev":
            a, b = ranges[s.seg_id]
            if not np.all(keep[a:b] == 1):
                ap_ok = False

    Path(args.out_mask).write_text("\n".join(str(int(x)) for x in keep.tolist()) + "\n", encoding="utf-8")
    summary = {
        "n_pos": n_pos,
        "n_keep": int(keep.sum()),
        "n_drop": int(n_pos - keep.sum()),
        "compression_ratio_tokens": float(keep.sum() / n_pos) if n_pos else 0.0,
        "action_prev_intact": ap_ok,
        "policy_tokens_in": result["tokens_in"],
        "policy_tokens_out_logical": result["tokens_out"],
        "policy_compression_logical": result["compression_ratio_tokens"],
        "note": "writeback mask uses exemplar-keep for merge clusters (not centroid KV rewrite)",
        "bytes_in_est": result["bytes_in"],
        "bytes_out_est_mask": int(mod.BYTES_PER_TOKEN * int(keep.sum())),
        "action_prev_detail": result.get("action_prev_detail"),
        "timing_ms_policy": result.get("timing_ms"),
        "npu_dma": result.get("npu_dma"),
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not ap_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
