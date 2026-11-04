#!/usr/bin/env python3
"""小测试：keep/merge 不破坏 action_prev；任务高关注保留；history 被压缩。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_keep_merge import (
    apply_policy,
    feature_checksum,
    make_synthetic_segments,
    npu_dma_warmup_move,
)


def test_action_prev_intact():
    segs = make_synthetic_segments(n_history=6, seed=7)
    ap = next(s for s in segs if s.kind == "action_prev")
    before_n = ap.n_tokens
    before_cs = feature_checksum(ap.features)
    before_feat = ap.features.copy()
    result = apply_policy(segs, high_thr=0.55, n_clusters=2, seed=7)
    assert result["action_prev_intact"] is True
    out_ap = next(s for s in result["out_segments"] if s.seg_id == "action_prev")
    assert out_ap.n_tokens == before_n
    assert feature_checksum(out_ap.features) == before_cs
    np.testing.assert_array_equal(out_ap.features, before_feat)
    # dispositions 含 keep_full
    ap_disps = [d for d in result["dispositions"] if d["seg_id"] == "action_prev"]
    assert len(ap_disps) == 1 and ap_disps[0]["action"] == "keep_full"
    print("PASS test_action_prev_intact")


def test_task_attention_partial_keep():
    segs = make_synthetic_segments(n_history=4, seed=3)
    result = apply_policy(segs, high_thr=0.55, mid_thr=0.35, n_clusters=2, seed=3)
    task_disps = [d for d in result["dispositions"] if d["seg_id"] == "task"]
    assert len(task_disps) == 1
    d = task_disps[0]
    assert d["detail"]["n_high"] >= 1
    assert d["n_tokens_out"] == d["detail"]["n_high"]
    # 输出中应有 task_kept
    kinds = result["out_segment_kinds"]
    assert "task" in kinds
    print("PASS test_task_attention_partial_keep", d["detail"])


def test_history_merged_and_compressed():
    segs = make_synthetic_segments(n_history=6, history_tokens=40, seed=11)
    tokens_hist_in = sum(s.n_tokens for s in segs if s.kind == "history")
    result = apply_policy(segs, n_clusters=2, seed=11)
    assert result["tokens_out"] < result["tokens_in"]
    assert result["compression_ratio_tokens"] < 1.0
    # merge 池含 history；代表段数 = n_clusters
    assert result["merge_stats"]["n_out"] == 2
    assert result["merge_stats"]["tokens_out"] < tokens_hist_in + 48  # 宽松
    print(
        "PASS test_history_merged_and_compressed",
        f"in={result['tokens_in']} out={result['tokens_out']} ratio={result['compression_ratio_tokens']:.3f}",
    )


def test_dma_stub():
    r = npu_dma_warmup_move()
    assert r["status"] == "stub"
    assert r["hexagon_dma"] is False
    print("PASS test_dma_stub")


def main():
    test_action_prev_intact()
    test_task_attention_partial_keep()
    test_history_merged_and_compressed()
    test_dma_stub()
    print("ALL PASS")


if __name__ == "__main__":
    main()
