"""三路排名使用逐样本返回后实测时间，续跑拒绝跨计划样本。"""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p_cost_truth_sweep', ROOT / 'scripts/collect/p_cost_truth_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_winner_includes_decode_and_post_return_wait(tmp_path):
    paths = ['CC', 'HC-Full', 'HC-Compact']
    plan = dict(H=384, D=16, p_points=[32], reps=1, paths=paths, metadata=dict(gap_ms=500))
    stamp = sweep.ensure_plan(tmp_path, plan)
    truth = dict(version=1, D=16, gap_ms=500)
    tokens = list(range(16))
    for name, prefill, compact, decode, wait, prepare in [
        ('CC', 10, 0, 90, 0, 0),
        ('HC-Full', 5, 0, 80, 30, 530),
        ('HC-Compact', 6, 5, 40, 0, 450),
    ]:
        truth[name] = dict(prepare_ms=prepare, wait_ms=wait, restore_ms=0, prefill_ms=prefill,
                           compact_ms=compact, handoff_ms=0, decode_ms=decode,
                           post_return_ms=wait+prefill+compact+decode, decode_ids=tokens)
    row = dict(paired_truth=truth, cosine=0.999, hc_compact_evicted_count=100, hc_compact_handoff_bytes=512)
    sweep.validate_truth(row, paths, tokens, 500)
    sweep.save(sweep.sample_path(tmp_path, 'shared', 32, 0),
               dict(plan_sha256=stamp, mode='shared', H=384, P=32, D=16, rep=0, row=row))
    summary, = sweep.summarize_mode(tmp_path, plan, stamp, 'shared')
    assert summary['winner_med'] == summary['winner_trim'] == 'HC-Compact'
    assert summary['hc_full_post_return_ms_med'] == 115
    assert summary['hc_compact_post_return_ms_med'] == 51

    truth['HC-Full']['post_return_ms'] -= 30
    with pytest.raises(ValueError, match='阶段之和'):
        sweep.validate_truth(row, paths, tokens, 500)


def test_resume_rejects_changed_build_and_unbound_sample(tmp_path):
    plan = dict(build=dict(build_id='old-label', artifacts=dict(agent='a')), gap_ms=500)
    stamp = sweep.ensure_plan(tmp_path, plan)
    assert sweep.ensure_plan(tmp_path, plan) == stamp
    with pytest.raises(ValueError, match='续跑'):
        sweep.ensure_plan(tmp_path, dict(plan, build=dict(build_id='old-label', artifacts=dict(agent='b'))))
    assert json.loads((tmp_path / 'plan.json').read_text()) == plan

    sample = sweep.sample_path(tmp_path, 'shared', 32, 0)
    sweep.save(sample, dict(mode='shared', H=384, P=32, D=16, rep=0, row={}))
    with pytest.raises(ValueError, match='样本与采集计划不符'):
        sweep.read_sample(sample, stamp, 'shared', 384, 32, 16, 0)
