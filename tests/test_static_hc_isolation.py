import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run.frozen_ablation as fa
import scripts.run.t6_mobile_gui_table1 as mg
from scripts.run.t6_table1_summary import TABLE1_EIGHT_ARMS, write_table1


def test_table1_eight_arms_summary(tmp_path: Path):
    cases = ["c1"]
    rows = []
    memory = {}
    for idx, arm in enumerate(TABLE1_EIGHT_ARMS):
        base = 2000.0 - idx * 100.0
        rows.append({
            "id": "c1",
            "arm": arm,
            "warm_rounds": 2,
            "post_return_ms": base,
            "post_return_ledger_ms": 100.0,
            "warm_decode_tokens": 20,
            "post_return_phases_ms": {
                "prep_wait": 10.0,
                "observation": 20.0,
                "restore": 50.0,
                "prefill_including_prior": base - 300.0,
                "prior": 0.0,
                "compact": 0.0,
                "handoff": 40.0,
                "decode": 180.0,
                "snapshot": 0.0,
                "other": 0.0,
            },
        })
        memory[arm] = {
            "peak_pss_plus_dma_mib": 7000.0,
            "peak_vmswap_mib": 0.0,
            "missing_pss_plus_dma_samples": 0,
        }
    write_table1(tmp_path, cases, rows, memory, arms=TABLE1_EIGHT_ARMS)
    csv_text = (tmp_path / "table1.csv").read_text()
    assert "Vanilla-Static-HC" in csv_text
    assert "Shared-Static-HC" in csv_text
    assert "speedup_vs_vanilla_static_hc" in csv_text


def test_table1_three_arm_subset_without_full_cpu(tmp_path: Path):
    subset = {
        "dual_static_hc": "Vanilla-Static-HC",
        "shared_static_hc": "Shared-Static-HC",
        "shiftserve_joint": "ShiftServe",
    }
    cases = ["c1"]
    rows = [
        {
            "id": "c1", "arm": arm, "warm_rounds": 2, "post_return_ms": 1000.0 - idx * 100.0,
            "post_return_ledger_ms": 50.0, "warm_decode_tokens": 10,
            "post_return_phases_ms": {
                "prep_wait": 0.0, "observation": 10.0, "restore": 20.0,
                "prefill_including_prior": 500.0, "prior": 0.0, "compact": 0.0,
                "handoff": 30.0, "decode": 100.0, "snapshot": 0.0, "other": 0.0,
            },
        }
        for idx, arm in enumerate(subset)
    ]
    memory = {
        arm: {"peak_pss_plus_dma_mib": 7000.0, "peak_vmswap_mib": 0.0, "missing_pss_plus_dma_samples": 0}
        for arm in subset
    }
    write_table1(tmp_path, cases, rows, memory, arms=subset)
    assert "ShiftServe" in (tmp_path / "table1.csv").read_text()


def test_mobile_campaign_continue_on_cell_error_cleans_up_and_resumes(tmp_path: Path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    for name in ("agent", "libggml-cpu.so", "libllama.so"):
        (build / name).write_bytes(name.encode())
    (build / "manifest.json").write_text(json.dumps({
        "git_commit": "abc123",
        "build_id": "test-build",
        "artifacts": {name: fa.digest(build / name) for name in ("agent", "libggml-cpu.so", "libllama.so")},
    }))
    cpu_reg = tmp_path / "libggml.so"
    cpu_reg.write_bytes(b"reg")
    fixtures = tmp_path / "fixtures.json"
    turn0 = {
        "index": 0, "H": 0, "P": 4, "D": 2, "observed_gap_ms": 0,
        "request": {"messages": [{"role": "user", "content": "hi"}], "frozen_prompt_ids": [1, 2, 3, 4], "frozen_decode_ids": [5, 6]},
    }
    turn1 = {
        "index": 1, "H": 4, "P": 3, "D": 2, "observed_gap_ms": 0,
        "request": {"messages": [{"role": "user", "content": "hi"}, {"role": "tool", "content": "ok"}], "frozen_prompt_ids": [1, 2, 3, 4, 7, 8, 9], "frozen_decode_ids": [10, 11]},
    }
    fixtures.write_text(json.dumps({
        "cases": [{"id": "c1", "benchmark": "androidcontrol_low", "source_model": "qwen3-8b", "keep_tokens": 64, "turns": [turn0, turn1]}],
        "calibration_cases": [],
    }))

    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(
        build=build, output=out, fixtures=fixtures, model=ROOT / "configs/model-default.json",
        cpu_registry=cpu_reg, kv_costs=None, context_size=256, gap_ms=0, verify_kv_state=0, kv_hwm_trim=0,
        device="xiaomi14pro", memory_ledger=True, compact_anchor=1, kv_layer_stride=4, kv_min_evict_tokens=256,
        conditions=["dual_static_hc", "shiftserve_joint"], case=None, force_path="none", single_turn=False,
        m1_explorer=True, graph_profile=True, cpu_threads=6, continue_on_cell_error=True,
    )
    campaign = mg.MobileCampaign(args, fa.read(fixtures))
    cleanups = []
    monkeypatch.setattr(campaign, "cleanup_failed_phone_cell", lambda attempt_name: cleanups.append(attempt_name))
    monkeypatch.setattr(campaign, "service", lambda action, arm, directory: (_ for _ in ()).throw(RuntimeError("oom killed")) if action == "start" and arm == "dual_static_hc" else None)

    target = out / "dual_static_hc" / "c1"
    res = campaign.run_cell(campaign.cases[0], "dual_static_hc", target, "formal")
    assert res is None
    assert (target / "FAILED.json").is_file()
    assert json.loads((target / "FAILED.json").read_text())["arm"] == "dual_static_hc"
    assert len(cleanups) == 1
    assert json.loads((out / "failed-cells.json").read_text())[0]["case"] == "c1"

    # Resume skips already failed cell without calling cleanup again.
    res2 = campaign.run_cell(campaign.cases[0], "dual_static_hc", target, "formal")
    assert res2 is None
    assert len(cleanups) == 1

    # write_details succeeds cleanly with 0 completed cases on dual_static_hc.
    mg.write_details(out, [{"case": "c1", "benchmark": "androidcontrol_low", "order": ["dual_static_hc", "shiftserve_joint"]}], arms=["dual_static_hc", "shiftserve_joint"])
    assert (out / "table1.csv").is_file()


def test_balanced_order_eight_arms():
    cases = [
        {"id": "t1", "benchmark": "toolsandbox"},
        {"id": "d1", "benchmark": "droidtask"},
        {"id": "a1", "benchmark": "androidcontrol_low"},
    ]
    arms = list(TABLE1_EIGHT_ARMS)
    order = mg.balanced_order(cases, arms=arms)
    assert len(order) == 3
    for row in order:
        assert sorted(row["order"]) == sorted(arms)


def test_static_hc_strips_async_prep_cold_route_and_deferred_anchor(tmp_path: Path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    for name in ("agent", "libggml-cpu.so", "libllama.so"):
        (build / name).write_bytes(name.encode())
    (build / "manifest.json").write_text(json.dumps({
        "git_commit": "abc123",
        "build_id": "test-build",
        "artifacts": {name: fa.digest(build / name) for name in ("agent", "libggml-cpu.so", "libllama.so")},
    }))
    cpu_reg = tmp_path / "libggml.so"
    cpu_reg.write_bytes(b"reg")
    fixtures = tmp_path / "fixtures.json"
    turn0 = {
        "index": 0,
        "H": 0,
        "P": 4,
        "D": 2,
        "observed_gap_ms": 0,
        "request": {
            "messages": [{"role": "user", "content": "hi"}],
            "frozen_prompt_ids": [1, 2, 3, 4],
            "frozen_decode_ids": [5, 6],
        },
    }
    turn1 = {
        "index": 1,
        "H": 4,
        "P": 3,
        "D": 2,
        "observed_gap_ms": 0,
        "request": {
            "messages": [{"role": "user", "content": "hi"}, {"role": "tool", "content": "ok"}],
            "frozen_prompt_ids": [1, 2, 3, 4, 7, 8, 9],
            "frozen_decode_ids": [10, 11],
        },
    }
    fixtures.write_text(json.dumps({"cases": [{"id": "c1", "benchmark": "toolsandbox", "source_model": "qwen3-8b", "keep_tokens": 64, "turns": [turn0, turn1]}], "calibration_cases": []}))

    args = SimpleNamespace(
        build=build, output=tmp_path / "out", fixtures=fixtures, model=ROOT / "configs/model-default.json",
        cpu_registry=cpu_reg, kv_costs=None, context_size=256, gap_ms=0, verify_kv_state=0, kv_hwm_trim=0,
        device="xiaomi14pro", memory_ledger=False, compact_anchor=1, kv_layer_stride=4, kv_min_evict_tokens=256,
        conditions=["dual_static_hc", "shared_static_hc", "shiftserve_joint"], case=None,
        force_path="none", single_turn=False, m1_explorer=True, graph_profile=True, cpu_threads=6,
    )
    args.output.mkdir()
    campaign = fa.Campaign(args, fa.read(fixtures))
    calls = []

    def fake_request(path, data, timeout=30, retries=4):
        calls.append((path, data))
        return {"started": True}

    def fake_chat(base, body, rid, total_timeout=2400):
        calls.append(("chat", body))
        idx = 0 if len(body["frozen_prompt_ids"]) == 4 else 1
        t = turn0 if idx == 0 else turn1
        return {
            "__verbose": {"tokens": t["request"]["frozen_decode_ids"], "stop_type": "forced"},
            "timings": {
                "H_anchor": t["H"], "H_effective": t["H"], "P_evaluated": t["P"], "predicted_n": t["D"],
                "route": "htp", "decode_backend": "cpu", "handoff_bytes": 1024, "kv_reduce": None,
                "deferred_hc_anchor": False if "dual_static_hc" in rid or "shared_static_hc" in rid else True,
                "availability": "absent" if "dual_static_hc" in rid or "shared_static_hc" in rid else "ready",
                "predicted_ms": 100.0, "total_ms": 500.0,
                "decode_resources": {"before": {"VmRSS_kib": 1000, "VmSwap_kib": 0}, "after": {"VmRSS_kib": 1000, "VmSwap_kib": 0}},
            },
        }

    monkeypatch.setattr(fa, "request", fake_request)
    monkeypatch.setattr(fa, "_agent_chat_resilient", fake_chat)
    monkeypatch.setattr(fa, "require_attached_phone_preflight", lambda *a, **k: None)

    for arm in ("dual_static_hc", "shared_static_hc", "shiftserve_joint"):
        calls.clear()
        p = tmp_path / arm
        p.mkdir()
        campaign.replay(arm, campaign.cases[0], p)
        tool_starts = [c for c in calls if c[0] == "/a2o/tool-start"]
        chats = [c[1] for c in calls if c[0] == "chat"]
        if arm in fa.STATIC_HC_ARMS:
            assert not tool_starts
            for chat in chats:
                assert chat["frozen_force_path"] == "hc_full"
                assert chat["frozen_sync_hc"] is True
                assert chat["frozen_legacy_snapshot"] is True
        else:
            assert len(tool_starts) == 1
            for chat in chats:
                assert "frozen_sync_hc" not in chat
                assert "frozen_legacy_snapshot" not in chat
