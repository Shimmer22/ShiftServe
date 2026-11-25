#!/usr/bin/env python3
"""Verify optimized HTP Prefill (OPBATCH=1024/16, merged EMPTY_THINK, cold_route_v1, big-core affinity)
for numerical accuracy, KV state verification, prefill_prior compaction, and Turn-2 dispatch attribution."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post_resilient

BASE = "http://10.29.131.50:18780"
BUILD = ROOT / "results/mainline-eval/t6-opt-htp-prefill"
MODEL = "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf"
MODEL_ID = "qwen3-8b-q4_0"
MODEL_SHA = "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066"
TS_FIXTURES = ROOT / "results/20261124-toolsandbox-10cases/fixtures.json"
DT_FIXTURES = ROOT / "results/20261124-droidtask-10cases/fixtures.json"
ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs.json"
KV_COSTS = ROOT / "results/mainline-eval/t6-8b-table1-revised/t5-kv-priors.json"
OUT = ROOT / "results/mainline-eval/20261125-htp-opt-verify"

BATCH_RE = re.compile(r"ggml-hex: HTP0 profile-batch n-ops (\d+) batch-dur-usec (\d+) htp-ops-usec (\d+)")

COLD_MODEL_HTP = {
    "kind": "cold_route_v1",
    "model_id": MODEL_ID,
    "shared_weights": True,
    "points": [
        {"P": 128, "cc_ms": 9000.0, "hc_ms": 5000.0},
        {"P": 2048, "cc_ms": 180000.0, "hc_ms": 50000.0},
    ],
}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def req(path: str, data: dict, timeout: int = 120) -> dict:
    return _http_post_resilient(BASE + path, data, timeout=timeout, retries=3)


def start_service(
    svc_dir: Path,
    condition: str,
    *,
    shared: bool = False,
    big_cores: int = 0,
    htp_profile: int = 1,
    verify_kv: int = 1,
    kv_reduce: int = 0,
    m1_explorer: int = 0,
    compact_anchor: int = 0,
) -> dict:
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), "start",
        "--condition", condition,
        "--output", str(svc_dir),
        "--logs-only", "--retry-network",
        "--runtime-overlay", str(BUILD),
        "--model", MODEL, "--model-id", MODEL_ID, "--model-sha256", MODEL_SHA,
        "--context-size", "9472", "--cpu-repack", "0", "--cpu-threads", "6",
        "--opbatch", "1024", "--opqueue", "16",
        "--htp-profile", str(htp_profile),
        "--htp-big-cores", str(big_cores),
        "--verify-kv-state", str(verify_kv),
        "--kv-hwm-trim", "0", "--memory-ledger", "--gap-ms", "0",
        "--max-quiet-c", "39.0", "--min-prime-khz", "2000000",
        "--min-mem-avail-kib", "1500000", "--cooldown-timeout-s", "360",
    ]
    if shared:
        cmd.append("--shared-weights")
    if kv_reduce:
        cmd += [
            "--kv-reduce", "1", "--kv-policy", "prefill_prior",
            "--kv-recent-tokens", "32", "--kv-tail-tokens", "16",
            "--compact-anchor", str(compact_anchor),
            "--kv-layer-stride", "4", "--kv-min-evict-tokens", "256",
        ]
    if m1_explorer:
        cmd += ["--m1-explorer", "1"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
        return json.load(r)


def stop_service(svc_dir: Path, condition: str) -> None:
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), "stop",
        "--condition", condition, "--output", str(svc_dir), "--logs-only", "--retry-network",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_batches_by_turn(runtime_log: Path) -> list[dict]:
    turns = []
    cur = None
    for line in runtime_log.read_text(errors="replace").splitlines():
        if '"stage":"before_prefill"' in line:
            cur = {"batches": 0, "ops": 0, "batch_dur_us": 0, "htp_ops_us": 0}
        elif '"stage":"after_prefill"' in line and cur is not None:
            turns.append(cur)
            cur = None
        elif cur is not None:
            m = BATCH_RE.search(line)
            if m:
                cur["batches"] += 1
                cur["ops"] += int(m.group(1))
                cur["batch_dur_us"] += int(m.group(2))
                cur["htp_ops_us"] += int(m.group(3))
    return turns


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts_corpus = json.loads(TS_FIXTURES.read_text())
    ts_case = next(c for c in ts_corpus["cases"] if c["id"] == "find_current_city_low_battery_mode")
    ts_turns = ts_case["turns"][:3]

    dt_corpus = json.loads(DT_FIXTURES.read_text())
    dt_case = next(c for c in dt_corpus["cases"] if c["id"] == "filemanager_e269c90eebf3")

    summary: dict = {"status": "running", "calibration": {}, "arms": {}, "shiftserve_check": {}}

    # 1. Calibration numerical gate (H=606, P=42) + 4 attribution arms
    arms = [
        ("armA_full_htp_big0", "htp", False, 0),
        ("armB_shared_hc_big0", "route", True, 0),
        ("armC_shared_hc_big1", "route", True, 1),
        ("armD_full_htp_big1", "htp", False, 1),
    ]
    for arm_name, cond, shared, big_cores in arms:
        arm_dir = OUT / arm_name
        svc_dir = arm_dir / f"svc-{time.time_ns()}"
        try:
            health = start_service(svc_dir, cond, shared=shared, big_cores=big_cores, htp_profile=1, verify_kv=1)
            assert health.get("htp_big_cores") == bool(big_cores), f"htp_big_cores mismatch: {health}"

            if arm_name == "armB_shared_hc_big0":
                # Numerical gate on warm point H=606, P=42
                warm = ts_turns[1]
                ids = warm["request"]["frozen_prompt_ids"]
                cal = req("/a2o/calibrate", {
                    "prefix_ids": ids[:606], "input_ids": ids[606:648],
                    "decode_evaluations": 1, "repetitions": 1,
                }, timeout=600)
                save(arm_dir / "cal-warm.json", cal)
                row = cal["rows"][0]
                assert row["cosine"] >= 0.99, f"cosine too low: {row['cosine']}"
                assert row["cc_ids"] == row["hc_ids"], "cc_ids != hc_ids"
                summary["calibration"] = {"cosine": row["cosine"], "htp_prefill_ms": row["htp_prefill_ms"], "cpu_prefill_ms": row["cpu_prefill_ms"]}

            if shared:
                req("/a2o/costs", {"cold_model": COLD_MODEL_HTP}, timeout=30)

            req("/a2o/reset", {})
            turn_rows = []
            for ti, turn in enumerate(ts_turns):
                if ti > 0 and shared:
                    req("/a2o/tool-start", {}, timeout=30)
                body = dict(
                    turn["request"],
                    max_tokens=turn["D"],
                    expected_decode_tokens=turn["D"],
                    frozen_graph_profile=True,
                )
                if ti > 0 and shared:
                    body["frozen_force_path"] = "hc_full"
                resp = _agent_chat_resilient(BASE, body, f"opt-verify:{arm_name}:{ti}", total_timeout=600)
                save(arm_dir / f"turn-{ti:02d}.json", resp)
                t = resp["timings"]
                assert resp["__verbose"]["tokens"] == turn["request"]["frozen_decode_ids"], f"{arm_name} T{ti} token mismatch"
                g = t["graph_profile"]["prefill"]
                if ti > 0:
                    assert g["graphs"] == 1, f"{arm_name} T{ti} expected 1 merged graph, got {g['graphs']}"
                turn_rows.append({
                    "turn": ti,
                    "route": t["route"],
                    "reason": t["route_reason"],
                    "prompt_ms": round(t["prompt_ms"], 2),
                    "dispatch_ms": round(g["dispatch_us"] / 1000.0, 2),
                    "handoff_ms": round(t["handoff_ms"], 2),
                    "decode_ms": round(t["predicted_ms"], 2),
                    "graphs": g["graphs"],
                })
        finally:
            stop_service(svc_dir, cond)

        batches = parse_batches_by_turn(svc_dir / "device/runtime.log")
        for tr, bt in zip(turn_rows, batches):
            htp_ops_ms = bt["htp_ops_us"] / 1000.0
            rpc_ms = (bt["batch_dur_us"] - bt["htp_ops_us"]) / 1000.0
            cpu_split_ms = tr["dispatch_ms"] - (bt["batch_dur_us"] / 1000.0)
            tr.update({
                "batches": bt["batches"],
                "ops": bt["ops"],
                "htp_ops_ms": round(htp_ops_ms, 2),
                "rpc_queue_ms": round(rpc_ms, 2),
                "cpu_split_ms": round(cpu_split_ms, 2),
            })
        summary["arms"][arm_name] = turn_rows

    # 2. ShiftServe compaction check on DroidTask filemanager_e269c90eebf3
    ss_dir = OUT / "shiftserve_filemanager"
    svc_dir = ss_dir / f"svc-{time.time_ns()}"
    try:
        start_service(
            svc_dir, "route", shared=True, big_cores=0, htp_profile=0, verify_kv=0,
            kv_reduce=1, m1_explorer=1, compact_anchor=1,
        )
        rc = json.loads(ROUTE_COSTS.read_text())
        kv_m = json.loads(KV_COSTS.read_text())
        for k in ("htp_prior_tax", "evict"):
            kv_m[k].setdefault("slope_ms_per_token", 0.0)
        payload = dict(rc, kv_model=kv_m, cold_model=COLD_MODEL_HTP, m1_explorer={"enabled": True})
        req("/a2o/costs", payload, timeout=30)
        req("/a2o/reset", {})
        ss_turns = []
        for ti, turn in enumerate(dt_case["turns"]):
            if ti > 0:
                req("/a2o/tool-start", {}, timeout=30)
            body = dict(
                turn["request"],
                max_tokens=turn["D"],
                expected_decode_tokens=turn["D"],
                frozen_graph_profile=True,
                frozen_kv_keep_tokens=dt_case["keep_tokens"] if ti else 9472,
            )
            resp = _agent_chat_resilient(BASE, body, f"opt-verify:ss:{ti}", total_timeout=600)
            save(ss_dir / f"turn-{ti:02d}.json", resp)
            t = resp["timings"]
            assert resp["__verbose"]["tokens"] == turn["request"]["frozen_decode_ids"], f"SS T{ti} token mismatch"
            kv = t["kv_reduce"] or {}
            swap_kib = max(s["process"]["VmSwap_kib"] for s in t["memory_ledger"]["samples"])
            assert swap_kib == 0, f"SS T{ti} swap={swap_kib}"
            ss_turns.append({
                "turn": ti, "route": t["route"], "reason": t["route_reason"],
                "prompt_ms": round(t["prompt_ms"], 2),
                "prior_ms": round(t["prior_ms"], 2),
                "callback_graphs": t["graph_profile"]["prefill"]["callback_graphs"],
                "prior_ranked": kv.get("prior_ranked_tokens", 0),
                "evicted": kv.get("evicted_count", 0),
                "swap_kib": swap_kib,
            })
        assert any(r["prior_ranked"] > 0 and r["callback_graphs"] == 1 for r in ss_turns[1:]), f"no prior compaction triggered: {ss_turns}"
        summary["shiftserve_check"] = ss_turns
    finally:
        stop_service(svc_dir, "route")

    summary["status"] = "passed"
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
