#!/usr/bin/env python3
"""Verify Mechanism 2 adaptive kv_reduce_cost_v1 & compact_anchor live on phone (Qwen3-1.7B)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import post

ENDPOINT = "http://10.29.131.50:18780"


def get_health():
    with urllib.request.urlopen(ENDPOINT + "/health", timeout=20) as resp:
        return json.loads(resp.read().decode())


def route_service(action: str, cfg: dict, condition: str, svc_dir: Path, log_path: Path, compact_anchor: int = 0):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run/route_phone.py"),
        action,
        "--condition",
        condition,
        "--output",
        str(svc_dir),
        "--retry-network",
    ]
    if action == "start":
        cmd += [
            "--binary", str((ROOT / cfg["binary"]).resolve()),
            "--registry", str((ROOT / cfg["registry"]).resolve()),
            "--model", cfg["model_path"],
            "--model-id", cfg["model_id"],
            "--model-sha256", cfg["model_sha256"],
            "--context-size", str(cfg["context_size"]),
            "--cpu-threads", str(cfg["cpu_threads"]),
            "--cpu-repack", str(cfg["cpu_repack"]),
            "--opbatch", str(cfg["opbatch"]),
            "--opqueue", str(cfg["opqueue"]),
            "--kv-offload", str(cfg["kv_offload"]),
            "--flash-attention", str(cfg["flash_attention"]),
            "--kv-reduce", "1",
            "--kv-policy", "adaptive_cost",
            "--kv-costs", str((ROOT / cfg["kv_costs"]).resolve()),
            "--kv-sink-tokens", "4",
            "--kv-recent-tokens", "256",
            "--kv-pool-window", "5",
            "--kv-tail-tokens", "64",
            "--compact-anchor", str(compact_anchor),
            "--expected-decode", "96",
        ]
        if condition == "cpu":
            cmd.append("--single-resident")
    else:
        cmd.append("--logs-only")
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_stage1_route_and_full_anchor(cfg: dict, out: Path):
    svc_dir = out / "svc-route-full"
    svc_log = out / "svc-route-full.log"
    route_service("start", cfg, "route", svc_dir, svc_log, compact_anchor=0)
    gap_rows = []
    chat_rows = []
    try:
        health = get_health()
        assert health["kv_cost_loaded"] is True
        assert health["compact_anchor"] is False
        assert health["kv_policy"] == "adaptive_cost"

        # 1. Gap-replay adaptive_cost across deadzone (1024, 2048) and active (4096, 6144)
        unit_text = "System telemetry block: node=phone-1, queue=4, temp=33C, power=nominal.\n"
        unit_toks = post(ENDPOINT, "/tokenize", {"content": unit_text})["tokens"]
        q_toks = post(ENDPOINT, "/tokenize", {"content": "\nAcknowledge."})["tokens"]
        gen_toks = post(ENDPOINT, "/tokenize", {"content": "Acknowledged."})["tokens"][:4]
        for h_target in [1024, 2048, 4096, 6144]:
            reps = max(1, h_target // len(unit_toks))
            prefix = (unit_toks * reps)[:h_target]
            input_ids = (unit_toks * max(1, 256 // len(unit_toks)))[: 256 - len(q_toks)] + q_toks
            body = {
                "prefix_ids": prefix,
                "input_ids": input_ids,
                "generated_ids": gen_toks,
                "order": ["route"],
                "allow_partial_order": True,
                "gap_ms": 2000.0,
                "return_policy": "greedy",
                "prediction": {
                    "validated": True,
                    "cpu_prefill_ms": 15000.0,
                    "cpu_restore_ms": 500.0,
                    "htp_prefill_handoff_ms": 500.0,
                    "prepare_ms": 1000.0,
                },
                "kv_reduce": True,
                "kv_reduce_on_cpu": False,
                "kv_reduce_policy": "adaptive_cost",
                "kv_tail_tokens": 64,
                "kv_sink_tokens": 4,
                "kv_recent_tokens": 256,
                "kv_pool_window": 5,
                "expected_decode_tokens": 96,
            }
            resp = retry_network(lambda b=body: post(ENDPOINT, "/a2o/gap-replay", b, 1800))
            row = resp["results"][0]
            dec = row["kv_cost_decision"]
            if h_target in (1024, 2048):
                assert dec["admit"] is False, f"Expected reject at H={h_target}, got {dec}"
                assert row["kv_reduce"] is None
                assert row["decode_n_kv"] == len(prefix) + len(input_ids)
            else:
                assert dec["admit"] is True, f"Expected admit at H={h_target}, got {dec}"
                assert row["kv_reduce"] is not None
                assert row["decode_n_kv"] == dec["keep_tokens"]
            gap_rows.append({"H_target": h_target, "decision": dec, "decode_n_kv": row["decode_n_kv"], "evict_ms": row["evict_ms"], "handoff_ms": row["handoff_ms"]})

        # 2. Multi-turn /v1/chat/completions with full_anchor=1 (compact_anchor=0) on CPU path
        post(ENDPOINT, "/a2o/reset", {})
        line = "Device diagnostic record #0192: battery=89%, thermal=normal, carrier=LTE-A.\n"
        l_toks = len(post(ENDPOINT, "/tokenize", {"content": line})["tokens"])
        turns = [
            ("Short check: reply OK.", 16),
            ("Log batch A:\n" + line * max(1, 2000 // l_toks) + "Reply OK.", 16),
            ("Log batch B:\n" + line * max(1, 2200 // l_toks) + "Summarize key status in 12 words.", 48),
        ]
        msgs = []
        for idx, (prompt, max_toks) in enumerate(turns):
            msgs.append({"role": "user", "content": prompt})
            resp = retry_network(lambda m=msgs, mt=max_toks: post(ENDPOINT, "/v1/chat/completions", {"messages": m, "temperature": 0.0, "max_tokens": mt, "expected_decode_tokens": 96}, 600))
            t = resp["timings"]
            txt = resp["choices"][0]["message"].get("content") or ""
            msgs.append({"role": "assistant", "content": txt})
            chat_rows.append({
                "turn": idx,
                "H_anchor": t["H_anchor"],
                "H_effective": t["H_effective"],
                "P_evaluated": t["P_evaluated"],
                "kv_cost_decision": t["kv_cost_decision"],
                "kv_reduce": t["kv_reduce"],
                "prompt_ms": t["prompt_ms"],
                "predicted_ms": t["predicted_ms"],
                "predicted_n": t["predicted_n"],
                "text": txt,
            })
        assert chat_rows[0]["kv_cost_decision"]["admit"] is False
        assert chat_rows[1]["H_anchor"] > 0 and chat_rows[1]["kv_cost_decision"]["admit"] is False
        assert chat_rows[2]["H_anchor"] > 2000 and chat_rows[2]["H_effective"] == chat_rows[2]["H_anchor"]
        assert chat_rows[2]["kv_cost_decision"]["admit"] is True
        assert chat_rows[2]["kv_reduce"]["kept_count"] == chat_rows[2]["kv_cost_decision"]["keep_tokens"]
        assert len(chat_rows[2]["text"].strip()) > 0
    finally:
        route_service("stop", cfg, "route", svc_dir, svc_log, compact_anchor=0)
    return {"gap_replay": gap_rows, "chat_full_anchor": chat_rows}


def run_stage2_compact_anchor(cfg: dict, out: Path):
    svc_dir = out / "svc-cpu-compact"
    svc_log = out / "svc-cpu-compact.log"
    route_service("start", cfg, "cpu", svc_dir, svc_log, compact_anchor=1)
    rows = []
    try:
        health = get_health()
        assert health["condition"] == "cpu"
        assert health["kv_cost_loaded"] is True
        assert health["compact_anchor"] is True

        post(ENDPOINT, "/a2o/reset", {})
        line = "Customer ticket record #841: status=pending, shipping=express, priority=standard.\n"
        l_toks = len(post(ENDPOINT, "/tokenize", {"content": line})["tokens"])
        # Turn 0 (~1000 tokens -> reject)
        # Turn 1 (+1500 tokens -> ~2500 -> reject)
        # Turn 2 (+1800 tokens -> ~4300 -> ADMIT & compact_anchor to ~1536!)
        # Turn 3 (+300 tokens -> H_eff ~1531 + 300 = ~1831 -> REJECT cleanly at 0 evict cost!)
        # Turn 4 (+2500 tokens -> H_eff ~1831 + 2500 = ~4331 -> ADMIT & RE-EVICT via live_positions!)
        sizes = [1000, 1500, 1800, 300, 2500]
        msgs = []
        for idx, tok_sz in enumerate(sizes):
            reps = max(1, tok_sz // l_toks)
            prompt = f"Batch {idx}:\n" + (line * reps) + f"Confirm batch {idx} in one short sentence."
            msgs.append({"role": "user", "content": prompt})
            resp = retry_network(lambda m=msgs: post(ENDPOINT, "/v1/chat/completions", {"messages": m, "temperature": 0.0, "max_tokens": 32, "expected_decode_tokens": 96}, 600))
            t = resp["timings"]
            txt = resp["choices"][0]["message"].get("content") or ""
            msgs.append({"role": "assistant", "content": txt})
            rows.append({
                "turn": idx,
                "H_anchor": t["H_anchor"],
                "H_effective": t["H_effective"],
                "P_evaluated": t["P_evaluated"],
                "restore_ms": t["restore_ms"],
                "kv_cost_decision": t["kv_cost_decision"],
                "kv_reduce": t["kv_reduce"],
                "prompt_ms": t["prompt_ms"],
                "predicted_ms": t["predicted_ms"],
                "predicted_n": t["predicted_n"],
                "text": txt,
            })
        assert rows[0]["kv_cost_decision"]["admit"] is False
        assert rows[1]["kv_cost_decision"]["admit"] is False
        assert rows[2]["kv_cost_decision"]["admit"] is True
        assert rows[2]["kv_reduce"].get("compact_anchor") is True
        assert rows[3]["H_effective"] < rows[3]["H_anchor"], f"Expected compacted H_effective < H_anchor, got {rows[3]}"
        assert rows[3]["kv_cost_decision"]["admit"] is False, f"Expected Turn 3 to skip re-evict due to compact anchor, got {rows[3]}"
        assert rows[4]["kv_cost_decision"]["admit"] is True, f"Expected Turn 4 to re-admit after accumulating >4k active KV, got {rows[4]}"
        assert rows[4]["kv_reduce"]["before_pos_count"] == rows[4]["H_effective"] + rows[4]["P_evaluated"]
        assert rows[4]["kv_reduce"].get("compact_anchor") is True
        assert len(rows[4]["text"].strip()) > 0
    finally:
        route_service("stop", cfg, "cpu", svc_dir, svc_log, compact_anchor=1)
    return rows


def main():
    out = ROOT / "results/20261115-qwen3-1_7b-adaptive-kv/step2-verify"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "binary": "results/20261115-qwen3-1_7b-adaptive-kv/build-service/agent",
        "registry": "results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so",
        "model_path": "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-1.7B-Q4_0.gguf",
        "model_id": "qwen3-1.7b-q4_0",
        "model_sha256": "5c71ea792028698598657d8f3e13fcf1980fcb2e1887c4931c5a129a5c8b488f",
        "kv_costs": "configs/qwen3-1_7b-kv-reduce-cost.json",
        "context_size": 12288,
        "cpu_threads": 6,
        "cpu_repack": 1,
        "opbatch": 1024,
        "opqueue": 16,
        "kv_offload": 0,
        "flash_attention": 0,
    }
    stage1 = run_stage1_route_and_full_anchor(cfg, out)
    stage2 = run_stage2_compact_anchor(cfg, out)
    report = {"stage1_route_and_full_anchor": stage1, "stage2_cpu_compact_anchor": stage2}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
