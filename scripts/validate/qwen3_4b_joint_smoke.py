#!/usr/bin/env python3
"""Step-3 smoke check: verify Joint (Route x KV Reduce) decisions on Qwen3-4B-Q4_0 against Python planner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.network_retry import retry_network
from a2o.scheduling.kv_reduce_cost import plan_joint_route_kv
from a2o.workloads.toolsandbox import post

ENDPOINT = "http://10.29.131.50:18780"


def get_health():
    with urllib.request.urlopen(ENDPOINT + "/health", timeout=20) as resp:
        return json.loads(resp.read().decode())


def route_phone(action: str, svc_dir: Path, log_path: Path, binary: Path):
    mcfg = json.loads((ROOT / "configs/model-qwen3-4b.json").read_text())
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), action,
        "--condition", "route", "--output", str(svc_dir), "--retry-network",
    ]
    if action == "start":
        cmd += [
            "--binary", str(binary.resolve()),
            "--registry", str((ROOT / "results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so").resolve()),
            "--model", mcfg["model_path"],
            "--model-id", mcfg["model_id"],
            "--model-sha256", mcfg["model_sha256"],
            "--context-size", "8192",
            "--cpu-threads", "6",
            "--cpu-repack", "0",
            "--opbatch", "1024",
            "--opqueue", "16",
            "--kv-offload", "0",
            "--flash-attention", "0",
            "--kv-reduce", "1",
            "--kv-policy", "adaptive_cost",
            "--kv-costs", str((ROOT / "configs/qwen3-4b-kv-reduce-cost.json").resolve()),
            "--route-costs", str((ROOT / "configs/qwen3-4b-route-costs.json").resolve()),
            "--kv-sink-tokens", "4",
            "--kv-recent-tokens", "256",
            "--kv-pool-window", "5",
            "--kv-tail-tokens", "64",
            "--compact-anchor", "1",
            "--expected-decode", "96",
        ]
    else:
        cmd.append("--logs-only")
    with log_path.open("a") as lf:
        subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, check=True)


def wait_ready(max_wait_s: float = 12.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait_s:
        res = post(ENDPOINT, "/a2o/tool-start", {}, 60)
        if res.get("started") or res.get("reason") == "previous_preparation_busy":
            time.sleep(0.5)
        else:
            break
    time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=ROOT / "results/20261115-qwen3-4b-joint/build-service/agent")
    ap.add_argument("--output", type=Path, default=ROOT / "results/20261115-qwen3-4b-joint/step3-smoke")
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    svc_dir = out / "service"
    log_path = out / "service.log"
    route_model = json.loads((ROOT / "configs/qwen3-4b-route-costs.json").read_text())["model"]
    kv_model = json.loads((ROOT / "configs/qwen3-4b-kv-reduce-cost.json").read_text())

    turns_out = []
    try:
        route_phone("start", svc_dir, log_path, args.binary)
        health = get_health()
        assert health["kv_cost_loaded"] is True
        assert health["compact_anchor"] is True
        assert health["cpu_repack"] is False
        assert health["single_resident"] is False
        print("[health]", health, flush=True)

        post(ENDPOINT, "/a2o/reset", {}, 60)
        chunk = "Telemetry entry: node=alpha, temp=34C, volt=4120mV, ticket=#841, status=OK, carrier=LTE-A.\n"
        messages = [{"role": "system", "content": "Reply concisely in one sentence."}]
        prompts = [
            f"Log A:\n{chunk * 70}\nAcknowledge receipt.",
            "OK?",
            f"Log B:\n{chunk * 20}\nConfirm ticket number.",
            f"Log C:\n{chunk * 15}\nRepeat ticket number.",
        ]
        for idx, up in enumerate(prompts):
            messages.append({"role": "user", "content": up})
            resp = retry_network(lambda: post(ENDPOINT, "/v1/chat/completions", {
                "model": "qwen3-4b-q4_0",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 24,
                "expected_decode_tokens": 96,
            }, 600))
            txt = resp["choices"][0]["message"].get("content", "")
            messages.append({"role": "assistant", "content": txt})
            tm = resp["timings"]
            pred = tm.get("prediction")
            kvd = tm.get("kv_cost_decision")
            parity = None
            if idx > 0:
                assert tm["availability"] == "ready", f"Expected ready at turn {idx}, got {tm['availability']}"
                py_plan = plan_joint_route_kv(
                    route_model, kv_model,
                    h_eff=tm["H_effective"], p=tm["P_evaluated"], expected_d=96,
                    htp_allowed=True, kv_enabled=True, default_route="cpu",
                    h_logical=tm["H_anchor"], pinned_tokens=tm.get("pinned_tokens", 0),
                )
                assert tm["route"] == py_plan["route"]
                assert tm["route_reason"] == py_plan["reason"]
                for fld in ("cpu_no_kv_total_ms", "htp_no_kv_total_ms", "cpu_best_total_ms", "htp_best_total_ms", "selected_total_ms"):
                    assert abs(pred[fld] - py_plan[fld]) < 1e-4, (fld, pred[fld], py_plan[fld])
                assert kvd["admit"] == py_plan["kv_plan"]["admit"]
                assert kvd["keep_tokens"] == py_plan["kv_plan"]["keep_tokens"]
                parity = "exact_match"
            rec = {
                "turn": idx,
                "H_anchor": tm["H_anchor"],
                "H_effective": tm["H_effective"],
                "P": tm["P_evaluated"],
                "availability": tm["availability"],
                "route": tm["route"],
                "route_reason": tm["route_reason"],
                "kv_admit": kvd["admit"] if kvd else None,
                "kv_keep": kvd["keep_tokens"] if kvd else None,
                "compact_after": (tm.get("kv_reduce") or {}).get("anchor_live_tokens"),
                "prompt_ms": tm["prompt_ms"],
                "handoff_ms": tm["handoff_ms"],
                "restore_ms": tm["restore_ms"],
                "parity": parity,
                "text": txt[:100],
            }
            turns_out.append(rec)
            print("[smoke]", rec, flush=True)
            if idx < len(prompts) - 1:
                post(ENDPOINT, "/a2o/tool-start", {}, 60)
                time.sleep(2.5)

        assert turns_out[1]["route"] == "cpu" and turns_out[1]["route_reason"] == "below_routing_p_min"
        assert turns_out[2]["route"] == "htp" and turns_out[2]["route_reason"] == "joint_route_kv_cost" and turns_out[2]["kv_admit"] is True
        assert turns_out[3]["route"] == "htp"
        summary = {"passed": True, "health": health, "turns": turns_out}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("STEP3_SMOKE_PASSED", out / "summary.json", flush=True)
    finally:
        route_phone("stop", svc_dir, log_path, args.binary)


if __name__ == "__main__":
    main()
