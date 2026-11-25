#!/usr/bin/env python3
"""Controlled ABBA probe on Xiaomi 14 Pro to isolate OPBATCH/OPQUEUE impact on Full-HTP Decode vs Prefill."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post_resilient

BASE = "http://10.29.131.50:18780"
BUILD = ROOT / "results/mainline-eval/t6-opt-htp-prefill"
MODEL = "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf"
MODEL_ID = "qwen3-8b-q4_0"
MODEL_SHA = "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066"
DT_FIXTURES = ROOT / "results/20261124-droidtask-10cases/fixtures.json"
COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-opt.json"
OUT = ROOT / "results/mainline-eval/20261125-htp-opbatch-decode-ab"


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(DT_FIXTURES.read_text())
    case = next(c for c in corpus["cases"] if c["id"] == "calendar_e8b601c47a04")
    turns = case["turns"][:2]

    arms = [
        ("arm1_htp_ob1_oq1", "htp", False, 1, 1),
        ("arm2_htp_ob1024_oq16", "htp", False, 1024, 16),
        ("arm3_htp_ob1024_oq1", "htp", False, 1024, 1),
        ("arm4_htp_ob1_oq1_repeat", "htp", False, 1, 1),
        ("arm5_hc_ob1_oq1", "route", True, 1, 1),
    ]
    summary: dict = {"status": "running", "arms": {}}
    for name, cond, shared, ob, oq in arms:
        arm_dir = OUT / name
        svc_dir = arm_dir / f"svc-{time.time_ns()}"
        start_cmd = [
            sys.executable, str(ROOT / "scripts/run/route_phone.py"), "start",
            "--condition", cond,
            "--output", str(svc_dir), "--logs-only", "--retry-network",
            "--runtime-overlay", str(BUILD),
            "--model", MODEL, "--model-id", MODEL_ID, "--model-sha256", MODEL_SHA,
            "--context-size", "9472", "--cpu-repack", "0", "--cpu-threads", "6",
            "--opbatch", str(ob), "--opqueue", str(oq),
            "--htp-profile", "0", "--htp-big-cores", "1",
            "--kv-hwm-trim", "0", "--memory-ledger", "--gap-ms", "0",
            "--max-quiet-c", "39.0", "--min-prime-khz", "2000000",
            "--min-mem-avail-kib", "1500000", "--cooldown-timeout-s", "360",
        ]
        if shared:
            start_cmd.append("--shared-weights")
        subprocess.run(start_cmd, cwd=ROOT, check=True)
        rows = []
        try:
            if shared:
                _http_post_resilient(BASE + "/a2o/costs", {"cold_model": json.loads(COLD_COSTS.read_text())}, timeout=30, retries=3)
            _http_post_resilient(BASE + "/a2o/reset", {}, timeout=30, retries=3)
            for ti, turn in enumerate(turns):
                if ti > 0 and shared:
                    _http_post_resilient(BASE + "/a2o/tool-start", {}, timeout=30, retries=3)
                body = dict(
                    turn["request"],
                    max_tokens=turn["D"],
                    expected_decode_tokens=turn["D"],
                    frozen_graph_profile=True,
                )
                if ti > 0 and shared:
                    body["frozen_force_path"] = "hc_full"
                resp = _agent_chat_resilient(BASE, body, f"ob-ab:{name}:{ti}", total_timeout=600)
                save(arm_dir / f"turn-{ti:02d}.json", resp)
                assert resp["__verbose"]["tokens"] == turn["request"]["frozen_decode_ids"], f"{name} T{ti} token mismatch"
                t = resp["timings"]
                cl = t["graph_profile"]["clocks"]["before_decode"]
                therm = cl["thermal_sysfs_raw"]
                quiet = (therm.get("quiet-therm") or therm.get("thermal_zone64") or {}).get("value")
                nsp = (therm.get("nsphvx-0") or {}).get("value")
                rows.append({
                    "turn": ti,
                    "route": t["route"],
                    "prompt_ms": round(t["prompt_ms"], 2),
                    "decode_ms": round(t["predicted_ms"], 2),
                    "decode_ms_per_tok": round(t["predicted_ms"] / t["predicted_n"], 2),
                    "quiet_therm": quiet,
                    "nsphvx_0": nsp,
                    "policy2_khz": cl["cpu_policy_khz"].get("policy2"),
                    "policy7_khz": cl["cpu_policy_khz"].get("policy7"),
                })
        finally:
            subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts/run/route_phone.py"), "stop",
                    "--condition", cond, "--output", str(svc_dir), "--logs-only", "--retry-network",
                ],
                cwd=ROOT, check=True,
            )
        summary["arms"][name] = rows

    summary["status"] = "passed"
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
