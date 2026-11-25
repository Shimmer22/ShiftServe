#!/usr/bin/env python3
"""Option A matched-temperature, CPU-probe-guarded 3-arm paired confirmation on calendar_e8b601c47a04:
- full_htp: opbatch=1, opqueue=1
- shared_static_hc: opbatch=1024, opqueue=16
- shiftserve_joint: opbatch=1024, opqueue=16 + ob1024 6T route & cold cost tables
All with htp_big_cores=1 and merged EMPTY_THINK."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.device_transport import DeviceTransport
from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post_resilient

BASE = "http://10.29.131.50:18780"
BUILD = ROOT / "results/mainline-eval/t6-opt-htp-prefill"
MODEL = "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf"
MODEL_ID = "qwen3-8b-q4_0"
MODEL_SHA = "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066"
DT_FIXTURES = ROOT / "results/20261124-droidtask-10cases/fixtures.json"
ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"
COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
KV_COSTS = ROOT / "results/mainline-eval/t6-8b-table1-revised/t5-kv-priors.json"
OUT = ROOT / "results/mainline-eval/20261125-dt-option-a-confirm"


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def run_cpu_health_probe(svc_dir: Path) -> dict:
    """Measure /proc/<pid>/smaps read latency (3x), cgroup, cpus_allowed, and scaling_max_freq."""
    t = DeviceTransport(ROOT, "ssh", config=ROOT / "configs/termux-ssh.json")
    remote = t.remote_root + "/" + svc_dir.name
    sh = (
        f'p=$(cat "{remote}/pid"); '
        'for i in 1 2 3; do '
        't0=$(date +%s%N); cat /proc/"$p"/smaps >/dev/null; t1=$(date +%s%N); '
        'echo "SMAPS_NS=$((t1-t0))"; '
        'done; '
        'echo "CGROUP=$(cat /proc/"$p"/cgroup | tr "\\n" ";")"; '
        'echo "CPUS_ALLOWED=$(grep Cpus_allowed_list /proc/"$p"/status | awk "{print \\$2}")"; '
        'for pol in 0 2 5 7; do '
        'echo "POL${pol}=$(cat /sys/devices/system/cpu/cpufreq/policy${pol}/scaling_max_freq 2>/dev/null || echo na)"; '
        'done'
    )
    out = t.call("shell", sh).stdout.decode()
    smaps_ms = []
    info: dict = {}
    for line in out.splitlines():
        if line.startswith("SMAPS_NS="):
            smaps_ms.append(round(int(line.split("=", 1)[1]) / 1e6, 2))
        elif "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    startup = json.loads((svc_dir / "startup-memory.json").read_text())
    after_load_col = next(
        (s["collection_ms"] for s in startup["samples"] if s["stage"] == "after_model_load"),
        None,
    )
    return {
        "smaps_read_ms": smaps_ms,
        "smaps_read_med_ms": round(statistics.median(smaps_ms), 2) if smaps_ms else None,
        "after_model_load_collection_ms": round(after_load_col, 2) if after_load_col is not None else None,
        "cgroup": info.get("CGROUP", ""),
        "cpus_allowed": info.get("CPUS_ALLOWED", ""),
        "policy_scaling_max_khz": {f"policy{p}": info.get(f"POL{p}") for p in (0, 2, 5, 7)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(DT_FIXTURES.read_text())
    case = next(c for c in corpus["cases"] if c["id"] == "calendar_e8b601c47a04")
    turns = case["turns"]

    arms = [
        ("full_htp", "htp", False, False, 1, 1),
        ("shared_static_hc", "route", True, False, 1024, 16),
        ("shiftserve_joint", "route", True, True, 1024, 16),
    ]
    summary: dict = {"status": "running", "case": case["id"], "arms": {}}
    for arm_name, cond, shared, is_shiftserve, ob, oq in arms:
        arm_dir = OUT / arm_name
        svc_dir = arm_dir / f"svc-{time.time_ns()}"
        cmd = [
            sys.executable, str(ROOT / "scripts/run/route_phone.py"), "start",
            "--condition", cond,
            "--output", str(svc_dir), "--logs-only", "--retry-network",
            "--runtime-overlay", str(BUILD),
            "--model", MODEL, "--model-id", MODEL_ID, "--model-sha256", MODEL_SHA,
            "--context-size", "9472", "--cpu-repack", "0", "--cpu-threads", "6",
            "--opbatch", str(ob), "--opqueue", str(oq), "--htp-big-cores", "1",
            "--kv-hwm-trim", "0", "--memory-ledger", "--gap-ms", "0",
            "--max-quiet-c", "36.0", "--min-prime-khz", "2000000",
            "--min-mem-avail-kib", "1500000", "--cooldown-timeout-s", "360",
        ]
        if shared:
            cmd.append("--shared-weights")
        if is_shiftserve:
            cmd += [
                "--kv-reduce", "1", "--kv-policy", "prefill_prior",
                "--kv-recent-tokens", "32", "--kv-tail-tokens", "16",
                "--compact-anchor", "1", "--kv-layer-stride", "4",
                "--kv-min-evict-tokens", "256", "--m1-explorer", "1",
            ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        cpu_probe = run_cpu_health_probe(svc_dir)
        print(f"[{arm_name}] CPU probe: {cpu_probe}", flush=True)

        turn_rows = []
        peak_pss_dma_mib = 0.0
        peak_swap_kib = 0
        try:
            if shared:
                payload: dict = {"cold_model": json.loads(COLD_COSTS.read_text())}
                if is_shiftserve:
                    rc = json.loads(ROUTE_COSTS.read_text())
                    kv_m = json.loads(KV_COSTS.read_text())
                    for k in ("htp_prior_tax", "evict"):
                        kv_m[k].setdefault("slope_ms_per_token", 0.0)
                    payload.update(rc)
                    payload["kv_model"] = kv_m
                    payload["m1_explorer"] = {"enabled": True}
                _http_post_resilient(BASE + "/a2o/costs", payload, timeout=30, retries=3)

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
                if is_shiftserve:
                    body["frozen_kv_keep_tokens"] = case["keep_tokens"] if ti else 9472
                elif ti > 0 and shared:
                    body["frozen_force_path"] = "hc_full"
                resp = _agent_chat_resilient(BASE, body, f"opt-a:{arm_name}:{ti}", total_timeout=600)
                save(arm_dir / f"turn-{ti:02d}.json", resp)
                assert resp["__verbose"]["tokens"] == turn["request"]["frozen_decode_ids"], f"{arm_name} T{ti} token mismatch"

                t = resp["timings"]
                if shared and ti == 0:
                    assert t["route"] == "htp" and t["route_reason"] == "cold_route_cost", f"{arm_name} T0 route={t['route']} reason={t['route_reason']}"

                ledger = t["memory_ledger"]
                col_ms = float(ledger["collection_ms"])
                for s in ledger["samples"]:
                    peak_swap_kib = max(peak_swap_kib, int(s["process"]["VmSwap_kib"]))
                    if s.get("pss_plus_identified_dma_kib") is not None:
                        peak_pss_dma_mib = max(peak_pss_dma_mib, float(s["pss_plus_identified_dma_kib"]) / 1024.0)
                assert peak_swap_kib == 0, f"{arm_name} T{ti} swap={peak_swap_kib}"

                kv = t.get("kv_reduce") or {}
                gp_pre = t["graph_profile"]["prefill"]
                cl = t["graph_profile"]["clocks"]["before_prefill"]
                therm = cl["thermal_sysfs_raw"]
                quiet = (therm.get("quiet-therm") or therm.get("thermal_zone64") or {}).get("value")
                turn_rows.append({
                    "turn": ti,
                    "H": t["H_effective"],
                    "P": t["P_evaluated"],
                    "D": t["predicted_n"],
                    "route": t["route"],
                    "reason": t["route_reason"],
                    "quiet_therm": quiet,
                    "prompt_ms": round(t["prompt_ms"], 2),
                    "handoff_ms": round(t["handoff_ms"], 2),
                    "decode_ms": round(t["predicted_ms"], 2),
                    "decode_ms_per_tok": round(t["predicted_ms"] / t["predicted_n"], 2),
                    "total_ms": round(t["total_ms"], 2),
                    "ledger_collection_ms": round(col_ms, 2),
                    "total_excl_ledger_ms": round(t["total_ms"] - col_ms, 2),
                    "callback_graphs": gp_pre["callback_graphs"],
                    "prior_ranked": kv.get("prior_ranked_tokens", 0),
                    "evicted_count": kv.get("evicted_count", 0),
                })
        finally:
            subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts/run/route_phone.py"), "stop",
                    "--condition", cond, "--output", str(svc_dir), "--logs-only", "--retry-network",
                ],
                cwd=ROOT, check=True,
            )

        if is_shiftserve:
            assert any(r["prior_ranked"] > 0 and r["callback_graphs"] == 1 for r in turn_rows[1:]), f"ShiftServe did not compact: {turn_rows}"

        warm = turn_rows[1:]
        summary["arms"][arm_name] = {
            "opbatch": ob,
            "opqueue": oq,
            "cpu_health_probe": cpu_probe,
            "post_return_s": round(sum(r["total_ms"] for r in warm) / 1000.0, 2),
            "post_return_excl_ledger_s": round(sum(r["total_excl_ledger_ms"] for r in warm) / 1000.0, 2),
            "inference_s": round(sum(r["total_ms"] for r in turn_rows) / 1000.0, 2),
            "inference_excl_ledger_s": round(sum(r["total_excl_ledger_ms"] for r in turn_rows) / 1000.0, 2),
            "warm_prefill_s": round(sum(r["prompt_ms"] for r in warm) / 1000.0, 2),
            "warm_handoff_s": round(sum(r["handoff_ms"] for r in warm) / 1000.0, 2),
            "warm_decode_s": round(sum(r["decode_ms"] for r in warm) / 1000.0, 2),
            "warm_ledger_s": round(sum(r["ledger_collection_ms"] for r in warm) / 1000.0, 2),
            "peak_pss_plus_dma_mib": round(peak_pss_dma_mib, 1),
            "peak_vmswap_kib": peak_swap_kib,
            "turns": turn_rows,
        }

    summary["status"] = "passed"
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
