#!/usr/bin/env python3
"""3-way controlled comparison on real telecom tool-spec payload (7.1k+ pinned tokens):
  1. cpu_baseline (single-resident CPU, kv_reduce=0)
  2. m1_m2_no_hwm_trim (M1+M2, A2O_KV_HWM_TRIM=0)
  3. m1_m2_with_hwm_trim (M1+M2, A2O_KV_HWM_TRIM=1)
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.workloads.toolsandbox import post

ENDPOINT = "http://10.29.131.50:18780"
CFG = json.loads((ROOT / "configs/tau2-qwen3-4b-joint-v2-e2e.json").read_text())
OUT_DIR = ROOT / "results/20261118-telecom-swap-3way"
TELECOM_FIXTURE = OUT_DIR / "telecom_fixture.json"
SSH_CMD = [
    "ssh", "-p", "8022",
    "-i", str(Path.home() / ".ssh/a2o_xiaomi14pro_ed25519"),
    "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", f"UserKnownHostsFile={ROOT / 'configs/termux-known-hosts'}",
    "u0_a320@10.29.131.50",
]


def read_mem(svc_name: str) -> dict:
    out = subprocess.check_output(
        SSH_CMD + [
            f"p=$(cat /data/data/com.termux/files/home/.local/a2o/runs/{svc_name}/pid); "
            "grep -E '^(VmRSS|VmHWM|VmSwap):' /proc/$p/status; "
            "awk '{print \"majflt=\"$12}' /proc/$p/stat"
        ],
        text=True,
    )
    res = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            res[k.strip() + "_mib"] = int(v.strip().split()[0]) / 1024.0
        elif ln.startswith("majflt="):
            res["majflt"] = int(ln.split("=", 1)[1])
    return res


def run_service(action: str, name: str, cond: str, single_resident: bool, kv_reduce: int, hwm_trim: int) -> None:
    svc_dir = OUT_DIR / f"svc-{name}"
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), action,
        "--condition", cond, "--output", str(svc_dir), "--retry-network",
    ]
    if action == "start":
        cmd += [
            "--binary", str((ROOT / CFG["binary"]).resolve()),
            "--registry", str((ROOT / CFG["registry"]).resolve()),
            "--model", CFG["model_path"],
            "--model-id", CFG["model_id"],
            "--model-sha256", CFG["model_sha256"],
            "--context-size", str(CFG["context_size"]),
            "--cpu-threads", str(CFG["cpu_threads"]),
            "--cpu-repack", "0", "--opbatch", "1024", "--opqueue", "16",
            "--kv-offload", "0", "--flash-attention", "0",
            "--kv-reduce", str(kv_reduce),
            "--kv-policy", "adaptive_cost" if kv_reduce else "prefill_prior",
            "--kv-sink-tokens", "4", "--kv-recent-tokens", "256",
            "--kv-pool-window", "5", "--kv-tail-tokens", "64",
            "--compact-anchor", str(kv_reduce),
            "--kv-hwm-trim", str(hwm_trim),
            "--expected-decode", "96",
        ]
        if single_resident:
            cmd.append("--single-resident")
        if kv_reduce:
            cmd += ["--kv-costs", str((ROOT / CFG["kv_costs"]).resolve())]
        if cond == "route":
            cmd += ["--route-costs", str((ROOT / CFG["route_costs"]).resolve())]
    else:
        cmd.append("--logs-only")
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_telecom_turns() -> tuple[list[dict], list[list[dict]]]:
    fix = json.loads(TELECOM_FIXTURE.read_text())
    tools = fix["tools"]
    base_msgs = [
        {"role": "system", "content": fix["policy"]},
        {"role": "user", "content": "Hi, my phone has been showing No Service and Data Disabled for hours. Can you check my account and lines? My name is John Smith and email is john.smith@example.com."},
    ]
    turn1_extra = (
        "Diagnostic customer lookup & line telemetry snapshot:\n"
        + ("line_status=SUSPENDED, bill_id=B1002, overdue=120.50, sim_state=LOCKED, airplane_mode=ON, roaming=OFF.\n" * 60)
        + "\nPlease summarize the exact issues and next tool step briefly."
    )
    turn2_extra = (
        "Updated device & billing verification trace:\n"
        + ("bill_status=PAID, sim_pin_attempts=1, network_attach=PENDING, apn=ims_default, signal_dbm=-112.\n" * 50)
        + "\nProvide a concise diagnosis."
    )
    return tools, [base_msgs, [{"role": "user", "content": turn1_extra}], [{"role": "user", "content": turn2_extra}]]


def run_one_condition(name: str, cond: str, single_resident: bool, kv_reduce: int, hwm_trim: int) -> dict:
    tools, turn_inputs = build_telecom_turns()
    svc_name = f"svc-{name}"
    run_service("start", name, cond, single_resident, kv_reduce, hwm_trim)
    rows = []
    try:
        m_init = read_mem(svc_name)
        print(f"[{name} init] RSS={m_init['VmRSS_mib']:.1f}MiB Swap={m_init['VmSwap_mib']:.1f}MiB", flush=True)
        msgs = list(turn_inputs[0])
        for ti, extra in enumerate(turn_inputs):
            if ti > 0:
                msgs.extend(extra)
            resp = post(
                ENDPOINT,
                "/v1/chat/completions",
                {"messages": msgs, "tools": tools, "max_tokens": 64, "temperature": 0, "seed": 20261118},
                1800,
            )
            t = resp["timings"]
            m = read_mem(svc_name)
            dec_spd = t["predicted_ms"] / max(1, t["predicted_n"])
            kvr = t.get("kv_reduce") or {}
            row = {
                "turn": ti,
                "route": t["route"],
                "H_anchor": t["H_anchor"],
                "H_effective": t["H_effective"],
                "P": t["P_evaluated"],
                "D": t["predicted_n"],
                "prompt_ms": t["prompt_ms"],
                "decode_ms": t["predicted_ms"],
                "decode_ms_per_tok": dec_spd,
                "kv_trim_ms": kvr.get("kv_trim_ms", 0.0),
                "kv_hwm_trimmed": kvr.get("kv_hwm_trimmed", False),
                "kept_count": kvr.get("kept_count"),
                **m,
            }
            rows.append(row)
            print(
                f"  [{name} t={ti}] route={row['route']} H={row['H_anchor']}(eff={row['H_effective']}) P={row['P']} D={row['D']} "
                f"prm={row['prompt_ms']:.0f}ms dec={dec_spd:.1f}ms/t trim={row['kv_trim_ms']:.0f}ms "
                f"| RSS={m['VmRSS_mib']:.0f}MiB HWM={m['VmHWM_mib']:.0f}MiB Swap={m['VmSwap_mib']:.0f}MiB",
                flush=True,
            )
            msgs.append(resp["choices"][0]["message"])
    finally:
        run_service("stop", name, cond, single_resident, kv_reduce, hwm_trim)
    return {"condition": name, "init_mem": m_init, "turns": rows}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("cpu_baseline", "cpu", True, 0, 0),
        ("m1_m2_no_hwm_trim", "route", False, 1, 0),
        ("m1_m2_with_hwm_trim", "route", False, 1, 1),
    ]
    results = []
    for name, cond, sr, kvr, trim in specs:
        out_p = OUT_DIR / f"{name}.json"
        if out_p.exists():
            results.append(json.loads(out_p.read_text()))
            print(f"[skip] loaded existing {out_p.name}", flush=True)
            continue
        res = run_one_condition(name, cond, sr, kvr, trim)
        out_p.write_text(json.dumps(res, indent=2))
        results.append(res)
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print("\nSaved 3-way telecom swap comparison to", OUT_DIR / "summary.json", flush=True)


if __name__ == "__main__":
    main()
