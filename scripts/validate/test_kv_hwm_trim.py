#!/usr/bin/env python3
"""Verify M2 KV high-water-mark memory trim on phone route service."""
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
SSH_CMD = [
    "ssh", "-p", "8022",
    "-i", str(Path.home() / ".ssh/a2o_xiaomi14pro_ed25519"),
    "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", f"UserKnownHostsFile={ROOT / 'configs/termux-known-hosts'}",
    "u0_a320@10.29.131.50",
]


def read_mem() -> dict:
    out = subprocess.check_output(
        SSH_CMD + [
            "p=$(cat /data/data/com.termux/files/home/.local/a2o/runs/svc-hwm-trim/pid); "
            "grep -E '^(VmRSS|VmHWM|VmSwap):' /proc/$p/status"
        ],
        text=True,
    )
    res = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            res[k.strip()] = int(v.strip().split()[0]) / 1024.0
    return res


def route_svc(action: str) -> None:
    svc_dir = ROOT / "results/20261118-kv-hwm-trim/svc-hwm-trim"
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), action,
        "--condition", "route", "--output", str(svc_dir), "--retry-network",
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
            "--kv-reduce", "1", "--kv-policy", "adaptive_cost",
            "--kv-costs", str((ROOT / CFG["kv_costs"]).resolve()),
            "--route-costs", str((ROOT / CFG["route_costs"]).resolve()),
            "--kv-sink-tokens", "4", "--kv-recent-tokens", "256",
            "--kv-pool-window", "5", "--kv-tail-tokens", "64",
            "--compact-anchor", "1", "--expected-decode", "64",
        ]
    else:
        cmd.append("--logs-only")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    route_svc("start")
    try:
        m0 = read_mem()
        print(f"[init] RSS={m0['VmRSS']:.1f} MiB Swap={m0['VmSwap']:.1f} MiB", flush=True)

        # Turn 0: short system prompt (~200 tok) + large user message (~5500 tok) -> triggers high HWM
        big_block = "Order telemetry record: item=phone-case, status=delivered, qty=1, notes=verified.\n" * 280
        msgs = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": big_block + "\nReply with OK only."},
        ]
        r1 = post(ENDPOINT, "/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0}, 1200)
        m1 = read_mem()
        t1 = r1["timings"]
        print(
            f"[turn 0] route={t1['route']} P={t1['P_evaluated']} kv_reduce={t1['kv_reduce']} "
            f"| RSS={m1['VmRSS']:.1f} MiB HWM={m1['VmHWM']:.1f} MiB Swap={m1['VmSwap']:.1f} MiB",
            flush=True,
        )

        # Turn 1: append another large observation (~3000 tok) -> triggers HTP + M2 compact + HWM trim
        msgs.append(r1["choices"][0]["message"])
        msgs.append({"role": "user", "content": ("Additional log entry: node=edge-2, latency=12ms.\n" * 180) + "\nReply with DONE."})
        r2 = post(ENDPOINT, "/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0}, 1200)
        m2 = read_mem()
        t2 = r2["timings"]
        print(
            f"[turn 1] route={t2['route']} H_anchor={t2['H_anchor']} P={t2['P_evaluated']} "
            f"kv_reduce={t2['kv_reduce']} | RSS={m2['VmRSS']:.1f} MiB HWM={m2['VmHWM']:.1f} MiB Swap={m2['VmSwap']:.1f} MiB",
            flush=True,
        )

        # Reset should also release any residual HWM
        post(ENDPOINT, "/a2o/reset", {}, 60)
        m3 = read_mem()
        print(f"[after reset] RSS={m3['VmRSS']:.1f} MiB HWM={m3['VmHWM']:.1f} MiB Swap={m3['VmSwap']:.1f} MiB", flush=True)
    finally:
        route_svc("stop")


if __name__ == "__main__":
    main()
