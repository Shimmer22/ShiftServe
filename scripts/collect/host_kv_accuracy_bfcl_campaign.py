#!/usr/bin/env python3
"""Parallel Host KV-accuracy campaign on wins (RTX 4090 CUDA) for BFCL Multi-Turn (60 tasks x 5 conditions).

Strictly host-only: never touches phone Termux SSH or ADB.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/collect"))
from host_kv_accuracy_tau2_campaign import (
    WINS_BIN,
    WINS_CUDA_BIN,
    WINS_HOST,
    WINS_PROJECT,
    _spawn_remote_bg,
    stop_remote_pid,
    sync_workspace,
)

WINS_BFCL_BASE = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/bfcl"
CONTAINER_PROJECT = "/workspace/projects/a2o-gap-profiling"
BFCL_CONTAINER = "a2o-bfcl-lc200"
BFCL_ROOT = "/workspace/references/gorilla-bfcl/berkeley-function-call-leaderboard"
BFCL_PORTS = [18931, 18932, 18933]


def start_bfcl_agent(cond: dict, cfg: dict, port: int, wins_out_base: str) -> str:
    cname = cond["name"]
    cdir = f"{wins_out_base}/conditions/{cname}"
    env_vars = {
        "LD_LIBRARY_PATH": WINS_CUDA_BIN,
        "A2O_HOST_GPU_LAYERS": "99",
        "A2O_CPU_THREADS": str(cfg["cpu_threads"]),
        "A2O_FLASH_ATTN": "0",
        "A2O_KV_REDUCE": str(cond["kv_reduce"]),
        "A2O_KV_POLICY": str(cond["kv_policy"]),
        "A2O_KV_KEEP_TOKENS": str(cond["kv_keep_tokens"]),
        "A2O_KV_KEEP_RATIO": str(cond.get("kv_keep_ratio", -1.0)),
        "A2O_KV_SINK_TOKENS": "4",
        "A2O_KV_RECENT_TOKENS": "128",
        "A2O_KV_POOL_WINDOW": "5",
        "A2O_KV_TAIL_TOKENS": "32",
        "A2O_COMPACT_ANCHOR": str(cond["compact_anchor"]),
        "A2O_SINGLE_RESIDENT": "1",
        "A2O_MODEL_ID": cfg["model_id"],
    }
    pid = _spawn_remote_bg(
        [WINS_BIN, cfg["model_path"], "cpu", str(cfg["context_size"]), str(port), "none"],
        cwd=WINS_CUDA_BIN,
        env_extra=env_vars,
        log_path=f"{cdir}/agent-server.log",
        pid_path=f"{cdir}/agent-server.pid",
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        chk = subprocess.run(
            ["ssh", "-n", WINS_HOST, f"curl -fsS http://127.0.0.1:{port}/health >/dev/null 2>&1"],
            capture_output=True, timeout=15,
        )
        if chk.returncode == 0:
            return pid
        time.sleep(1)
    raise TimeoutError(f"BFCL agent ({cname}) failed to start on :{port}")


def run_condition(cond: dict, cfg: dict, port_q: queue.Queue, out_dir: Path, wins_out_base: str, config_rel: str, limit_tasks: int) -> dict:
    cname = cond["name"]
    local_cdir = out_dir / "conditions" / cname
    local_cdir.mkdir(parents=True, exist_ok=True)
    if (local_cdir / "worker-summary.json").exists():
        print(f"[bfcl] {cname}: already complete, skipping.", flush=True)
        return json.loads((local_cdir / "worker-summary.json").read_text())

    port = port_q.get()
    pid = ""
    try:
        print(f"[bfcl] {cname}: starting phone_service_host on :{port} ...", flush=True)
        pid = start_bfcl_agent(cond, cfg, port, wins_out_base)
        print(f"[bfcl] {cname}: agent ready (pid={pid}, port={port}), running bfcl worker ...", flush=True)
        q = shlex.quote
        container_out_base = wins_out_base.replace("/home/wins/ai-workspace", "/workspace", 1)
        cmd = (
            f"docker exec -e PYTHONPATH={q(BFCL_ROOT)} {q(BFCL_CONTAINER)} python3 -u "
            f"{q(CONTAINER_PROJECT + '/scripts/collect/host_kv_accuracy_bfcl_worker.py')} "
            f"--config {q(CONTAINER_PROJECT + '/' + config_rel)} "
            f"--condition-name {q(cname)} "
            f"--agent-base http://127.0.0.1:{port} "
            f"--output {q(container_out_base + '/conditions/' + cname)} "
            f"--limit-tasks {limit_tasks}"
        )
        with (local_cdir / "worker.log").open("w") as lf:
            proc = subprocess.Popen(["ssh", "-n", WINS_HOST, cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                print(line, end="", flush=True)
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"BFCL {cname} worker exited with code {rc}")
        subprocess.run(
            ["rsync", "-az", f"{WINS_HOST}:{wins_out_base}/conditions/{cname}/", str(local_cdir) + "/"],
            check=True,
        )
        return json.loads((local_cdir / "worker-summary.json").read_text())
    finally:
        stop_remote_pid(pid)
        port_q.put(port)


def build_summary(out_dir: Path, cfg: dict):
    rows = []
    for cond in cfg["conditions"]:
        cname = cond["name"]
        sp = out_dir / "conditions" / cname / "worker-summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        rows.append({
            "condition": cname,
            "kv_reduce": cond["kv_reduce"],
            "kv_policy": cond["kv_policy"],
            "kv_keep_ratio": cond.get("kv_keep_ratio", -1.0),
            "total_tasks": s["total_tasks"],
            "complete": s["complete"],
            "valid": s["valid"],
            "accuracy": s["accuracy"],
            "by_category": s["by_category"],
        })
    (out_dir / "summary.json").write_text(json.dumps({"conditions": rows}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/host-kv-accuracy-bfcl.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/20261118-host-kv-accuracy/bfcl")
    p.add_argument("--conditions", nargs="+", default=[])
    p.add_argument("--max-workers", type=int, default=3)
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()

    cfg = json.loads(args.config.resolve().read_text())
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (out_dir / "campaign.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    sync_workspace()
    selected = args.conditions or [c["name"] for c in cfg["conditions"]]
    conds = [c for c in cfg["conditions"] if c["name"] in selected]
    wins_out_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/{out_dir.name}"
    config_rel = str(args.config.resolve().relative_to(ROOT))

    port_q: queue.Queue[int] = queue.Queue()
    for pt in BFCL_PORTS[: args.max_workers]:
        port_q.put(pt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(run_condition, c, cfg, port_q, out_dir, wins_out_base, config_rel, args.limit_tasks) for c in conds]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            print(
                f"[bfcl] Finished {res['condition']}: complete={res['complete']}/{res['total_tasks']} "
                f"valid={res['valid']}/{res['total_tasks']} acc={res['accuracy']*100:.2f}%",
                flush=True,
            )
            build_summary(out_dir, cfg)


if __name__ == "__main__":
    main()
