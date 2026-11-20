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
    add_run_options,
    agent_identity,
    install_run_plan,
    verify_agent_health,
    worker_scope,
    stream_worker,
    validated_summary,
    run_metadata,
    same_run,
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


def start_bfcl_agent(cond: dict, cfg: dict, port: int, wins_out_base: str, runtime: dict) -> str:
    cname = cond["name"]
    cdir = f"{wins_out_base}/conditions/{cname}"
    env_vars = {
        "LD_LIBRARY_PATH": WINS_CUDA_BIN,
        "A2O_HOST_GPU_LAYERS": "99",
        "A2O_CPU_THREADS": str(cfg["cpu_threads"]),
        "A2O_FLASH_ATTN": "0",
        "A2O_KV_REDUCE": str(cond["kv_reduce"]),
        "A2O_KV_LAYER_STRIDE": str(runtime["kv_layer_stride"]),
        "A2O_KV_MIN_EVICT_TOKENS": str(runtime["kv_min_evict_tokens"]),
        "A2O_ENABLE_THINKING": str(int(cfg.get("enable_thinking", False))),
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
        [runtime["binary"], cfg["model_path"], "cpu", str(cfg["context_size"]), str(port), "none"],
        cwd=WINS_CUDA_BIN,
        env_extra=env_vars,
        log_path=f"{cdir}/agent-server.log",
        pid_path=f"{cdir}/agent-server.pid",
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        chk = subprocess.run(
            ["ssh", "-n", WINS_HOST, f"curl -fsS http://127.0.0.1:{port}/health"],
            capture_output=True, text=True, timeout=15,
        )
        if chk.returncode == 0:
            try:
                verify_agent_health(json.loads(chk.stdout), runtime, cfg, cond, pid)
            except Exception:
                stop_remote_pid(pid)
                raise
            return pid
        time.sleep(1)
    stop_remote_pid(pid)
    raise TimeoutError(f"BFCL agent ({cname}) failed to start on :{port}")


def bfcl_summary(path, meta):
    summary = validated_summary(path, meta)
    cases = summary["results"]
    if summary["selected_task_ids"] != [row["id"] for row in cases]:
        raise RuntimeError(f"BFCL 汇总题集与逐题结果不一致: {path}")
    for row in cases:
        raw = path.parent / "tasks" / row["id"] / "result.json"
        if json.loads(raw.read_text()) != row:
            raise RuntimeError(f"BFCL 汇总与逐题文件不一致: {raw}")
    return summary


def run_condition(cond, cfg, port_q, out_dir, wins_out_base, config_rel, meta):
    cname = cond["name"]
    local_cdir = out_dir / "conditions" / cname
    local_cdir.mkdir(parents=True, exist_ok=True)
    summary_path = local_cdir / "worker-summary.json"
    if summary_path.exists() and same_run(json.loads(summary_path.read_text()), meta):
        print(f"[bfcl] {cname}: 已有同构建、同题集汇总", flush=True)
        return bfcl_summary(summary_path, meta)

    port = port_q.get()
    pid = ""
    try:
        print(f"[bfcl] {cname}: starting phone_service_host on :{port} ...", flush=True)
        pid = start_bfcl_agent(cond, cfg, port, wins_out_base, meta["runtime"])
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
            + worker_scope(meta)
        )
        stream_worker(cmd, local_cdir / "worker.log")
        subprocess.run(
            ["rsync", "-az", f"{WINS_HOST}:{wins_out_base}/conditions/{cname}/", str(local_cdir) + "/"],
            check=True,
        )
        return bfcl_summary(summary_path, meta)
    finally:
        stop_remote_pid(pid)
        port_q.put(port)


def build_summary(out_dir: Path, cfg: dict, metadata: dict):
    rows = []
    for cond in cfg["conditions"]:
        cname = cond["name"]
        sp = out_dir / "conditions" / cname / "worker-summary.json"
        if cname not in metadata or not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if not same_run(s, metadata[cname]):
            continue
        rows.append({
            **metadata[cname],
            "selected_task_ids": s["selected_task_ids"],
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
    p.add_argument("--agent-port", type=int, action="append", help="wins 服务端口，可重复指定")
    p.add_argument("--limit-tasks", type=int, default=0)
    add_run_options(p)
    args = p.parse_args()

    cfg = json.loads(args.config.resolve().read_text())
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (out_dir / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[bfcl] 等待当前目录的任务完成: {out_dir}", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)

    sync_workspace()
    selected = args.conditions or [c["name"] for c in cfg["conditions"]]
    conds = [c for c in cfg["conditions"] if c["name"] in selected]
    wins_out_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/{out_dir.name}"
    config_rel = str(args.config.resolve().relative_to(ROOT))
    runtime = agent_identity(args)
    metadata = {c["name"]:run_metadata("bfcl-fc-v2", cfg, c["name"], args.limit_tasks,
                                       args.task_id, runtime) for c in conds}
    install_run_plan(out_dir, metadata)

    port_q: queue.Queue[int] = queue.Queue()
    for pt in (args.agent_port or BFCL_PORTS)[: args.max_workers]:
        port_q.put(pt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(run_condition, c, cfg, port_q, out_dir, wins_out_base, config_rel, metadata[c["name"]]) for c in conds]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            print(
                f"[bfcl] Finished {res['condition']}: complete={res['complete']}/{res['total_tasks']} "
                f"valid={res['valid']}/{res['total_tasks']} acc={res['accuracy']*100:.2f}%",
                flush=True,
            )
            build_summary(out_dir, cfg, metadata)


if __name__ == "__main__":
    main()
