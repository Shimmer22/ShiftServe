#!/usr/bin/env python3
"""Teacher-Forced Turn-Level Host BFCL campaign on wins (RTX 4090 CUDA)."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

from host_kv_accuracy_bfcl_campaign import BFCL_CONTAINER, BFCL_ROOT, CONTAINER_PROJECT, start_bfcl_agent
from host_kv_accuracy_tau2_campaign import (
    WINS_HOST, add_run_options, agent_identity, install_run_plan,
    run_metadata, same_run, stop_remote_pid, stream_worker, sync_workspace, validated_summary, worker_scope,
)

DEFAULT_PORTS = [18941, 18942, 18943]
BUILD_ID = "20261120-t3-final-host"
DEFAULT_BIN = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261120-t3-final-host-build/agent"


def run_condition(cond, cfg, port_q, out_dir, wins_out_base, config_rel, meta):
    cname = cond["name"]
    local_cdir = out_dir / "conditions" / cname
    local_cdir.mkdir(parents=True, exist_ok=True)
    summary_path = local_cdir / "worker-summary.json"
    if summary_path.exists() and same_run(json.loads(summary_path.read_text()), meta):
        print(f"[bfcl-tf] {cname}: 已有同构建、同题集汇总", flush=True)
        return validated_summary(summary_path, meta)

    port = port_q.get()
    pid = ""
    try:
        print(f"[bfcl-tf] {cname}: starting phone_service_host on :{port} ...", flush=True)
        pid = start_bfcl_agent(cond, cfg, port, wins_out_base, meta["runtime"])
        print(f"[bfcl-tf] {cname}: agent ready (pid={pid}, port={port}), running turn-level worker ...", flush=True)
        q = shlex.quote
        container_out_base = wins_out_base.replace("/home/wins/ai-workspace", "/workspace", 1)
        cmd = (
            f"docker exec -e PYTHONPATH={q(BFCL_ROOT)} {q(BFCL_CONTAINER)} python3 -u "
            f"{q(CONTAINER_PROJECT + '/scripts/collect/host_bfcl_turn_level_worker.py')} "
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
        return validated_summary(summary_path, meta)
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
        kv_agg = {"compact_turns": 0, "prior_turns": 0, "recent_turns": 0, "reuse_turns": 0,
                  "before_tokens": 0, "logical_tokens": 0, "kept_tokens": 0, "evicted_tokens": 0}
        for r in s["results"]:
            for k in kv_agg:
                kv_agg[k] += r.get("kv_stats", {}).get(k, 0)
        cum_evict = (1.0 - kv_agg["kept_tokens"] / kv_agg["logical_tokens"]) if kv_agg["logical_tokens"] else 0.0
        rows.append({
            "condition": cname,
            "kv_reduce": cond["kv_reduce"],
            "kv_keep_ratio": cond.get("kv_keep_ratio", -1.0),
            "total_tasks": s["total_tasks"],
            "total_turns": s["total_turns"],
            "complete_turns": s["complete_turns"],
            "valid_turns": s["valid_turns"],
            "turn_accuracy": s["turn_accuracy"],
            "all_turns_valid_tasks": s["all_turns_valid_tasks"],
            "cum_evict_vs_logical": cum_evict,
            **kv_agg,
            "by_category": s["by_category"],
        })
    (out_dir / "summary.json").write_text(json.dumps({"conditions": rows}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/host-kv-accuracy-bfcl-8b.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/mainline-eval/bfcl-8b-turn-tf")
    p.add_argument("--conditions", nargs="+", default=["full_kv", "m2_prior_mid", "m2_prior_high"])
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--limit-tasks", type=int, default=0)
    add_run_options(p)
    p.set_defaults(agent_binary=DEFAULT_BIN, agent_build_id=BUILD_ID)
    args = p.parse_args()

    cfg = json.loads(args.config.resolve().read_text())
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (out_dir / "campaign.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    sync_workspace()
    conds = [c for c in cfg["conditions"] if c["name"] in args.conditions]
    wins_out_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/{out_dir.name}"
    config_rel = str(args.config.resolve().relative_to(ROOT))
    runtime = agent_identity(args)
    metadata = {c["name"]: run_metadata("bfcl-turn-tf-v1", cfg, c["name"], args.limit_tasks,
                                        args.task_id, runtime) for c in conds}
    install_run_plan(out_dir, metadata)

    port_q: queue.Queue[int] = queue.Queue()
    for pt in DEFAULT_PORTS[: args.max_workers]:
        port_q.put(pt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(run_condition, c, cfg, port_q, out_dir, wins_out_base, config_rel, metadata[c["name"]]) for c in conds]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            print(
                f"[bfcl-tf] Finished {res['condition']}: valid_turns={res['valid_turns']}/{res['total_turns']} "
                f"turn_acc={res['turn_accuracy']*100:.2f}% (all_turns_tasks={res['all_turns_valid_tasks']}/{res['total_tasks']})",
                flush=True,
            )
            build_summary(out_dir, cfg, metadata)


if __name__ == "__main__":
    main()
