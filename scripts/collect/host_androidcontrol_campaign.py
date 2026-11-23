#!/usr/bin/env python3
"""Host KV-accuracy campaign for Teacher-Forced Multi-Turn AndroidControl on wins."""
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

from host_androidcontrol_worker import PROTOCOL_VERSION
from host_kv_accuracy_tau2_campaign import (
    WINS_HOST, WINS_PROJECT, add_run_options, agent_identity, install_run_plan,
    run_metadata, same_run, start_agent_server, stop_remote_pid, stream_worker,
    sync_workspace, validated_summary, worker_scope,
)

DEFAULT_PORTS = [18961, 18962, 18963]
BUILD_ID = "20261120-t3-final-host"
DEFAULT_BIN = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261120-t3-final-host-build/agent"


def run_condition(cond, cfg, port_q, out_dir, wins_out_base, config_rel, meta):
    cname = cond["name"]
    local_cdir = out_dir / "conditions" / cname
    local_cdir.mkdir(parents=True, exist_ok=True)
    summary_path = local_cdir / "worker-summary.json"
    if summary_path.exists() and same_run(json.loads(summary_path.read_text()), meta):
        print(f"[androidcontrol] {cname}: 已有同构建、同题集汇总", flush=True)
        return validated_summary(summary_path, meta)

    port = port_q.get()
    pid = ""
    try:
        print(f"[androidcontrol] {cname}: starting phone_service_host on :{port} ...", flush=True)
        pid = start_agent_server(cond, cfg, port, wins_out_base, meta["runtime"])
        print(f"[androidcontrol] {cname}: agent ready (pid={pid}, port={port}), running worker ...", flush=True)
        q = shlex.quote
        cmd = (
            f"python3 -u {q(WINS_PROJECT + '/scripts/collect/host_androidcontrol_worker.py')} "
            f"--config {q(WINS_PROJECT + '/' + config_rel)} "
            f"--condition-name {q(cname)} "
            f"--agent-base http://127.0.0.1:{port} "
            f"--output {q(wins_out_base + '/conditions/' + cname)} "
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
            "strict_valid_turns": s["strict_valid_turns"],
            "turn_accuracy": s["turn_accuracy"],
            "strict_turn_accuracy": s["strict_turn_accuracy"],
            "all_turns_valid_tasks": s["all_turns_valid_tasks"],
            "cum_evict_vs_logical": cum_evict,
            **kv_agg,
        })
    (out_dir / "summary.json").write_text(json.dumps({"conditions": rows}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/host-kv-accuracy-androidcontrol-8b.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/mainline-eval/androidcontrol-8b-pilot")
    p.add_argument("--limit-tasks", type=int, default=20)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--conditions", nargs="+")
    add_run_options(p)
    p.set_defaults(agent_binary=DEFAULT_BIN, agent_build_id=BUILD_ID)
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    config_rel = str(args.config.resolve().relative_to(ROOT))
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wins_out_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/{out_dir.name}"
    runtime = agent_identity(args)

    conds = cfg["conditions"]
    if args.conditions:
        wanted = set(args.conditions)
        conds = [c for c in conds if c["name"] in wanted]
        if len(conds) != len(wanted):
            raise ValueError(f"Unknown condition in {args.conditions}")

    metadata = {
        c["name"]: run_metadata(PROTOCOL_VERSION, cfg, c["name"], args.limit_tasks, args.task_id, runtime)
        for c in conds
    }
    install_run_plan(out_dir, {**metadata, "_selected_conditions": [c["name"] for c in conds]})

    lock_f = (out_dir / ".campaign.lock").open("w")
    fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sync_workspace()

    workers = max(1, min(args.max_workers, len(conds), len(DEFAULT_PORTS)))
    port_q: queue.Queue[int] = queue.Queue()
    for port in DEFAULT_PORTS[:workers]:
        port_q.put(port)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_condition, c, cfg, port_q, out_dir, wins_out_base, config_rel, metadata[c["name"]]) for c in conds]
        for fut in futs:
            fut.result()

    build_summary(out_dir, cfg, metadata)
    print(json.dumps(json.loads((out_dir / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
