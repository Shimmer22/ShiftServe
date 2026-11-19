#!/usr/bin/env python3
"""Wait for wins RTX 4090 GPU to become idle, then run queued Host KV-accuracy campaigns:
  1. ToolSandbox (28 scenarios x 5 conditions, 3-way parallel)
  2. tau2-bench ratio ablation (36 tasks x 5 conditions, 2-way parallel)

Strictly host-only: never touches phone Termux SSH or ADB.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
    stop_remote_pid,
    sync_workspace,
    _spawn_remote_bg,
)

WINS_TS_BASE = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/toolsandbox"
CONTAINER_TS_BASE = "/workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/toolsandbox"
TS_CONTAINER = "a2o-toolsandbox-py311"
TS_PYTHON = "/workspace/tools/a2o-toolsandbox-py311-env/bin/python"
TS_SCRIPT = "/workspace/projects/a2o-gap-profiling/scripts/collect/toolsandbox.py"
TS_PORTS = [18921, 18922, 18923]


def wait_for_wins_gpu_idle(poll_s: int = 60, required_idle: int = 2) -> None:
    idle_streak = 0
    while True:
        out = subprocess.check_output(
            ["ssh", "-n", WINS_HOST, "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"],
            text=True,
            timeout=20,
        ).strip()
        if not out:
            idle_streak += 1
            print(f"[queue] GPU idle ({idle_streak}/{required_idle}) at {time.strftime('%H:%M:%S')}", flush=True)
            if idle_streak >= required_idle:
                return
        else:
            idle_streak = 0
            first_line = out.splitlines()[0]
            print(f"[queue] GPU busy ({first_line}); sleeping {poll_s}s at {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(poll_s)


def start_ts_agent(cond: dict, cfg: dict, port: int) -> str:
    cname = cond["name"]
    cdir = f"{WINS_TS_BASE}/conditions/{cname}"
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
        "A2O_KV_RECENT_TOKENS": "64",
        "A2O_KV_POOL_WINDOW": "5",
        "A2O_KV_TAIL_TOKENS": "16",
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
    raise TimeoutError(f"ToolSandbox agent ({cname}) failed to start on :{port}")


def run_ts_condition(cond: dict, cfg: dict, port_q: queue.Queue, out_dir: Path) -> dict:
    cname = cond["name"]
    local_cdir = out_dir / "conditions" / cname
    local_cdir.mkdir(parents=True, exist_ok=True)
    run_out_local = local_cdir / "run"
    if (run_out_local / "completion.json").exists():
        print(f"[toolsandbox] {cname}: already complete, skipping.", flush=True)
        return json.loads((run_out_local / "summary.json").read_text())

    port = port_q.get()
    pid = ""
    try:
        print(f"[toolsandbox] {cname}: starting phone_service_host on :{port} ...", flush=True)
        pid = start_ts_agent(cond, cfg, port)
        cond_cfg = {
            "endpoint": f"http://127.0.0.1:{port}",
            "model": "Qwen3-4B-Q4_0.gguf",
            "model_id": cfg["model_id"],
            "model_sha256": cfg["model_sha256"],
            "backend": "CPU",
            "context_size": cfg["context_size"],
            "max_tokens": cfg["max_tokens"],
            "request_timeout_s": cfg["request_timeout_s"],
            "seed": cfg["seed"],
            "kv_policy": cond["kv_policy"] if cond["kv_reduce"] else "none",
            "concurrency": 1,
            "collector_lock": f"{CONTAINER_TS_BASE}/conditions/{cname}/collector.lock",
            "scenarios": cfg["scenarios"],
            "a2o_service": True,
        }
        remote_cfg = f"{WINS_TS_BASE}/conditions/{cname}/run-cfg.json"
        container_cfg = f"{CONTAINER_TS_BASE}/conditions/{cname}/run-cfg.json"
        container_run = f"{CONTAINER_TS_BASE}/conditions/{cname}/run"
        q = shlex.quote
        subprocess.run(
            ["ssh", WINS_HOST, f"rm -rf {q(WINS_TS_BASE + '/conditions/' + cname + '/run')} && cat > {q(remote_cfg)}"],
            input=json.dumps(cond_cfg, indent=2) + "\n",
            text=True,
            check=True,
        )
        cmd = f"docker exec {q(TS_CONTAINER)} {q(TS_PYTHON)} -u {q(TS_SCRIPT)} --config {q(container_cfg)} --output {q(container_run)}"
        with (local_cdir / "worker.log").open("w") as lf:
            proc = subprocess.Popen(["ssh", "-n", WINS_HOST, cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ToolSandbox {cname} exited with code {rc}")
        subprocess.run(
            ["rsync", "-az", f"{WINS_HOST}:{WINS_TS_BASE}/conditions/{cname}/run/", str(run_out_local) + "/"],
            check=True,
        )
        summary = json.loads((run_out_local / "summary.json").read_text())
        print(
            f"[toolsandbox] Finished {cname}: completed={summary['completed']} "
            f"trace_complete={summary['trace_complete']} full_score={summary['full_score']}",
            flush=True,
        )
        return summary
    finally:
        stop_remote_pid(pid)
        port_q.put(port)


def run_toolsandbox_campaign(out_dir: Path, max_workers: int = 3) -> None:
    cfg = json.loads((ROOT / "configs/host-kv-accuracy-toolsandbox.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_workspace()
    port_q: queue.Queue[int] = queue.Queue()
    for pt in TS_PORTS[:max_workers]:
        port_q.put(pt)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_ts_condition, c, cfg, port_q, out_dir): c for c in cfg["conditions"]}
        for fut in concurrent.futures.as_completed(futs):
            c = futs[fut]
            s = fut.result()
            sims = [float(r["evaluation"]["similarity"]) for r in s["results"] if r.get("evaluation")]
            rows.append({
                "condition": c["name"],
                "kv_reduce": c["kv_reduce"],
                "kv_policy": c["kv_policy"],
                "kv_keep_ratio": c.get("kv_keep_ratio", -1.0),
                "completed": s["completed"],
                "trace_complete": s["trace_complete"],
                "full_score": s["full_score"],
                "mean_similarity": sum(sims) / len(sims) if sims else 0.0,
            })
    (out_dir / "summary.json").write_text(json.dumps({"conditions": rows}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--skip-wait", action="store_true")
    p.add_argument("--skip-tau2-ratio", action="store_true")
    args = p.parse_args()

    if not args.skip_wait:
        print("[queue] Waiting for wins RTX 4090 GPU to become idle ...", flush=True)
        wait_for_wins_gpu_idle(poll_s=args.poll_seconds, required_idle=2)

    print("[queue] GPU idle! Starting Stage 1: ToolSandbox (28 scenarios x 5 conditions) ...", flush=True)
    run_toolsandbox_campaign(ROOT / "results/20261118-host-kv-accuracy/toolsandbox", max_workers=3)

    if not args.skip_tau2_ratio:
        print("[queue] Starting Stage 2: tau2-bench ratio ablation (36 tasks x 5 conditions) ...", flush=True)
        subprocess.run(
            [
                sys.executable, "-u",
                str(ROOT / "scripts/collect/host_kv_accuracy_tau2_campaign.py"),
                "--config", str(ROOT / "configs/host-kv-accuracy-tau2-ratio.json"),
                "--output", str(ROOT / "results/20261118-host-kv-accuracy/tau2-ratio"),
            ],
            cwd=ROOT,
            check=True,
        )
    print("[queue] All queued campaigns completed!", flush=True)


if __name__ == "__main__":
    main()
