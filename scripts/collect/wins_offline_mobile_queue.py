#!/usr/bin/env python3
"""Detached on-wins queue for AndroidControl-Low Top-50 and DroidTask Top-50."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

from a2o.profiling.host_accuracy import run_metadata, same_run
import host_androidcontrol_campaign as ac_camp
import host_droidtask_campaign as dt_camp

WINS_CUDA_BIN = "/home/wins/ai-workspace/tools/llama.cpp-a2o-cuda/bin"
DEFAULT_BIN = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261122-final-host-05b8810-build/agent"
DEFAULT_OUT_ROOT = Path("/home/wins/ai-workspace/outputs/a2o-gap-profiling")
BUILD_ID = "20261122-final-host-05b8810"


def local_runtime_identity(binary: str, build_id: str, stride: int, min_evict: int, git_commit: str = "") -> dict:
    sha = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    ident = {
        "binary": binary,
        "binary_sha256": sha,
        "build_id": build_id,
        "kv_layer_stride": stride,
        "kv_min_evict_tokens": min_evict,
    }
    if git_commit:
        ident["git_commit"] = git_commit
    return ident


def start_local_agent(cond: dict, cfg: dict, port: int, out_dir: Path, runtime: dict) -> subprocess.Popen:
    cdir = out_dir / "conditions" / cond["name"]
    cdir.mkdir(parents=True, exist_ok=True)
    log_path = cdir / "agent-server.log"
    pid_path = cdir / "agent-server.pid"

    env = os.environ.copy()
    env.update({
        "LD_LIBRARY_PATH": f"{Path(runtime['binary']).parent}:{WINS_CUDA_BIN}",
        "A2O_HOST_GPU_LAYERS": "99",
        "A2O_CPU_THREADS": str(cfg["cpu_threads"]),
        "A2O_FLASH_ATTN": "0",
        "A2O_KV_REDUCE": str(cond["kv_reduce"]),
        "A2O_KV_LAYER_STRIDE": str(runtime["kv_layer_stride"]),
        "A2O_KV_MIN_EVICT_TOKENS": str(runtime["kv_min_evict_tokens"]),
        "A2O_ENABLE_THINKING": str(int(cfg.get("agent_enable_thinking", False))),
        "A2O_KV_POLICY": str(cond["kv_policy"]),
        "A2O_KV_KEEP_TOKENS": str(cond["kv_keep_tokens"]),
        "A2O_KV_KEEP_RATIO": str(cond.get("kv_keep_ratio", -1.0)),
        "A2O_KV_SINK_TOKENS": "4",
        "A2O_KV_RECENT_TOKENS": "256",
        "A2O_KV_POOL_WINDOW": "5",
        "A2O_KV_TAIL_TOKENS": "64",
        "A2O_COMPACT_ANCHOR": str(cond["compact_anchor"]),
        "A2O_SINGLE_RESIDENT": "1",
        "A2O_MODEL_ID": cfg["model_id"],
    })
    if "kv_hwm_trim" in cfg:
        env["A2O_KV_HWM_TRIM"] = str(int(bool(cfg["kv_hwm_trim"])))

    lf = log_path.open("wb")
    proc = subprocess.Popen(
        [runtime["binary"], cfg["model_path"], "cpu", str(cfg["context_size"]), str(port), "none"],
        cwd=WINS_CUDA_BIN,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    lf.close()
    pid_path.write_text(str(proc.pid))

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Agent exited early rc={proc.returncode}, see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                health = json.loads(resp.read().decode())
            for key, expected in {
                "model": cfg["model_id"],
                "context_size": cfg["context_size"],
                "kv_reduce": bool(cond["kv_reduce"]),
                "kv_policy": cond["kv_policy"],
                "compact_anchor": bool(cond["compact_anchor"]),
                "kv_layer_stride": runtime["kv_layer_stride"],
                "kv_min_evict_tokens": runtime["kv_min_evict_tokens"],
            }.items():
                if health.get(key) != expected:
                    raise RuntimeError(f"Health mismatch {key}: {health.get(key)} != {expected}")
            if "kv_hwm_trim" in cfg and health.get("kv_hwm_trim") != bool(cfg["kv_hwm_trim"]):
                raise RuntimeError(f"Health mismatch kv_hwm_trim: {health.get('kv_hwm_trim')}")
            if runtime["build_id"] and health.get("build_id") != runtime["build_id"]:
                raise RuntimeError(f"Build mismatch: {health.get('build_id')}")
            if runtime.get("git_commit") and health.get("git_commit") != runtime["git_commit"]:
                raise RuntimeError(f"Git commit mismatch: {health.get('git_commit')} != {runtime['git_commit']}")
            return proc
        except Exception:
            time.sleep(1)

    stop_local_proc(proc)
    raise TimeoutError(f"Agent failed health check on :{port}, see {log_path}")


def stop_local_proc(proc: subprocess.Popen | None):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass


def run_job(
    name: str,
    cfg_rel: str,
    worker_rel: str,
    protocol_version: str,
    build_summary_fn,
    out_dir: Path,
    cond_names: list[str],
    limit_tasks: int,
    max_workers: int,
    ports: list[int],
    runtime: dict,
):
    cfg_path = ROOT / cfg_rel
    cfg = json.loads(cfg_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = [c for c in cfg["conditions"] if c["name"] in set(cond_names)]
    metadata = {
        c["name"]: run_metadata(protocol_version, cfg, c["name"], limit_tasks, [], runtime)
        for c in conds
    }
    plan_path = out_dir / "run-metadata.json"
    plan_payload = {**metadata, "_selected_conditions": [c["name"] for c in conds]}
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan_payload:
            raise ValueError(f"Metadata mismatch in existing {plan_path}")
    else:
        plan_path.write_text(json.dumps(plan_payload, indent=2) + "\n")

    port_q: queue.Queue[int] = queue.Queue()
    for p in ports[:max_workers]:
        port_q.put(p)

    def _run_cond(cond: dict):
        cname = cond["name"]
        cdir = out_dir / "conditions" / cname
        cdir.mkdir(parents=True, exist_ok=True)
        sp = cdir / "worker-summary.json"
        if sp.exists() and same_run(json.loads(sp.read_text()), metadata[cname]):
            print(f"[{name}] {cname}: cached summary verified, skipping", flush=True)
            return

        port = port_q.get()
        proc = None
        try:
            print(f"[{name}] {cname}: starting local agent on :{port} (kv_hwm_trim={cfg.get('kv_hwm_trim')})", flush=True)
            proc = start_local_agent(cond, cfg, port, out_dir, runtime)
            print(f"[{name}] {cname}: agent ready pid={proc.pid}, launching worker ...", flush=True)
            argv = [
                sys.executable, "-u", str(ROOT / worker_rel),
                "--config", str(cfg_path),
                "--condition-name", cname,
                "--agent-base", f"http://127.0.0.1:{port}",
                "--output", str(cdir),
                "--limit-tasks", str(limit_tasks),
                "--runtime-identity", json.dumps(runtime),
            ]
            with (cdir / "worker.log").open("w") as wf:
                res = subprocess.run(argv, stdout=wf, stderr=subprocess.STDOUT, text=True)
            if res.returncode != 0:
                tail = "\n".join((cdir / "worker.log").read_text().splitlines()[-30:])
                raise RuntimeError(f"[{name}] {cname} worker failed rc={res.returncode}:\n{tail}")
            s = json.loads(sp.read_text())
            if not same_run(s, metadata[cname]):
                raise RuntimeError(f"[{name}] {cname} worker-summary metadata mismatch")
            print(f"[{name}] {cname}: DONE acc={s['valid_turns']}/{s['total_turns']} ({s['turn_accuracy']*100:.2f}%)", flush=True)
        finally:
            stop_local_proc(proc)
            port_q.put(port)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_cond, c) for c in conds]
        for f in futs:
            f.result()

    build_summary_fn(out_dir, cfg, metadata)
    print(f"[{name}] SUMMARY WRITTEN -> {out_dir / 'summary.json'}", flush=True)
    print((out_dir / "summary.json").read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--agent-binary", default=DEFAULT_BIN)
    p.add_argument("--agent-build-id", default=BUILD_ID)
    p.add_argument("--expected-git-commit", default="")
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--ac-cfg", default="configs/host-kv-accuracy-androidcontrol-low-120-8b.json")
    p.add_argument("--dt-cfg", default="configs/host-kv-accuracy-droidtask-full-8b.json")
    p.add_argument("--ac-out-name", default="androidcontrol-low-8b-120-05b8810")
    p.add_argument("--dt-out-name", default="droidtask-8b-full158-05b8810")
    p.add_argument("--ac-limit", type=int, default=120)
    p.add_argument("--dt-limit", type=int, default=158)
    p.add_argument("--jobs", nargs="+", choices=["dt", "ac"], default=["dt", "ac"])
    p.add_argument("--conditions", nargs="+", default=["full_kv", "recent_only", "m2_prior_mid", "m2_prior_high"])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--lock-name", default=".mobile_queue.lock")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    lock_f = (args.out_root / args.lock_name).open("w")
    fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)

    runtime = local_runtime_identity(
        args.agent_binary, args.agent_build_id, stride=args.stride, min_evict=256,
        git_commit=args.expected_git_commit,
    )
    print(f"[queue] started pid={os.getpid()} runtime={json.dumps(runtime)}", flush=True)

    for job in args.jobs:
        if job == "dt":
            run_job(
                name=args.dt_out_name,
                cfg_rel=args.dt_cfg,
                worker_rel="scripts/collect/host_droidtask_worker.py",
                protocol_version=dt_camp.PROTOCOL_VERSION,
                build_summary_fn=dt_camp.build_summary,
                out_dir=args.out_root / args.dt_out_name,
                cond_names=args.conditions,
                limit_tasks=args.dt_limit,
                max_workers=args.max_workers,
                ports=[18951, 18952, 18953],
                runtime=runtime,
            )
        elif job == "ac":
            run_job(
                name=args.ac_out_name,
                cfg_rel=args.ac_cfg,
                worker_rel="scripts/collect/host_androidcontrol_worker.py",
                protocol_version=ac_camp.PROTOCOL_VERSION,
                build_summary_fn=ac_camp.build_summary,
                out_dir=args.out_root / args.ac_out_name,
                cond_names=args.conditions,
                limit_tasks=args.ac_limit,
                max_workers=args.max_workers,
                ports=[18961, 18962, 18963],
                runtime=runtime,
            )
    print("[queue] ALL JOBS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
