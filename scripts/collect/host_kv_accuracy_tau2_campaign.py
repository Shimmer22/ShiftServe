#!/usr/bin/env python3
"""Parallel Host KV-accuracy campaign on wins (RTX 4090 CUDA) for tau2-bench (36 tasks x 5 conditions).

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
WINS_HOST = "wins"
WINS_PROJECT = "/home/wins/ai-workspace/projects/a2o-gap-profiling"
CONTAINER_PROJECT = "/workspace/projects/a2o-gap-profiling"
WINS_OUT_BASE = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/tau2"
CONTAINER_OUT_BASE = "/workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/tau2"
WINS_BIN = "/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/bin/phone_service_host"
WINS_CUDA_BIN = "/home/wins/ai-workspace/tools/llama.cpp-a2o-cuda/bin"
CONTAINER = "a2o-tau2-q4-vllm-20261118"
CONTAINER_PYTHON = "/workspace/private/a2o-tau2/env/bin/python3"
USER_PORT = 18909
AGENT_PORTS = [18911, 18912]


def load_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def sync_workspace():
    for sub in ["configs", "scripts", "src"]:
        subprocess.run(
            [
                "rsync", "-az", "--exclude", "__pycache__/", "--exclude", "*.pyc",
                str(ROOT / sub) + "/", f"{WINS_HOST}:{WINS_PROJECT}/{sub}/",
            ],
            check=True,
        )


def _spawn_remote_bg(cmd_args: list[str], cwd: str, env_extra: dict[str, str], log_path: str, pid_path: str) -> str:
    py_launcher = (
        "import json,os,signal,subprocess,sys;"
        "spec=json.loads(sys.argv[1]);"
        "os.makedirs(os.path.dirname(spec['log']),exist_ok=True);"
        "\ntry:\n"
        " old=int(open(spec['pid']).read().strip());os.kill(old,signal.SIGTERM)\n"
        "except Exception: pass\n"
        "env=os.environ.copy();env.update(spec['env']);"
        "lf=open(spec['log'],'wb');"
        "p=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);"
        "open(spec['pid'],'w').write(str(p.pid));print(p.pid)"
    )
    spec = json.dumps({"argv": cmd_args, "cwd": cwd, "env": env_extra, "log": log_path, "pid": pid_path})
    return subprocess.check_output(
        ["ssh", "-n", WINS_HOST, f"python3 -c {shlex.quote(py_launcher)} {shlex.quote(spec)}"],
        text=True, timeout=30,
    ).strip()


def start_user_server(model_path: str, context_size: int, wins_out_base: str = WINS_OUT_BASE) -> str:
    log_path = f"{wins_out_base}/user-server.log"
    pid_path = f"{wins_out_base}/user-server.pid"
    pid = _spawn_remote_bg(
        [
            f"{WINS_CUDA_BIN}/llama-server",
            "-m", model_path,
            "--alias", "qwen3-4b-q4_0",
            "--host", "127.0.0.1",
            "--port", str(USER_PORT),
            "-ngl", "99",
            "-c", str(context_size),
            "-np", "2",
            "-t", "8",
        ],
        cwd=WINS_CUDA_BIN,
        env_extra={"LD_LIBRARY_PATH": WINS_CUDA_BIN},
        log_path=log_path,
        pid_path=pid_path,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        chk = subprocess.run(
            ["ssh", "-n", WINS_HOST, f"curl -fsS http://127.0.0.1:{USER_PORT}/health >/dev/null 2>&1"],
            capture_output=True, timeout=15,
        )
        if chk.returncode == 0:
            return pid
        time.sleep(1)
    raise TimeoutError(f"User llama-server did not become ready on :{USER_PORT}")


def stop_remote_pid(pid: str):
    if pid:
        subprocess.run(["ssh", "-n", WINS_HOST, f"kill {shlex.quote(pid)} 2>/dev/null || true"], check=False, timeout=15)


def start_agent_server(cond: dict, cfg: dict, port: int, wins_out_base: str = WINS_OUT_BASE) -> str:
    q = shlex.quote
    cname = cond["name"]
    cdir = f"{wins_out_base}/conditions/{cname}"
    log_path = f"{cdir}/agent-server.log"
    pid_path = f"{cdir}/agent-server.pid"
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
        "A2O_KV_RECENT_TOKENS": "256",
        "A2O_KV_POOL_WINDOW": "5",
        "A2O_KV_TAIL_TOKENS": "64",
        "A2O_COMPACT_ANCHOR": str(cond["compact_anchor"]),
        "A2O_SINGLE_RESIDENT": "1",
        "A2O_MODEL_ID": cfg["model_id"],
    }
    pid = _spawn_remote_bg(
        [WINS_BIN, cfg["model_path"], "cpu", str(cfg["context_size"]), str(port), "none"],
        cwd=WINS_CUDA_BIN,
        env_extra=env_vars,
        log_path=log_path,
        pid_path=pid_path,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        chk = subprocess.run(
            ["ssh", "-n", WINS_HOST, f"curl -fsS http://127.0.0.1:{port}/health"],
            capture_output=True, text=True, timeout=15,
        )
        if chk.returncode == 0:
            return pid
        time.sleep(1)
    log = subprocess.check_output(["ssh", "-n", WINS_HOST, f"tail -n 40 {q(log_path)} 2>/dev/null || true"], text=True, timeout=15)
    raise TimeoutError(f"Agent phone_service_host ({cname}) failed on :{port}:\n{log}")


def run_condition(cond: dict, cfg: dict, env: dict[str, str], port_q: queue.Queue, out_dir: Path, limit_tasks: int, wins_out_base: str = WINS_OUT_BASE, container_out_base: str = CONTAINER_OUT_BASE, config_rel: str = "configs/host-kv-accuracy-tau2.json") -> dict:
    cname = cond["name"]
    local_cdir = out_dir / "conditions" / cname
    local_cdir.mkdir(parents=True, exist_ok=True)
    if (local_cdir / "worker-summary.json").exists():
        print(f"[campaign] {cname}: already complete, skipping.", flush=True)
        return json.loads((local_cdir / "worker-summary.json").read_text())

    port = port_q.get()
    agent_pid = ""
    try:
        print(f"[campaign] {cname}: starting phone_service_host on :{port} ...", flush=True)
        agent_pid = start_agent_server(cond, cfg, port, wins_out_base=wins_out_base)
        print(f"[campaign] {cname}: agent ready (pid={agent_pid}, port={port}), running tau2 worker ...", flush=True)
        q = shlex.quote
        worker_cmd = (
            f"docker exec "
            f"-e JUDGE_BASE_URL={q(env['JUDGE_BASE_URL'])} "
            f"-e JUDGE_API_KEY={q(env['JUDGE_API_KEY'])} "
            f"-e JUDGE_MODEL={q(env['JUDGE_MODEL'])} "
            f"{q(CONTAINER)} {q(CONTAINER_PYTHON)} -u {q(CONTAINER_PROJECT + '/scripts/collect/host_kv_accuracy_tau2_worker.py')} "
            f"--config {q(CONTAINER_PROJECT + '/' + config_rel)} "
            f"--condition-name {q(cname)} "
            f"--output {q(container_out_base + '/conditions/' + cname)} "
            f"--agent-base http://127.0.0.1:{port} "
            f"--user-base http://127.0.0.1:{USER_PORT} "
            f"--limit-tasks {limit_tasks}"
        )
        log_file = local_cdir / "worker.log"
        with log_file.open("w") as lf:
            proc = subprocess.Popen(["ssh", WINS_HOST, worker_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                print(line, end="", flush=True)
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"{cname} worker exited with code {rc}")
        subprocess.run(
            ["rsync", "-az", f"{WINS_HOST}:{wins_out_base}/conditions/{cname}/", str(local_cdir) + "/"],
            check=True,
        )
        return json.loads((local_cdir / "worker-summary.json").read_text())
    finally:
        stop_remote_pid(agent_pid)
        port_q.put(port)


def build_summary(out_dir: Path, cfg: dict):
    rows = []
    for cond in cfg["conditions"]:
        cname = cond["name"]
        sp = out_dir / "conditions" / cname / "worker-summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        by_dom = {}
        for r in s.get("results", []):
            d = r["domain"]
            by_dom.setdefault(d, {"total": 0, "complete": 0, "reward_sum": 0.0})
            by_dom[d]["total"] += 1
            by_dom[d]["complete"] += int(bool(r.get("complete")))
            by_dom[d]["reward_sum"] += float(r.get("reward") or 0.0)
        for d, st in by_dom.items():
            st["reward_mean"] = st["reward_sum"] / st["total"] if st["total"] else 0.0
        rows.append({
            "condition": cname,
            "kv_reduce": cond["kv_reduce"],
            "kv_policy": cond["kv_policy"],
            "kv_keep_tokens": cond["kv_keep_tokens"],
            "total_tasks": s["total_tasks"],
            "completed_tasks": s["completed_tasks"],
            "reward_sum": s["reward_sum"],
            "reward_mean": s["reward_mean"],
            "by_domain": by_dom,
        })
    (out_dir / "summary.json").write_text(json.dumps({"conditions": rows}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/host-kv-accuracy-tau2.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/20261118-host-kv-accuracy/tau2")
    p.add_argument("--conditions", nargs="+", default=[])
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()

    cfg = json.loads(args.config.resolve().read_text())
    env = load_env(ROOT / ".env")
    for k in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"):
        if not env.get(k):
            raise RuntimeError(f"Missing {k} in .env")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (out_dir / "campaign.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    sync_workspace()
    selected = args.conditions or [c["name"] for c in cfg["conditions"]]
    conds = [c for c in cfg["conditions"] if c["name"] in selected]

    wins_out_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/{out_dir.name}"
    container_out_base = f"/workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/{out_dir.name}"
    config_rel = str(args.config.resolve().relative_to(ROOT))

    print(f"[campaign] Starting shared user simulator llama-server on wins:{USER_PORT} ...", flush=True)
    user_pid = start_user_server(cfg.get("user_model_path", cfg["model_path"]), cfg["context_size"], wins_out_base=wins_out_base)
    print(f"[campaign] User simulator ready (pid={user_pid}).", flush=True)

    port_q: queue.Queue[int] = queue.Queue()
    for pt in AGENT_PORTS[: args.max_workers]:
        port_q.put(pt)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futs = [pool.submit(run_condition, c, cfg, env, port_q, out_dir, args.limit_tasks, wins_out_base, container_out_base, config_rel) for c in conds]
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                print(
                    f"[campaign] Finished {res['condition']}: "
                    f"completed={res['completed_tasks']}/{res['total_tasks']} "
                    f"reward_mean={res['reward_mean']:.4f} (sum={res['reward_sum']:.1f})",
                    flush=True,
                )
                build_summary(out_dir, cfg)
    finally:
        stop_remote_pid(user_pid)


if __name__ == "__main__":
    main()
