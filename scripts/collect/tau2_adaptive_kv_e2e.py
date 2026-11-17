#!/usr/bin/env python3
"""Orchestrate live multi-turn tau2-bench E2E campaign across host user simulator (wins) and phone agent."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analyze"))

from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network
from tau2_adaptive_kv_e2e import summarize_campaign

PHONE_ENDPOINT = "http://10.29.131.50:18780"
WINS_HOST = "wins"
WINS_PROJECT = "/home/wins/ai-workspace/projects/a2o-gap-profiling"
CONTAINER_PROJECT = "/workspace/projects/a2o-gap-profiling"


def route_phone(action: str, cfg: dict, cond: dict, svc_dir: Path, log_path: Path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run/route_phone.py"),
        action,
        "--condition",
        cond["condition"],
        "--output",
        str(svc_dir),
        "--retry-network",
    ]
    if action == "start":
        cmd += [
            "--binary", str((ROOT / cfg["binary"]).resolve()),
            "--registry", str((ROOT / cfg["registry"]).resolve()),
            "--model", cfg["model_path"],
            "--model-id", cfg["model_id"],
            "--model-sha256", cfg["model_sha256"],
            "--context-size", str(cfg["context_size"]),
            "--cpu-threads", str(cfg["cpu_threads"]),
            "--cpu-repack", str(cond.get("cpu_repack", cfg["cpu_repack"])),
            "--opbatch", str(cfg["opbatch"]),
            "--opqueue", str(cfg["opqueue"]),
            "--kv-offload", str(cfg["kv_offload"]),
            "--flash-attention", str(cfg["flash_attention"]),
            "--kv-reduce", str(cond["kv_reduce"]),
            "--kv-policy", cond["kv_policy"],
            "--kv-sink-tokens", "4",
            "--kv-recent-tokens", "256",
            "--kv-pool-window", "5",
            "--kv-tail-tokens", "64",
            "--compact-anchor", str(cond["compact_anchor"]),
            "--expected-decode", str(cfg["expected_decode_tokens"]),
        ]
        if cond.get("single_resident"):
            cmd.append("--single-resident")
        if cond.get("use_kv_costs"):
            cmd += ["--kv-costs", str((ROOT / cfg["kv_costs"]).resolve())]
        if cond.get("use_route_costs") or (cond["condition"] == "route" and cfg.get("route_costs")):
            rc_path = cond.get("route_costs") or cfg["route_costs"]
            cmd += ["--route-costs", str((ROOT / rc_path).resolve())]
    else:
        cmd.append("--logs-only")
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def sample_phone_memory(svc_dir: Path) -> dict:
    t = DeviceTransport(ROOT, "ssh", config=ROOT / "configs/termux-ssh.json")
    remote = t.remote_root + "/" + svc_dir.name
    q = shlex.quote
    out = retry_network(
        lambda: t.call(
            "shell",
            f"p=$(cat {q(remote + '/pid')}); grep -E '^(VmRSS|VmHWM|VmSwap|VmSize):' /proc/$p/status",
        ).stdout.decode()
    )
    parsed = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def ensure_wins_workspace():
    for sub in ["configs", "scripts", "src"]:
        subprocess.run(
            ["rsync", "-az", "--exclude", "__pycache__/", "--exclude", "*.pyc",
             str(ROOT / sub) + "/", f"{WINS_HOST}:{WINS_PROJECT}/{sub}/"],
            check=True,
        )


def start_wins_user_server(remote_server_dir: str, log_path: Path):
    q = shlex.quote
    # Launch tau2_host_server.py on wins in background, cleaning stop-server first.
    subprocess.run(
        ["ssh", WINS_HOST, f"rm -f {q(remote_server_dir + '/stop-server')} && mkdir -p {q(remote_server_dir)}"],
        check=True,
    )
    cmd = [
        "ssh", WINS_HOST,
        f"nohup python3 -u {q(WINS_PROJECT + '/scripts/run/tau2_host_server.py')} --output {q(remote_server_dir)} "
        f"> {q(remote_server_dir + '/driver.log')} 2>&1 < /dev/null & echo $!",
    ]
    pid = subprocess.check_output(cmd, text=True).strip()
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        ready = subprocess.run(
            ["ssh", WINS_HOST, f"test -f {q(remote_server_dir + '/server-ready.json')} && curl -fsS http://127.0.0.1:18769/health"],
            capture_output=True, text=True,
        )
        if ready.returncode == 0:
            return pid
        alive = subprocess.run(["ssh", WINS_HOST, f"kill -0 {shlex.quote(pid)}"], capture_output=True)
        if alive.returncode != 0:
            log = subprocess.check_output(["ssh", WINS_HOST, f"tail -n 60 {q(remote_server_dir + '/driver.log')} {q(remote_server_dir + '/server.log')} 2>/dev/null || true"], text=True)
            raise RuntimeError(f"wins vLLM server exited early:\n{log}")
        time.sleep(2)
    raise TimeoutError("Timed out waiting for wins vLLM server on :18769")


def stop_wins_user_server(remote_server_dir: str):
    q = shlex.quote
    subprocess.run(["ssh", WINS_HOST, f"touch {q(remote_server_dir + '/stop-server')}"], check=False)
    for _ in range(30):
        res = subprocess.run(
            ["ssh", WINS_HOST, "curl -fsS http://127.0.0.1:18769/health >/dev/null 2>&1"],
            capture_output=True,
        )
        if res.returncode != 0:
            break
        time.sleep(1)


class PhoneTunnelGuard:
    """Keep the SSH reverse tunnel alive across transient disconnects."""

    def __init__(self):
        self._stop = False
        self._proc = None
        self._spawn()
        import threading
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _spawn(self):
        self._proc = subprocess.Popen(
            [
                "ssh", "-N", "-T",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "TCPKeepAlive=yes",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=8",
                "-R", "18780:10.29.131.50:18780",
                WINS_HOST,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _watch(self):
        while not self._stop:
            time.sleep(2.0)
            if not self._stop and self._proc is not None and self._proc.poll() is not None:
                self._spawn()

    def terminate(self):
        self._stop = True
        if self._proc is not None:
            self._proc.terminate()

    def wait(self, timeout=None):
        if self._proc is not None:
            return self._proc.wait(timeout=timeout)
        return 0

    def kill(self):
        self._stop = True
        if self._proc is not None:
            self._proc.kill()


def open_phone_tunnel() -> PhoneTunnelGuard:
    guard = PhoneTunnelGuard()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        proc = guard._proc
        if proc.poll() is not None:
            err = proc.stderr.read().decode() if proc.stderr else ""
            guard.terminate()
            raise RuntimeError(f"SSH reverse tunnel failed: {err}")
        chk = subprocess.run(["ssh", WINS_HOST, "curl -fsS http://127.0.0.1:18780/health >/dev/null"], capture_output=True)
        if chk.returncode == 0:
            return guard
        time.sleep(0.5)
    guard.terminate()
    raise TimeoutError("SSH reverse tunnel did not become ready on wins:18780")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/tau2-qwen3-1_7b-adaptive-e2e.json")
    p.add_argument("--limit-tasks", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=[])
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    cfg = json.loads(args.config.resolve().read_text())
    out = (args.output or (ROOT / cfg["output_dir"])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_file = (out / "campaign.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    selected_names = args.conditions or [c["name"] for c in cfg["conditions"]]
    conds = [c for c in cfg["conditions"] if c["name"] in selected_names]

    remote_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/{out.name}"
    remote_vllm = f"{remote_base}/vllm-server"
    ensure_wins_workspace()

    use_ninfer = cfg.get("user_backend") == "ninfer"
    user_base = cfg.get("user_base", "http://10.29.243.183:18881" if use_ninfer else "http://127.0.0.1:18769")
    user_model = cfg.get("user_model", "qwen3.8-27b" if use_ninfer else "qwen35-9b")
    if use_ninfer:
        print(f"[e2e] Verifying Qwen3.8-27B NInfer user/judge server on {user_base} ...", flush=True)
        subprocess.run(
            ["ssh", WINS_HOST,
             "docker start qwen38-ninfer >/dev/null 2>&1 || true; "
             "KEY=$(grep '^NINFER_API_KEY=' /home/wins/qwen38-ninfer/.env | cut -d= -f2-); "
             "for i in $(seq 1 20); do "
             f"if curl -fsS -H \"Authorization: Bearer $KEY\" {shlex.quote(user_base.rstrip('/') + '/v1/models')} >/dev/null 2>&1; then exit 0; fi; sleep 2; "
             "done; exit 1"],
            check=True,
        )
    else:
        print("[e2e] Starting Qwen3.5-9B user/judge server on wins:18769 ...", flush=True)
        start_wins_user_server(remote_vllm, out / "vllm-driver.log")
    try:
        for cond in conds:
            cname = cond["name"]
            cdir = out / "conditions" / cname
            cdir.mkdir(parents=True, exist_ok=True)
            if (cdir / "worker-summary.json").exists():
                print(f"[e2e] Condition {cname} already completed; skipping.", flush=True)
                continue

            svc_dir = out / f"service-{cname}"
            svc_log = out / f"service-{cname}.log"
            remote_cdir = f"{remote_base}/conditions/{cname}"
            container_cdir = f"/workspace/outputs/a2o-gap-profiling/{out.name}/conditions/{cname}"

            print(f"[e2e] Starting phone service for {cname} ...", flush=True)
            route_phone("start", cfg, cond, svc_dir, svc_log)
            tunnel = None
            try:
                mem_before = sample_phone_memory(svc_dir)
                tunnel = open_phone_tunnel()
                cfg_rel = args.config.resolve().relative_to(ROOT)
                env_prefix = ""
                key_args = ""
                if use_ninfer:
                    env_prefix = "KEY=$(grep '^NINFER_API_KEY=' /home/wins/qwen38-ninfer/.env | cut -d= -f2-) && "
                    key_args = " --user-api-key-env NINFER_API_KEY"
                    docker_env = "-e NINFER_API_KEY=\"$KEY\" "
                else:
                    docker_env = ""
                worker_cmd = (
                    f"{env_prefix}"
                    f"docker exec {docker_env}-e PYTHONPATH=/workspace/references/tau2-bench-v0.1.0-source/src "
                    f"a2o-bfcl-lc200 /workspace/private/a2o-tau2/env/bin/python3 -u {CONTAINER_PROJECT}/scripts/collect/tau2_e2e_worker.py "
                    f"--config {CONTAINER_PROJECT}/{cfg_rel} "
                    f"--condition-name {shlex.quote(cname)} "
                    f"--output {shlex.quote(container_cdir)} "
                    f"--user-base {shlex.quote(user_base)} "
                    f"--user-model {shlex.quote(user_model)}{key_args} "
                    f"--limit-tasks {int(args.limit_tasks)}"
                )
                print(f"[e2e] Running tau2 worker for {cname} (limit_tasks={args.limit_tasks}) ...", flush=True)
                subprocess.run(["ssh", WINS_HOST, worker_cmd], check=True)
                mem_after = sample_phone_memory(svc_dir)
                (cdir / "phone-memory.json").write_text(json.dumps({"before": mem_before, "after": mem_after}, indent=2) + "\n")
                subprocess.run(["rsync", "-az", f"{WINS_HOST}:{remote_cdir}/", str(cdir) + "/"], check=True)
            finally:
                if tunnel is not None:
                    tunnel.terminate()
                    try:
                        tunnel.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        tunnel.kill()
                print(f"[e2e] Stopping phone service for {cname} ...", flush=True)
                route_phone("stop", cfg, cond, svc_dir, svc_log)
    finally:
        if not use_ninfer:
            print("[e2e] Stopping wins user/judge server ...", flush=True)
            stop_wins_user_server(remote_vllm)

    summary = summarize_campaign(out, [c["name"] for c in conds])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[e2e] DONE:", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
