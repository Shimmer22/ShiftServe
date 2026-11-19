#!/usr/bin/env python3
"""Sequential Host KV-accuracy pipeline on wins:
  1. Resume & finish 4B BFCL Multi-Turn (60 tasks x 5 conditions, 3-way)
  2. 8B ToolSandbox (28 scenarios x 5 conditions, 2-way)
  3. 8B BFCL Multi-Turn (60 tasks x 5 conditions, 2-way)
  4. 8B tau2-bench ratio ablation (36 tasks x 5 conditions, 1-way + 4B user simulator)

Strictly host-only: never touches phone Termux SSH or ADB.
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import queue
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/collect"))
from host_kv_accuracy_queue import (
    TS_CONTAINER,
    TS_PORTS,
    TS_PYTHON,
    TS_SCRIPT,
    start_ts_agent,
)
from host_kv_accuracy_tau2_campaign import (
    WINS_HOST,
    stop_remote_pid,
    sync_workspace,
)


def run_toolsandbox_8b(cfg_path: Path, out_dir: Path, max_workers: int = 2) -> None:
    cfg = json.loads(cfg_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_workspace()
    wins_ts_base = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/{out_dir.name}"
    container_ts_base = f"/workspace/outputs/a2o-gap-profiling/20261118-host-kv-accuracy/{out_dir.name}"

    def run_one(cond: dict, port_q: queue.Queue) -> dict:
        cname = cond["name"]
        local_cdir = out_dir / "conditions" / cname
        local_cdir.mkdir(parents=True, exist_ok=True)
        run_out_local = local_cdir / "run"
        if (run_out_local / "completion.json").exists():
            print(f"[toolsandbox-8b] {cname}: already complete, skipping.", flush=True)
            return json.loads((run_out_local / "summary.json").read_text())

        port = port_q.get()
        pid = ""
        try:
            print(f"[toolsandbox-8b] {cname}: starting phone_service_host on :{port} ...", flush=True)
            # Reuse start_ts_agent with custom log/pid dir via a small override
            from host_kv_accuracy_tau2_campaign import WINS_BIN, WINS_CUDA_BIN, _spawn_remote_bg
            import time
            cdir = f"{wins_ts_base}/conditions/{cname}"
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
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"ToolSandbox 8B agent ({cname}) failed on :{port}")

            cond_cfg = {
                "endpoint": f"http://127.0.0.1:{port}",
                "model": "Qwen3-8B-Q4_0.gguf",
                "model_id": cfg["model_id"],
                "model_sha256": cfg["model_sha256"],
                "backend": "CPU",
                "context_size": cfg["context_size"],
                "max_tokens": cfg["max_tokens"],
                "request_timeout_s": cfg["request_timeout_s"],
                "seed": cfg["seed"],
                "kv_policy": cond["kv_policy"] if cond["kv_reduce"] else "none",
                "concurrency": 1,
                "collector_lock": f"{container_ts_base}/conditions/{cname}/collector.lock",
                "scenarios": cfg["scenarios"],
                "a2o_service": True,
            }
            remote_cfg = f"{wins_ts_base}/conditions/{cname}/run-cfg.json"
            container_cfg = f"{container_ts_base}/conditions/{cname}/run-cfg.json"
            container_run = f"{container_ts_base}/conditions/{cname}/run"
            q = shlex.quote
            subprocess.run(
                ["ssh", WINS_HOST, f"rm -rf {q(wins_ts_base + '/conditions/' + cname + '/run')} && cat > {q(remote_cfg)}"],
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
                    raise RuntimeError(f"ToolSandbox 8B {cname} exited with code {rc}")
            subprocess.run(
                ["rsync", "-az", f"{WINS_HOST}:{wins_ts_base}/conditions/{cname}/run/", str(run_out_local) + "/"],
                check=True,
            )
            summary = json.loads((run_out_local / "summary.json").read_text())
            print(
                f"[toolsandbox-8b] Finished {cname}: completed={summary['completed']} "
                f"trace_complete={summary['trace_complete']} full_score={summary['full_score']}",
                flush=True,
            )
            return summary
        finally:
            stop_remote_pid(pid)
            port_q.put(port)

    port_q: queue.Queue[int] = queue.Queue()
    for pt in TS_PORTS[:max_workers]:
        port_q.put(pt)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_one, c, port_q): c for c in cfg["conditions"]}
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
    print("[pipeline] Step 1/4: 4B BFCL Multi-Turn (3-way) ...", flush=True)
    subprocess.run(
        [
            sys.executable, "-u",
            str(ROOT / "scripts/collect/host_kv_accuracy_bfcl_campaign.py"),
            "--config", str(ROOT / "configs/host-kv-accuracy-bfcl.json"),
            "--output", str(ROOT / "results/20261118-host-kv-accuracy/bfcl"),
            "--max-workers", "3",
        ],
        cwd=ROOT, check=True,
    )

    print("[pipeline] Step 2/4: 8B ToolSandbox (2-way) ...", flush=True)
    run_toolsandbox_8b(
        ROOT / "configs/host-kv-accuracy-toolsandbox-8b.json",
        ROOT / "results/20261118-host-kv-accuracy/toolsandbox-8b",
        max_workers=2,
    )

    print("[pipeline] Step 3/4: 8B BFCL Multi-Turn (2-way) ...", flush=True)
    subprocess.run(
        [
            sys.executable, "-u",
            str(ROOT / "scripts/collect/host_kv_accuracy_bfcl_campaign.py"),
            "--config", str(ROOT / "configs/host-kv-accuracy-bfcl-8b.json"),
            "--output", str(ROOT / "results/20261118-host-kv-accuracy/bfcl-8b"),
            "--max-workers", "2",
        ],
        cwd=ROOT, check=True,
    )

    print("[pipeline] Step 4/4: 8B tau2-bench ratio ablation (1-way) ...", flush=True)
    subprocess.run(
        [
            sys.executable, "-u",
            str(ROOT / "scripts/collect/host_kv_accuracy_tau2_campaign.py"),
            "--config", str(ROOT / "configs/host-kv-accuracy-tau2-ratio-8b.json"),
            "--output", str(ROOT / "results/20261118-host-kv-accuracy/tau2-ratio-8b"),
            "--max-workers", "1",
        ],
        cwd=ROOT, check=True,
    )
    print("[pipeline] All 4 stages completed!", flush=True)


if __name__ == "__main__":
    main()
