#!/usr/bin/env python3
"""运行与封版 Table 3 同口径的 Qwen3-4B 离线准确率评测。"""

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback
import urllib.request

import wins_offline_mobile_queue as mobile

WINS_ROOT = mobile.DEFAULT_OUT_ROOT
BASE_BINARY = Path(mobile.DEFAULT_BIN)
BASE_SHA = "5f76c24c8a66865c1a3b6e282921d6c6c9910a4d1e1eaaee684869e340a2def1"
COMMIT = "05b88107d89c7253d33f91603c878ffe348c0593"
MODEL_4B_SHA = "a6603667c91ef018c50bc3c36490398c0885b134349707f71c88e0671fe8a55b"
MODEL_4B = Path.home() / "ai-workspace/cache/a2o-qwen3-4b/Qwen3-4B-Q4_0.gguf"
BENCHMARKS = (
    ("droidtask", "droidtask-full", "droidtask-8b-full158-05b8810", 158, 718, 18951, mobile.dt_camp),
    ("androidcontrol_low", "androidcontrol-low-120", "androidcontrol-low-8b-120-05b8810", 120, 728, 18961, mobile.ac_camp),
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n")
    pending.replace(path)


def check_gpu():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                       "--format=csv,noheader"], text=True)
    if "RTX 4090" not in gpu or active.strip():
        raise RuntimeError(f"RTX 4090 unavailable or busy: {gpu.strip()} {active.strip()}")


def audited_start(start):
    def launch(cond, cfg, port, out_dir, runtime):
        check_gpu()
        proc = start(cond, cfg, port, out_dir, runtime)
        try:
            path = out_dir / "conditions" / cond["name"]
            log = (path / "agent-server.log").read_text(errors="replace")
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
            if not offload or offload[1] != offload[2] or "NVIDIA GeForce RTX 4090" not in log:
                raise RuntimeError("full RTX 4090 model offload was not verified")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                save(path / "health.json", json.load(response))
        except BaseException:
            mobile.stop_local_proc(proc)
            raise
        return proc
    return launch


def evaluate(spec, target, config, conditions, runtime):
    benchmark, _, sealed, n_tasks, n_turns, port, campaign = spec
    worker = "droidtask" if benchmark == "droidtask" else "androidcontrol"
    mobile.run_job(benchmark, str(config), f"scripts/collect/host_{worker}_worker.py",
                   campaign.PROTOCOL_VERSION, campaign.build_summary, target,
                   conditions, n_tasks, 1, [port], runtime)
    rows = read(target / "summary.json")["conditions"]
    if {row["condition"] for row in rows} != set(conditions):
        raise RuntimeError(f"missing condition results at {target}")
    reference = read(WINS_ROOT / sealed / "conditions/full_kv/worker-summary.json")
    for row in rows:
        worker_path = target / "conditions" / row["condition"] / "worker-summary.json"
        selected = read(worker_path)["selected_task_ids"]
        if (row["total_tasks"], row["total_turns"]) != (n_tasks, n_turns) or selected != reference["selected_task_ids"]:
            raise RuntimeError(f"sealed task cohort mismatch at {worker_path}")
    return rows


def append_rows(rows, benchmark, variants, origin):
    for row in variants:
        rows.append({
            "benchmark": benchmark, "condition": row["condition"],
            "kv_keep_ratio": row["kv_keep_ratio"], "tasks": row["total_tasks"],
            "turns": row["total_turns"], "complete_turns": row["complete_turns"], "correct_turns": row["valid_turns"],
            "accuracy_pct": 100 * row["turn_accuracy"], "eviction_pct": 100 * row["cum_evict_vs_logical"],
            "compaction_turns": row["compact_turns"], "prior_turns": row["prior_turns"], "source": str(origin),
        })


def write_table(path, rows):
    with path.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True)
    lock = (WINS_ROOT / ".mobile_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage = "preflight"
    status = out / "status.json"
    try:
        check_gpu()
        base = mobile.local_runtime_identity(str(BASE_BINARY), mobile.BUILD_ID, 4, 256, COMMIT)
        if base["binary_sha256"] != BASE_SHA:
            raise RuntimeError("sealed host agent identity changed")
        configs = {spec[0]: mobile.ROOT / f"configs/host-kv-accuracy-{spec[1]}-8b.json" for spec in BENCHMARKS}
        mobile.start_local_agent = audited_start(mobile.start_local_agent)
        if digest(MODEL_4B) != MODEL_4B_SHA:
            raise RuntimeError("4B model checksum mismatch")
        stage = "table2"
        save(status, {"stage": stage, "state": "running", "started_unix": time.time()})
        rows_for_table = []
        for spec in BENCHMARKS:
            benchmark = spec[0]
            sealed_config = configs[benchmark]
            cfg = read(sealed_config)
            sealed_summary = read(WINS_ROOT / spec[2] / "conditions/full_kv/worker-summary.json")
            config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            if sealed_summary["config_sha256"] != config_sha or sealed_summary["runtime"] != base:
                raise RuntimeError(f"sealed Table 3 identity mismatch: {benchmark}")
            cfg.update(name=f"20261124-{benchmark}-4b", model_path=str(MODEL_4B), model_id="qwen3-4b-q4_0",
                       model_sha256=MODEL_4B_SHA)
            config = out / f"{benchmark}-config.json"
            save(config, cfg)
            target = out / benchmark
            rows = evaluate(spec, target, config, ["full_kv", "recent_only", "m2_prior_mid"], base)
            append_rows(rows_for_table, benchmark, rows, target)
            write_table(out / "table2.csv", rows_for_table)
        save(status, {"stage": stage, "state": "completed", "finished_unix": time.time()})
    except BaseException:
        save(status, {"stage": stage, "state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
