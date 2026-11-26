#!/usr/bin/env python3
"""Detached sequential runner for Stage 2 (Fig 2(a) ob1024 sweep + plot) -> Stage 3 (Table 1 10-case queue)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "results/mainline-eval/t6-final-ob1024"
FIG2A_CFG = ROOT / "configs/prefill-shared-16-64-h4096-cpu6-ob1024-t10.json"
FIG2A_DIR = ROOT / "results/mainline-eval/prefill-shared-16-64-h4096-8b-ob1024"
FIG2A_PNG = FIG2A_DIR / "figures/prefill-shared-16-64-ob1024.png"
ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"
COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
QUEUE_STATUS = ROOT / "results/mainline-eval/t6-8b-table1-10case-ob1024-queue-status.json"
STATUS_FILE = ROOT / "results/mainline-eval/20261125-stage2-stage3-status.json"


def save_status(payload: dict) -> None:
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(STATUS_FILE)


def main() -> None:
    state = {
        "status": "running",
        "stage": "stage2_fig2a",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    save_status(state)

    if not FIG2A_PNG.is_file():
        print("=== Stage 2: Fig. 2(a) ob1024 clean sweep (H=4096, P=16..64 step 4) ===", flush=True)
        subprocess.run(
            [
                sys.executable, "-u", str(ROOT / "scripts/collect/pd_stage_sweep.py"),
                "--config", str(FIG2A_CFG),
                "--output", str(FIG2A_DIR),
                "--binary", str(BUILD / "agent"),
                "--runtime-overlay", str(BUILD),
            ],
            cwd=ROOT, check=True,
        )
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/analyze/prefill_shared_16_48.py"), "--dir", str(FIG2A_DIR)],
            cwd=ROOT, check=True,
        )
        plot_py = Path("/Users/hui/miniforge3/envs/data/bin/python")
        plot_exe = str(plot_py) if plot_py.is_file() else sys.executable
        subprocess.run(
            [plot_exe, "-u", str(ROOT / "scripts/plot/prefill_shared_16_64_ob1024.py"), "--dir", str(FIG2A_DIR)],
            cwd=ROOT, check=True,
        )
        print("=== Stage 2 complete ===", flush=True)
    else:
        print("=== Stage 2 already complete; skipping ===", flush=True)

    state["stage"] = "stage3_table1_queue"
    state["stage2_completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_status(state)

    print("=== Stage 3: Table 1 7-arm 10-case queue (ToolSandbox -> DroidTask -> AC-Low) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(ROOT / "scripts/run/t6_ten_case_queue.py"),
            "--build", str(BUILD),
            "--route-costs", str(ROUTE_COSTS),
            "--cold-costs", str(COLD_COSTS),
            "--status-file", str(QUEUE_STATUS),
        ],
        cwd=ROOT, check=True,
    )

    state["status"] = "passed"
    state["stage"] = "complete"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_status(state)
    print("=== Stage 2 & Stage 3 pipeline complete ===", flush=True)


if __name__ == "__main__":
    main()
