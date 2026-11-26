#!/usr/bin/env python3
"""串行执行 ToolSandbox(10) -> DroidTask(10) -> AC-Low(10) 3 组配对补跑（Vanilla-Static-HC / Shared-Static-HC / ShiftServe）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "results/mainline-eval/t6-final-ob1024-clean"
DEFAULT_ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"
DEFAULT_COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
DEFAULT_STATUS = ROOT / "results/mainline-eval/t6-8b-static-hc-clean-3arm-queue-status.json"
ARMS = ("dual_static_hc", "shared_static_hc", "shiftserve_joint")

QUEUE = (
    (
        "toolsandbox_10",
        ROOT / "results/20261124-toolsandbox-10cases/fixtures.json",
        ROOT / "results/mainline-eval/t6-8b-table1-toolsandbox-10case-6t-ob1024-clean3",
    ),
    (
        "droidtask_10",
        ROOT / "results/20261124-droidtask-10cases/fixtures.json",
        ROOT / "results/mainline-eval/t6-8b-table1-droidtask-10case-6t-ob1024-clean3",
    ),
    (
        "aclow_10",
        ROOT / "results/20261124-aclow-10cases/fixtures.json",
        ROOT / "results/mainline-eval/t6-8b-table1-aclow-10case-6t-ob1024-clean3",
    ),
)


def save_status(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def is_complete(output_dir: Path) -> bool:
    status_file = output_dir / "status.json"
    if not status_file.is_file():
        return False
    return json.loads(status_file.read_text()).get("status") in ("passed", "passed_with_cell_failures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--route-costs", type=Path, default=DEFAULT_ROUTE_COSTS)
    parser.add_argument("--cold-costs", type=Path, default=DEFAULT_COLD_COSTS)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()

    build = args.build.resolve()
    route_costs = args.route_costs.resolve()
    cold_costs = args.cold_costs.resolve()
    status_file = args.status_file.resolve()
    status_file.parent.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "build": str(build),
        "route_costs": str(route_costs),
        "cold_costs": str(cold_costs),
        "arms": list(ARMS),
        "cohorts": {},
    }
    for name, fixtures, output_dir in QUEUE:
        if not fixtures.is_file():
            raise FileNotFoundError(f"缺少冻结夹具: {fixtures}")
        if is_complete(output_dir):
            cohort_status = json.loads((output_dir / "status.json").read_text())
            state["cohorts"][name] = {
                "status": cohort_status.get("status"),
                "output": str(output_dir),
                "failed_cells": cohort_status.get("failed_cells", 0),
                "resumed_skip": True,
            }
            save_status(status_file, state)
            print(f"=== 跳过已完成阶段 {name} ({output_dir}) ===", flush=True)
            continue

        state["current"] = name
        state["cohorts"][name] = {
            "status": "running",
            "output": str(output_dir),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        save_status(status_file, state)
        print(f"=== 开始阶段 {name}: {fixtures} -> {output_dir} ===", flush=True)
        cmd = [
            sys.executable, "-u", str(ROOT / "scripts/run/t6_mobile_gui_table1.py"),
            "--build", str(build),
            "--output", str(output_dir),
            "--fixtures", str(fixtures),
            "--route-costs", str(route_costs),
            "--cold-costs", str(cold_costs),
            "--arms", *ARMS,
            "--opbatch", "1024",
            "--opqueue", "16",
            "--htp-big-cores", "1",
            "--cpu-threads", "6",
            "--max-quiet-c", "39.0",
            "--min-prime-khz", "2000000",
            "--min-mem-avail-kib", "0",
            "--cooldown-timeout-s", "360",
            "--skip-gate",
            "--continue-on-cell-error",
        ]
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except Exception as exc:
            state["status"] = "failed"
            state["cohorts"][name]["status"] = "failed"
            state["cohorts"][name]["error"] = str(exc)
            save_status(status_file, state)
            raise
        cohort_status = json.loads((output_dir / "status.json").read_text())
        state["cohorts"][name].update({
            "status": cohort_status.get("status", "passed"),
            "formal_cells": cohort_status.get("formal_cells"),
            "failed_cells": cohort_status.get("failed_cells", 0),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        })
        save_status(status_file, state)
        table_csv = output_dir / "table1.csv"
        if table_csv.is_file():
            print(f"=== {name} table1.csv ===\n{table_csv.read_text()}", flush=True)

    has_cell_failures = any(c.get("failed_cells", 0) > 0 for c in state["cohorts"].values())
    state["status"] = "passed_with_cell_failures" if has_cell_failures else "passed"
    state.pop("current", None)
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_status(status_file, state)
    print("=== 全部三套 10-case 3-arm 补跑实验已完成 ===", flush=True)


if __name__ == "__main__":
    main()
