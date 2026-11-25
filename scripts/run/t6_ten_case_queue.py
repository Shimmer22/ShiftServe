#!/usr/bin/env python3
"""串行执行 ToolSandbox(10) -> DroidTask(10) -> AC-Low(10) 全量 7 组 6 线程 Table 1 实验队列。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "results/mainline-eval/t6-final-ob1024"
DEFAULT_ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"
DEFAULT_COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
DEFAULT_STATUS = ROOT / "results/mainline-eval/t6-8b-table1-10case-ob1024-queue-status.json"

QUEUE = (
    (
        "toolsandbox_10",
        ROOT / "results/20261124-toolsandbox-10cases/fixtures.json",
        ROOT / "results/mainline-eval/t6-8b-table1-toolsandbox-10case-6t-ob1024",
    ),
    (
        "droidtask_10",
        ROOT / "results/20261124-droidtask-10cases/fixtures.json",
        ROOT / "results/mainline-eval/t6-8b-table1-droidtask-10case-6t-ob1024",
    ),
    (
        "aclow_10",
        ROOT / "results/20261124-aclow-10cases/fixtures.json",
        ROOT / "results/mainline-eval/t6-8b-table1-aclow-10case-6t-ob1024",
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
    return json.loads(status_file.read_text()).get("status") == "passed"


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

    state = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "build": str(build),
        "route_costs": str(route_costs),
        "cohorts": {},
    }
    for name, fixtures, output_dir in QUEUE:
        if not fixtures.is_file():
            raise FileNotFoundError(f"缺少冻结夹具: {fixtures}")
        if is_complete(output_dir):
            state["cohorts"][name] = {"status": "passed", "output": str(output_dir), "resumed_skip": True}
            save_status(status_file, state)
            print(f"=== 跳过已完成阶段 {name} ({output_dir}) ===", flush=True)
            continue

        state["current"] = name
        state["cohorts"][name] = {"status": "running", "output": str(output_dir), "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
        save_status(status_file, state)
        print(f"=== 开始阶段 {name}: {fixtures} -> {output_dir} ===", flush=True)
        cmd = [
            sys.executable, "-u", str(ROOT / "scripts/run/t6_mobile_gui_table1.py"),
            "--build", str(build),
            "--output", str(output_dir),
            "--fixtures", str(fixtures),
            "--route-costs", str(route_costs),
            "--cold-costs", str(cold_costs),
            "--opbatch", "1024",
            "--opqueue", "16",
            "--htp-big-cores", "1",
            "--cpu-threads", "6",
            "--max-quiet-c", "39.0",
            "--min-prime-khz", "2000000",
            "--min-mem-avail-kib", "1500000",
            "--cooldown-timeout-s", "360",
        ]
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except Exception as exc:
            state["status"] = "failed"
            state["cohorts"][name]["status"] = "failed"
            state["cohorts"][name]["error"] = str(exc)
            save_status(status_file, state)
            raise
        state["cohorts"][name]["status"] = "passed"
        state["cohorts"][name]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        save_status(status_file, state)

    state["status"] = "passed"
    state.pop("current", None)
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_status(status_file, state)
    print("=== 全部三套 10-case Table 1 串行实验已完成 ===", flush=True)


if __name__ == "__main__":
    main()
