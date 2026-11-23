#!/usr/bin/env python3
"""运行 T4 主机自由生成评测，逐阶段生成 Table 3 数据。"""

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from host_kv_accuracy_bfcl_campaign import start_bfcl_agent
from host_kv_accuracy_queue import TS_CONTAINER, TS_PORTS, TS_PYTHON, TS_SCRIPT, wait_for_wins_gpu_idle
from host_kv_accuracy_tau2_campaign import (
    ROOT, WINS_HOST, add_run_options, agent_identity, install_run_plan,
    run_metadata, same_run, stop_remote_pid, stream_worker, sync_workspace,
)
from t4_quality_table import write_table


CONDITIONS = ("full_kv", "m2_prior_mid", "m2_prior_high")
BENCHMARKS = {"bfcl": "bfcl", "toolsandbox": "toolsandbox", "tau2": "tau2-ratio"}
BUILD_ID = "20261120-t3-final-host"
BUILD_SHA256 = "570f925677c73c593cf20a8f04a6256752046187af6f7a4b0a1b67f1b9e2e77c"
OUTPUT = ROOT / "results/mainline-eval/t4-stride4"


def dump(path, data):
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    staging.replace(path)


def validate_config(bench, cfg):
    thinking_key = "agent_enable_thinking" if bench == "tau2" else "enable_thinking"
    if cfg.get(thinking_key) is not True:
        raise ValueError(f"T4 要求开启 thinking {cfg['name']}")
    conditions = {row["name"]: row for row in cfg["conditions"]}
    full = conditions["full_kv"]
    if full["kv_reduce"] or full["compact_anchor"]:
        raise ValueError(f"Full-KV 必须关闭裁剪与裁剪锚点 {cfg['name']}")
    for name, keep_ratio in zip(CONDITIONS[1:], (0.5, 0.75)):
        cond = conditions[name]
        if (cond["kv_reduce"], cond["compact_anchor"], cond["kv_policy"], cond["kv_keep_ratio"]) != (
            1, 1, "prefill_prior", keep_ratio
        ):
            raise ValueError(f"T4 KV 条件与计划不匹配 {cfg['name']}/{name}")


def run_toolsandbox(cfg, out, runtime, scenarios):
    protocol = "toolsandbox-t4-v1"
    plans = {name: run_metadata(protocol, cfg, name, 0, scenarios, runtime) for name in CONDITIONS}
    out.mkdir(parents=True, exist_ok=True)
    install_run_plan(out, plans)
    sync_workspace()
    for cond in cfg["conditions"]:
        name = cond["name"]
        if name not in plans:
            continue
        local = out / "conditions" / name
        local.mkdir(parents=True, exist_ok=True)
        plan = plans[name]
        completed = local / "worker-summary.json"
        if completed.exists():
            summary = json.loads(completed.read_text())
            if not same_run(summary, plan) or summary["selected_task_ids"] != scenarios:
                raise ValueError(f"ToolSandbox 续跑身份或题集不匹配 {completed}")
            continue
        # 新尝试使用独立目录，保留先前的部分轨迹与失败日志。
        attempt = uuid.uuid4().hex[:12]
        attempt_dir = local / attempt
        attempt_dir.mkdir()
        remote = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/{out.name}/{attempt}"
        remote_dir = f"{remote}/conditions/{name}"
        container_dir = remote_dir.replace("/home/wins/ai-workspace", "/workspace", 1)
        port = TS_PORTS[0]
        pid = ""
        try:
            pid = start_bfcl_agent(cond, cfg, port, remote, runtime, recent_tokens=64, tail_tokens=16)
            run_cfg = {
                **cfg,
                "endpoint": f"http://127.0.0.1:{port}",
                "model": Path(cfg["model_path"]).name,
                "backend": "CPU",
                "kv_policy": cond["kv_policy"] if cond["kv_reduce"] else "none",
                "concurrency": 1,
                "collector_lock": f"{container_dir}/collector.lock",
                "scenarios": scenarios,
                "a2o_service": True,
                "run_metadata": plan,
            }
            subprocess.run(["ssh", WINS_HOST, f"cat > {shlex.quote(remote_dir + '/run-cfg.json')}"],
                           input=json.dumps(run_cfg), text=True, check=True)
            stream_worker(shlex.join(["docker", "exec", TS_CONTAINER, TS_PYTHON, "-u", TS_SCRIPT,
                                      "--config", container_dir + "/run-cfg.json",
                                      "--output", container_dir + "/run"]), attempt_dir / "worker.log")
        finally:
            stop_remote_pid(pid)
            subprocess.run(["rsync", "-az", f"{WINS_HOST}:{remote_dir}/", str(attempt_dir) + "/"],
                           check=bool(pid))
        run = attempt_dir / "run"
        summary = json.loads((run / "summary.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        completion = json.loads((run / "completion.json").read_text())
        selected = [case["name"] for case in summary["results"]]
        if (manifest["config"] != run_cfg or selected != scenarios or
                completion != {"completed": True, "groups": len(scenarios)}):
            raise ValueError(f"ToolSandbox 结果身份或题集不匹配 {run}")
        dump(completed, {**plan, **summary, "selected_task_ids": selected, "attempt": attempt})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--model", choices=("8b", "4b"), action="append")
    add_run_options(parser)
    parser.set_defaults(
        agent_binary="/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261120-t3-final-host-build/agent",
        agent_build_id=BUILD_ID, limit_tasks=0,
    )
    args = parser.parse_args()
    if args.task_id or (args.kv_layer_stride, args.kv_min_evict_tokens) != (4, 256):
        parser.error("T4 固定 stride=4、min_evict=256，题集由评测配置指定")
    if args.agent_build_id != BUILD_ID:
        parser.error(f"T4 固定使用 {BUILD_ID}")
    out = (args.output or (OUTPUT.with_name(OUTPUT.name + "-smoke") if args.smoke else OUTPUT)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "t4.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime = agent_identity(args)
    if runtime["binary_sha256"] != BUILD_SHA256:
        raise ValueError("T4 二进制与 T3 终版 SHA-256 不一致")
    checked_models = set()
    failures = []
    for model in args.model or ("8b", "4b"):
        for bench in args.benchmark or BENCHMARKS:
            suffix = "-8b" if model == "8b" else ""
            path = ROOT / f"configs/host-kv-accuracy-{BENCHMARKS[bench]}{suffix}.json"
            cfg = json.loads(path.read_text())
            validate_config(bench, cfg)
            model_asset = cfg["model_path"], cfg["model_sha256"]
            if model_asset not in checked_models:
                model_sha = subprocess.check_output(
                    ["ssh", "-n", WINS_HOST, shlex.join(["sha256sum", "--", cfg["model_path"]])],
                    text=True, timeout=120,
                ).split()[0]
                if model_sha != cfg["model_sha256"]:
                    raise ValueError(f"模型 SHA-256 不匹配 {path}")
                checked_models.add(model_asset)
            stage = out / f"{out.name}-{bench}-{model}"
            stage.mkdir(exist_ok=True)
            snapshot = stage / "input-config.json"
            if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
                raise ValueError(f"续跑配置已变更 {snapshot}")
            dump(snapshot, cfg)
            print(f"[t4] {bench} {model} → {stage}", flush=True)
            try:
                if not args.smoke:
                    wait_for_wins_gpu_idle(poll_s=15, required_idle=1)
                if bench == "toolsandbox":
                    scenarios = cfg["scenarios"][:1] if args.smoke else cfg["scenarios"]
                    run_toolsandbox(cfg, stage, runtime, scenarios)
                else:
                    campaign = ROOT / f"scripts/collect/host_kv_accuracy_{bench}_campaign.py"
                    cmd = [sys.executable, "-u", str(campaign), "--config", str(path), "--output", str(stage),
                           "--agent-binary", runtime["binary"], "--agent-build-id", runtime["build_id"],
                           "--kv-layer-stride", "4", "--kv-min-evict-tokens", "256",
                           "--max-workers", "2" if bench == "bfcl" else "1",
                           "--conditions", *CONDITIONS]
                    if args.smoke:
                        cmd += ["--task-id", "multi_turn_long_context_94" if bench == "bfcl" else "51"]
                    subprocess.run(cmd, check=True)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                failures.append(f"{bench}/{model}: {exc}")
                print(f"[t4] 阶段失败 {failures[-1]}", flush=True)
            write_table(out, CONDITIONS, smoke=args.smoke)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
