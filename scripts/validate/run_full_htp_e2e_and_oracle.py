#!/usr/bin/env python3
"""Run Full-HTP E2E on tau2-bench 9 tasks and compute live E2E Oracle summary
combining Full-CPU, Full-HTP, and M1-Explorer (+M2).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_e2e_spec = importlib.util.spec_from_file_location(
    "collect_tau2_e2e", ROOT / "scripts/collect/tau2_adaptive_kv_e2e.py"
)
_e2e_mod = importlib.util.module_from_spec(_e2e_spec)
assert _e2e_spec and _e2e_spec.loader
_e2e_spec.loader.exec_module(_e2e_mod)
CONTAINER_PROJECT = _e2e_mod.CONTAINER_PROJECT
WINS_HOST = _e2e_mod.WINS_HOST
WINS_PROJECT = _e2e_mod.WINS_PROJECT
ensure_wins_workspace = _e2e_mod.ensure_wins_workspace
open_phone_tunnel = _e2e_mod.open_phone_tunnel
route_phone = _e2e_mod.route_phone
sample_phone_memory = _e2e_mod.sample_phone_memory

DEFAULT_CFG = ROOT / "configs/tau2-qwen3-4b-joint-v2-e2e.json"
DEFAULT_CAMPAIGN_DIR = ROOT / "results/20261117-m1-explorer-large-campaign"
CPU_BASE_TASKS = ROOT / "results/20261116-tau2-qwen3-4b-joint-v2-e2e/conditions/cpu_baseline/tasks"
CPU_KV_TASKS = ROOT / "results/20261116-tau2-qwen3-4b-joint-v2-e2e/conditions/cpu_adaptive_compact/tasks"


def run_full_htp(cfg: dict, campaign_dir: Path) -> Path:
    htp_dir = campaign_dir / "e2e-full-htp"
    htp_dir.mkdir(parents=True, exist_ok=True)
    svc_dir = htp_dir / "service-htp"
    svc_log = htp_dir / "service-htp.log"
    cond_name = "full_htp"
    # Note: phone_service.cpp requires single_resident=false when condition=="htp"
    cond = {
        "name": cond_name,
        "label": "Full HTP (no CPU fallback, kv=0)",
        "condition": "htp",
        "single_resident": False,
        "cpu_repack": 0,
        "kv_reduce": 0,
        "kv_policy": "prefill_prior",
        "compact_anchor": 0,
        "use_kv_costs": False,
        "use_route_costs": False,
    }

    ensure_wins_workspace()
    step_cfg = dict(cfg)
    step_cfg["conditions"] = [cond]
    step_cfg_name = "m1_explorer_full_htp.json"
    step_cfg_local = ROOT / "configs" / step_cfg_name
    step_cfg_local.write_text(json.dumps(step_cfg, indent=2))
    subprocess.run(
        ["rsync", "-az", str(step_cfg_local), f"{WINS_HOST}:{WINS_PROJECT}/configs/{step_cfg_name}"],
        check=True,
    )

    route_phone("start", cfg, cond, svc_dir, svc_log)
    tunnel = open_phone_tunnel()
    # Wait until the reverse tunnel is genuinely reachable inside wins container
    import time
    for _ in range(30):
        chk = subprocess.run(
            ["ssh", WINS_HOST, "docker exec a2o-bfcl-lc200 curl -fsS http://127.0.0.1:18780/health >/dev/null 2>&1"],
            capture_output=True,
        )
        if chk.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Reverse tunnel http://127.0.0.1:18780/health did not become ready inside container")
    remote_out = f"/workspace/outputs/a2o-gap-profiling/{campaign_dir.name}/e2e-full-htp/{cond_name}"
    host_out = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/{campaign_dir.name}/e2e-full-htp/{cond_name}"
    local_cond_dir = htp_dir / cond_name
    local_cond_dir.mkdir(parents=True, exist_ok=True)
    try:
        mem_before = sample_phone_memory(svc_dir)
        q = shlex.quote
        worker_cmd = (
            f"KEY=$(grep '^NINFER_API_KEY=' /home/wins/qwen38-ninfer/.env | cut -d= -f2-) && "
            f"docker exec -e NINFER_API_KEY=\"$KEY\" -e PYTHONPATH=/workspace/references/tau2-bench-v0.1.0-source/src "
            f"a2o-bfcl-lc200 /workspace/private/a2o-tau2/env/bin/python3 -u "
            f"{q(CONTAINER_PROJECT + '/scripts/collect/tau2_e2e_worker.py')} "
            f"--config {q(CONTAINER_PROJECT + '/configs/' + step_cfg_name)} "
            f"--condition-name {q(cond_name)} "
            f"--output {q(remote_out)} "
            f"--agent-base http://127.0.0.1:18780 "
            f"--user-base {q(cfg['user_base'])} "
            f"--user-model {q(cfg['user_model'])} "
            f"--user-api-key-env NINFER_API_KEY"
        )
        subprocess.run(["ssh", WINS_HOST, worker_cmd], check=True)
        subprocess.run(["rsync", "-az", f"{WINS_HOST}:{host_out}/", str(local_cond_dir) + "/"], check=True)
        mem_after = sample_phone_memory(svc_dir)
        (local_cond_dir / "phone-memory.json").write_text(json.dumps({"before": mem_before, "after": mem_after}, indent=2))
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except Exception:
            tunnel.kill()
        try:
            route_phone("stop", cfg, cond, svc_dir, svc_log)
        except Exception as exc:
            print(f"Full HTP stop warning: {exc}", flush=True)
    return local_cond_dir / "tasks"


def load_task_metrics(tdir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for p in sorted(tdir.glob("*.json")):
        if ".failed-" in p.name:
            continue
        d = json.loads(p.read_text())
        dt = (d["domain"], str(d["task_id"]))
        calls = [c for c in d.get("calls", []) if c.get("is_agent")]
        warm = [c for c in calls if c["timings"].get("H_anchor", 0) > 0]
        svc_s = sum(c["timings"].get("total_ms", 0.0) for c in calls) / 1000.0
        warm_s = statistics.mean(c["timings"]["total_ms"] / 1000.0 for c in warm) if warm else 0.0
        prm_s = statistics.mean(c["timings"]["prompt_ms"] / 1000.0 for c in warm) if warm else 0.0
        dec_ms = sum(c["timings"].get("predicted_ms", 0.0) for c in calls)
        dec_n = sum(c["timings"].get("predicted_n", 0) for c in calls)
        out[dt] = {
            "domain": d["domain"],
            "task_id": str(d["task_id"]),
            "complete": bool(d.get("complete")),
            "reward": float(d.get("reward") or 0.0),
            "turns": len(calls),
            "warm_turns": len(warm),
            "service_s": svc_s,
            "warm_turn_s": warm_s,
            "warm_prompt_s": prm_s,
            "decode_ms_per_tok": dec_ms / max(1, dec_n),
        }
    return out


def compute_oracle_summary(campaign_dir: Path, htp_tasks_dir: Path) -> dict:
    sources = {
        "full_cpu_baseline": load_task_metrics(CPU_BASE_TASKS),
        "full_cpu_compact": load_task_metrics(CPU_KV_TASKS),
        "full_htp": load_task_metrics(htp_tasks_dir),
        "m1_explorer_m2": load_task_metrics(campaign_dir / "e2e-live/m1_explorer_m2_compact/tasks"),
    }
    all_keys = sorted(set.intersection(*(set(m.keys()) for m in sources.values())))
    per_task = []
    oracle_warm_sum = 0.0
    oracle_svc_sum = 0.0
    for k in all_keys:
        row = {"domain": k[0], "task_id": k[1]}
        warm_cands = {}
        svc_cands = {}
        for cname, m in sources.items():
            row[cname] = m[k]
            warm_cands[cname] = m[k]["warm_turn_s"]
            svc_cands[cname] = m[k]["service_s"]
        best_warm_cond = min(warm_cands, key=warm_cands.get)
        best_svc_cond = min(svc_cands, key=svc_cands.get)
        row["oracle_best_warm_cond"] = best_warm_cond
        row["oracle_best_warm_s"] = warm_cands[best_warm_cond]
        row["oracle_best_svc_cond"] = best_svc_cond
        row["oracle_best_svc_s"] = svc_cands[best_svc_cond]
        oracle_warm_sum += warm_cands[best_warm_cond]
        oracle_svc_sum += svc_cands[best_svc_cond]
        per_task.append(row)

    totals = {}
    for cname, m in sources.items():
        vals = [m[k] for k in all_keys]
        totals[cname] = {
            "n_tasks": len(vals),
            "reward_sum": sum(x["reward"] for x in vals),
            "total_service_s": sum(x["service_s"] for x in vals),
            "mean_warm_turn_s": statistics.mean(x["warm_turn_s"] for x in vals) if vals else 0.0,
            "mean_warm_prompt_s": statistics.mean(x["warm_prompt_s"] for x in vals) if vals else 0.0,
            "mean_decode_ms_per_tok": statistics.mean(x["decode_ms_per_tok"] for x in vals) if vals else 0.0,
        }
    totals["oracle_per_task_best"] = {
        "n_tasks": len(all_keys),
        "total_service_s": oracle_svc_sum,
        "mean_warm_turn_s": oracle_warm_sum / max(1, len(all_keys)),
    }
    summary = {"totals": totals, "per_task": per_task}
    (campaign_dir / "e2e_oracle_comparison.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.skip_run:
        htp_tasks = args.campaign_dir / "e2e-full-htp/full_htp/tasks"
    else:
        htp_tasks = run_full_htp(cfg, args.campaign_dir)
    summary = compute_oracle_summary(args.campaign_dir, htp_tasks)
    print(json.dumps(summary["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
