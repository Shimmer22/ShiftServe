#!/usr/bin/env python3
"""Large-scale M1-Explorer validation campaign:
Stage 1: Multi-H (2048, 6144, 9216) x P ground-truth Oracle grid & from-zero convergence.
Stage 2: Live tau2-bench 9-task E2E starting from 4 tiny probes + M1-Explorer online updates + M2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.network_retry import retry_network
from a2o.scheduling.m1_explorer import M1Explorer
from a2o.workloads.toolsandbox import post

_e2e_spec = importlib.util.spec_from_file_location(
    "collect_tau2_e2e", ROOT / "scripts/collect/tau2_adaptive_kv_e2e.py"
)
_e2e_mod = importlib.util.module_from_spec(_e2e_spec)
assert _e2e_spec and _e2e_spec.loader
_e2e_spec.loader.exec_module(_e2e_mod)
CONTAINER_PROJECT = _e2e_mod.CONTAINER_PROJECT
PHONE_ENDPOINT = _e2e_mod.PHONE_ENDPOINT
WINS_HOST = _e2e_mod.WINS_HOST
WINS_PROJECT = _e2e_mod.WINS_PROJECT
ensure_wins_workspace = _e2e_mod.ensure_wins_workspace
open_phone_tunnel = _e2e_mod.open_phone_tunnel
route_phone = _e2e_mod.route_phone
sample_phone_memory = _e2e_mod.sample_phone_memory

DEFAULT_CFG = ROOT / "configs/tau2-qwen3-4b-joint-v2-e2e.json"
DEFAULT_OUT = ROOT / "results/20261117-m1-explorer-large-campaign"
TINY_PROBES_PATH = ROOT / "results/20261117-m1-explorer-bootstrap/tiny_probes.json"
H4096_TRUTH_PATH = ROOT / "results/20261117-m1-explorer-truth-h4096/truth.json"

MULTI_H_LIST = [2048, 6144, 9216]
MULTI_P_LIST = [24, 32, 36, 40, 48, 64, 80, 96, 128, 256]


def run_stage1_oracle_grid(cfg: dict, out_dir: Path, skip_measure: bool) -> dict:
    grid_dir = out_dir / "oracle-grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    raw_path = grid_dir / "raw_multi_h_truth.json"

    if skip_measure and raw_path.exists():
        rows = json.loads(raw_path.read_text())
    else:
        svc_dir = grid_dir / "service-route"
        svc_log = grid_dir / "service-route.log"
        cond = {
            "name": "oracle_grid",
            "condition": "route",
            "single_resident": False,
            "cpu_repack": 0,
            "kv_reduce": 0,
            "kv_policy": "prefill_prior",
            "compact_anchor": 0,
            "use_kv_costs": False,
            "use_route_costs": False,
        }
        route_phone("start", cfg, cond, svc_dir, svc_log)
        rows = []
        try:
            unit = post(PHONE_ENDPOINT, "/tokenize", {"content": "Agent telemetry context block for multi-H Oracle scan.\n"}, 60)["tokens"]
            gen = post(PHONE_ENDPOINT, "/tokenize", {"content": "OK."}, 60)["tokens"][:1]
            for h_len in MULTI_H_LIST:
                prefix = (unit * ((h_len // len(unit)) + 1))[:h_len]
                for p_len in MULTI_P_LIST:
                    inp = (unit * ((p_len // len(unit)) + 1))[:p_len]
                    body = {
                        "prefix_ids": prefix,
                        "input_ids": inp,
                        "generated_ids": gen,
                        "order": ["cpu", "route"],
                        "allow_partial_order": True,
                        "gap_ms": 2000.0,
                        "return_policy": "greedy",
                        "prediction": {
                            "validated": True,
                            "prepare_ms": 500.0,
                            "cpu_prefill_ms": 999999.0,
                            "cpu_restore_ms": 1000.0,
                            "htp_prefill_handoff_ms": 1.0,
                        },
                        "kv_reduce": False,
                    }
                    resp = retry_network(lambda: post(PHONE_ENDPOINT, "/a2o/gap-replay", body, 1800))
                    by_c = {r["condition"]: r for r in resp["results"]}
                    c_r, r_r = by_c["cpu"], by_c["route"]
                    if r_r["route"] != "htp":
                        raise RuntimeError(f"Expected htp route at H={h_len} P={p_len}, got {r_r['route']}")
                    row = {
                        "H": h_len,
                        "P": p_len,
                        "cpu_restore_ms": float(c_r["restore_ms"]),
                        "cpu_prefill_ms": float(c_r["prefill_ms"]),
                        "cpu_total_ms": float(c_r["restore_ms"] + c_r["prefill_ms"]),
                        "htp_prepare_ms": float(r_r["prepare_ms"]),
                        "htp_prefill_ms": float(r_r["prefill_ms"]),
                        "htp_to_cpu_ms": float(r_r["handoff_ms"]),
                        "htp_total_ms": float(r_r["prefill_ms"] + r_r["handoff_ms"]),
                    }
                    rows.append(row)
                    raw_path.write_text(json.dumps(rows, indent=2))
                    print(
                        f"  [Oracle H={h_len:4d} P={p_len:3d}] "
                        f"CPU={row['cpu_total_ms']:6.0f}ms (rst={row['cpu_restore_ms']:.0f}, prm={row['cpu_prefill_ms']:.0f}) | "
                        f"HTP={row['htp_total_ms']:6.0f}ms (prm={row['htp_prefill_ms']:.0f}, ho={row['htp_to_cpu_ms']:.0f})",
                        flush=True,
                    )
        finally:
            try:
                route_phone("stop", cfg, cond, svc_dir, svc_log)
            except Exception as exc:
                print(f"Stage 1 stop warning: {exc}", flush=True)

    # Combine with existing H=4096 truth if present
    all_truth = list(rows)
    if H4096_TRUTH_PATH.exists():
        all_truth.extend(json.loads(H4096_TRUTH_PATH.read_text()))

    tiny_probes = json.loads(TINY_PROBES_PATH.read_text())["probes"]
    static_tiny = M1Explorer(online_update_enabled=False, exploration_enabled=False)
    static_tiny.bootstrap_probes(tiny_probes)
    greedy = M1Explorer(beta=0.0, online_update_enabled=True, exploration_enabled=False)
    greedy.bootstrap_probes(tiny_probes)
    explorer = M1Explorer(beta=1.5, max_explore_penalty_ms=4000.0, online_update_enabled=True, exploration_enabled=True)
    explorer.bootstrap_probes(tiny_probes)

    rng = random.Random(20261117)
    stream = all_truth[:]
    rng.shuffle(stream)

    routers = {"static_tiny": static_tiny, "online_greedy": greedy, "m1_explorer": explorer}
    cum_oracle = 0.0
    cum_cost = {k: 0.0 for k in routers}
    curve = []
    for idx, pt in enumerate(stream, start=1):
        h, p = int(pt["H"]), int(pt["P"])
        cc, hc = float(pt["cpu_total_ms"]), float(pt["htp_total_ms"])
        best = min(cc, hc) if p >= 32 else cc
        cum_oracle += best
        row = {"step": idx, "H": h, "P": p}
        for name, r in routers.items():
            dec = r.select_route(h, p, htp_ready=True)
            cum_cost[name] += hc if dec.route == "htp" else cc
            row[f"{name}_closeness"] = cum_oracle / max(1.0, cum_cost[name])
            if dec.route == "htp":
                r.observe(h=h, p=p, route="htp", prefill_ms=float(pt["htp_prefill_ms"]), handoff_ms=float(pt["htp_to_cpu_ms"]))
            else:
                r.observe(h=h, p=p, route="cpu", restore_ms=float(pt["cpu_restore_ms"]), prefill_ms=float(pt["cpu_prefill_ms"]))
        curve.append(row)

    summary = {
        "n_truth_points": len(all_truth),
        "final_closeness": {k: curve[-1][f"{k}_closeness"] for k in routers},
        "curve": curve,
    }
    (grid_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Stage 1] Oracle summary saved: {summary['final_closeness']}", flush=True)
    return summary


def run_stage2_live_e2e(cfg: dict, out_dir: Path) -> dict:
    e2e_dir = out_dir / "e2e-live"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    svc_dir = e2e_dir / "service-route"
    svc_log = e2e_dir / "service-route.log"
    cond_name = "m1_explorer_m2_compact"
    cond = {
        "name": cond_name,
        "label": "M1-Explorer (from zero tiny probes) + M2 Adaptive Compact",
        "condition": "route",
        "single_resident": False,
        "cpu_repack": 0,
        "kv_reduce": 1,
        "kv_policy": "adaptive_cost",
        "compact_anchor": 1,
        "use_kv_costs": True,
        "use_route_costs": False,
    }

    tiny_probes = json.loads(TINY_PROBES_PATH.read_text())["probes"]
    explorer = M1Explorer(beta=1.5, max_explore_penalty_ms=4000.0, online_update_enabled=True, exploration_enabled=True)
    explorer.bootstrap_probes(tiny_probes)
    kv_model = json.loads((ROOT / cfg["kv_costs"]).read_text())

    ensure_wins_workspace()
    route_phone("start", cfg, cond, svc_dir, svc_log)
    tunnel = open_phone_tunnel()
    remote_out = f"/workspace/outputs/a2o-gap-profiling/{out_dir.name}/e2e-live/{cond_name}"
    host_out = f"/home/wins/ai-workspace/outputs/a2o-gap-profiling/{out_dir.name}/e2e-live/{cond_name}"
    local_cond_dir = e2e_dir / cond_name
    local_cond_dir.mkdir(parents=True, exist_ok=True)

    evolution = []
    try:
        # Install initial cold-start linearized cost model (from 4 tiny probes)
        init_model = explorer.export_linear_hp_v1(reference_h=5500)
        retry_network(lambda: post(PHONE_ENDPOINT, "/a2o/costs", {"model": init_model, "kv_model": kv_model}, 60))

        mem_before = sample_phone_memory(svc_dir)
        all_tasks = cfg["tasks"]
        for k in range(len(all_tasks)):
            step_cfg = dict(cfg)
            step_cfg["tasks"] = all_tasks[: k + 1]
            step_cfg["conditions"] = [cond]
            step_cfg_name = f"m1_explorer_step_{k:02d}.json"
            step_cfg_local = ROOT / "configs" / step_cfg_name
            step_cfg_local.write_text(json.dumps(step_cfg, indent=2))
            subprocess.run(
                ["rsync", "-az", str(step_cfg_local), f"{WINS_HOST}:{WINS_PROJECT}/configs/{step_cfg_name}"],
                check=True,
            )

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
            print(f"[Stage 2] Running task {k+1}/{len(all_tasks)}: {all_tasks[k]} ...", flush=True)
            subprocess.run(["ssh", WINS_HOST, worker_cmd], check=True)
            subprocess.run(
                ["rsync", "-az", f"{WINS_HOST}:{host_out}/", str(local_cond_dir) + "/"],
                check=True,
            )

            # Ingest newly completed task observations into M1-Explorer
            task_files = [
                p for p in sorted((local_cond_dir / "tasks").glob(f"{k:02d}-*.json"))
                if ".failed-" not in p.name
            ]
            ingested_turns = 0
            recent_hs = []
            for tf in task_files:
                td = json.loads(tf.read_text())
                for c in td.get("calls", []):
                    if not c.get("is_agent"):
                        continue
                    t = c["timings"]
                    h = int(t.get("H_anchor", 0))
                    p_ev = int(t.get("P_evaluated", 0))
                    if h <= 0 or p_ev <= 0:
                        continue
                    recent_hs.append(h)
                    explorer.observe(
                        h=h,
                        p=p_ev,
                        route=t["route"],
                        restore_ms=float(t.get("restore_ms", 0.0)),
                        prepare_ms=float(t.get("prepare_ms", 0.0)),
                        prefill_ms=float(t.get("prompt_ms", 0.0)),
                        handoff_ms=float(t.get("handoff_ms", 0.0)),
                    )
                    ingested_turns += 1

            ref_h = int(statistics.mean(recent_hs)) if recent_hs else 6000
            updated_model = explorer.export_linear_hp_v1(reference_h=ref_h)
            retry_network(lambda: post(PHONE_ENDPOINT, "/a2o/costs", {"model": updated_model, "kv_model": kv_model}, 60))
            mem_now = sample_phone_memory(svc_dir)
            evolution.append({
                "tasks_completed": k + 1,
                "task": all_tasks[k],
                "ingested_turns": ingested_turns,
                "reference_h": ref_h,
                "cpu_prefill_slope": updated_model["cpu_prefill"]["slope_ms_per_token"],
                "htp_prefill_slope": updated_model["htp_prefill"]["slope_ms_per_token"],
                "htp_to_cpu_slope": updated_model["htp_to_cpu"]["slope_ms_per_token"],
                "memory": mem_now,
            })
            print(
                f"  [M1-Explorer updated after task {k+1}] ref_H={ref_h} "
                f"cpu_slope={updated_model['cpu_prefill']['slope_ms_per_token']:.1f} "
                f"htp_slope={updated_model['htp_prefill']['slope_ms_per_token']:.1f} "
                f"RSS={mem_now.get('VmRSS')} Swap={mem_now.get('VmSwap')}",
                flush=True,
            )

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
            print(f"Stage 2 stop warning: {exc}", flush=True)

    summary = {"evolution": evolution, "final_model": explorer.export_linear_hp_v1(reference_h=8000)}
    (e2e_dir / "live_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-stage1-measure", action="store_true")
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--skip-stage2", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_stage1:
        run_stage1_oracle_grid(cfg, args.out_dir, args.skip_stage1_measure)
    if not args.skip_stage2:
        run_stage2_live_e2e(cfg, args.out_dir)


if __name__ == "__main__":
    main()
