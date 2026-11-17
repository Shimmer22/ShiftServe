#!/usr/bin/env python3
"""Measure H=4096 fine-grained CPU vs HTP ground-truth (P=24..128) and compare M1-Explorer from zero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.network_retry import retry_network
from a2o.scheduling.m1_explorer import M1Explorer
from a2o.workloads.toolsandbox import post

ENDPOINT = "http://10.29.131.50:18780"
DEFAULT_CFG = ROOT / "configs/tau2-qwen3-4b-joint-v2-e2e.json"
DEFAULT_OUT = ROOT / "results/20261117-m1-explorer-truth-h4096"
TINY_PROBES_PATH = ROOT / "results/20261117-m1-explorer-bootstrap/tiny_probes.json"
P_GRID = [24, 28, 31, 32, 33, 36, 40, 44, 48, 52, 56, 60, 63, 64, 65, 72, 80, 96, 128]


def route_service(action: str, cfg: dict, svc_dir: Path, log_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run/route_phone.py"),
        action,
        "--condition", "route",
        "--output", str(svc_dir),
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
            "--cpu-repack", str(cfg["cpu_repack"]),
            "--opbatch", str(cfg["opbatch"]),
            "--opqueue", str(cfg["opqueue"]),
            "--kv-offload", str(cfg["kv_offload"]),
            "--flash-attention", str(cfg["flash_attention"]),
            "--kv-reduce", "0",
            "--expected-decode", str(cfg["expected_decode_tokens"]),
            "--replay-cache",
        ]
    else:
        cmd.append("--logs-only")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def measure_truth(cfg: dict, out_dir: Path, h_len: int, repeats: int) -> list[dict]:
    svc_dir = out_dir / "service-route"
    svc_log = out_dir / "service-route.log"
    route_service("start", cfg, svc_dir, svc_log)
    rows = []
    try:
        unit = post(ENDPOINT, "/tokenize", {"content": "Agent telemetry context block for H=4096 scan.\n"}, 60)["tokens"]
        gen = post(ENDPOINT, "/tokenize", {"content": "OK."}, 60)["tokens"][:1]
        prefix = (unit * ((h_len // len(unit)) + 1))[:h_len]

        for rep in range(repeats):
            p_order = P_GRID[:] if rep % 2 == 0 else list(reversed(P_GRID))
            for p_len in p_order:
                inp = (unit * ((p_len // len(unit)) + 1))[:p_len]
                body = {
                    "prefix_ids": prefix,
                    "input_ids": inp,
                    "generated_ids": gen,
                    "order": ["cpu", "route"],
                    "allow_partial_order": True,
                    "gap_ms": 1500.0,
                    "return_policy": "greedy",
                    "prediction": {
                        "validated": True,
                        "prepare_ms": 400.0,
                        "cpu_prefill_ms": 999999.0,
                        "cpu_restore_ms": 1000.0,
                        "htp_prefill_handoff_ms": 1.0,
                    },
                    "kv_reduce": False,
                }
                resp = retry_network(lambda: post(ENDPOINT, "/a2o/gap-replay", body, 1200))
                by_c = {r["condition"]: r for r in resp["results"]}
                c_r, r_r = by_c["cpu"], by_c["route"]
                if r_r["route"] != "htp":
                    raise RuntimeError(f"Expected forced htp route at P={p_len}, got {r_r['route']} ({r_r['route_reason']})")
                rows.append({
                    "rep": rep,
                    "H": h_len,
                    "P": p_len,
                    "cpu_restore_ms": float(c_r["restore_ms"]),
                    "cpu_prefill_ms": float(c_r["prefill_ms"]),
                    "cpu_total_ms": float(c_r["restore_ms"] + c_r["prefill_ms"]),
                    "htp_prepare_ms": float(r_r["prepare_ms"]),
                    "htp_prefill_ms": float(r_r["prefill_ms"]),
                    "htp_to_cpu_ms": float(r_r["handoff_ms"]),
                    "htp_total_ms": float(r_r["prefill_ms"] + r_r["handoff_ms"]),
                })
                print(
                    f"  [rep={rep} H={h_len} P={p_len:3d}] "
                    f"CPU(rst={c_r['restore_ms']:.0f} prm={c_r['prefill_ms']:.0f} tot={c_r['restore_ms']+c_r['prefill_ms']:.0f}) | "
                    f"HTP(prm={r_r['prefill_ms']:.0f} ho={r_r['handoff_ms']:.0f} tot={r_r['prefill_ms']+r_r['handoff_ms']:.0f})",
                    flush=True,
                )
    finally:
        try:
            route_service("stop", cfg, svc_dir, svc_log)
        except Exception as exc:
            print(f"Service stop warning: {exc}", flush=True)
    return rows


def summarize_truth(raw_rows: list[dict]) -> list[dict]:
    by_p: dict[int, list[dict]] = {}
    for r in raw_rows:
        by_p.setdefault(int(r["P"]), []).append(r)
    out = []
    for p in sorted(by_p):
        grp = by_p[p]
        out.append({
            "H": grp[0]["H"],
            "P": p,
            "n": len(grp),
            "cpu_restore_ms": statistics.median(x["cpu_restore_ms"] for x in grp),
            "cpu_prefill_ms": statistics.median(x["cpu_prefill_ms"] for x in grp),
            "cpu_total_ms": statistics.median(x["cpu_total_ms"] for x in grp),
            "htp_prefill_ms": statistics.median(x["htp_prefill_ms"] for x in grp),
            "htp_to_cpu_ms": statistics.median(x["htp_to_cpu_ms"] for x in grp),
            "htp_total_ms": statistics.median(x["htp_total_ms"] for x in grp),
        })
    return out


def compare_from_zero(truth: list[dict], tiny_probes: list[dict]) -> dict:
    # Build routers starting purely from tiny universal probes (no H=4096 data!)
    static_tiny = M1Explorer(online_update_enabled=False, exploration_enabled=False)
    static_tiny.bootstrap_probes(tiny_probes)

    greedy = M1Explorer(beta=0.0, online_update_enabled=True, exploration_enabled=False)
    greedy.bootstrap_probes(tiny_probes)

    explorer = M1Explorer(beta=1.5, max_explore_penalty_ms=4000.0, online_update_enabled=True, exploration_enabled=True)
    explorer.bootstrap_probes(tiny_probes)

    def snapshot_curve(router: M1Explorer) -> list[dict]:
        pts = []
        for t in truth:
            st = router.predict_paths(t["H"], t["P"])
            pts.append({
                "P": t["P"],
                "cc_pred_ms": st["cc_mean_ms"],
                "hc_pred_ms": st["hc_mean_ms"],
            })
        return pts

    init_curve = snapshot_curve(explorer)

    # Stream H=4096 points in shuffled task order (3 passes over the P grid to mimic multi-task stream)
    rng = random.Random(20261117)
    stream = truth[:] * 2
    rng.shuffle(stream)

    cum_oracle = 0.0
    cum_cost = {"static_tiny": 0.0, "online_greedy": 0.0, "m1_explorer": 0.0}
    step_curve = []
    routers = {"static_tiny": static_tiny, "online_greedy": greedy, "m1_explorer": explorer}

    for step_idx, pt in enumerate(stream, start=1):
        h, p = pt["H"], pt["P"]
        true_cc = pt["cpu_total_ms"]
        true_hc = pt["htp_total_ms"]
        best = min(true_cc, true_hc) if p >= 32 else true_cc
        cum_oracle += best

        step_row = {"step": step_idx, "P": p, "oracle_route": "htp" if (p >= 32 and true_hc < true_cc) else "cpu"}
        for name, r in routers.items():
            dec = r.select_route(h, p, htp_ready=True)
            chosen = true_hc if dec.route == "htp" else true_cc
            cum_cost[name] += chosen
            step_row[f"{name}_closeness"] = cum_oracle / max(1.0, cum_cost[name])
            step_row[f"{name}_route"] = dec.route
            if dec.route == "htp":
                r.observe(h=h, p=p, route="htp", prefill_ms=pt["htp_prefill_ms"], handoff_ms=pt["htp_to_cpu_ms"])
            else:
                r.observe(h=h, p=p, route="cpu", restore_ms=pt["cpu_restore_ms"], prefill_ms=pt["cpu_prefill_ms"])
        step_curve.append(step_row)

    final_curve = snapshot_curve(explorer)
    greedy_curve = snapshot_curve(greedy)
    return {
        "init_curve": init_curve,
        "greedy_curve": greedy_curve,
        "explorer_final_curve": final_curve,
        "step_curve": step_curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--h-len", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--skip-measure", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_truth.json"

    if args.skip_measure and raw_path.exists():
        raw_rows = json.loads(raw_path.read_text())
    else:
        print(f"Measuring H={args.h_len} ground-truth P={P_GRID} (repeats={args.repeats})...", flush=True)
        raw_rows = measure_truth(cfg, args.out_dir, args.h_len, args.repeats)
        raw_path.write_text(json.dumps(raw_rows, indent=2))

    truth = summarize_truth(raw_rows)
    (args.out_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    tiny_probes = json.loads(TINY_PROBES_PATH.read_text())["probes"]
    comp = compare_from_zero(truth, tiny_probes)
    (args.out_dir / "compare.json").write_text(json.dumps(comp, indent=2))

    print("\n=== H=4096 GROUND TRUTH vs M1-EXPLORER (from zero tiny probes) ===", flush=True)
    for t, ic, fc in zip(truth, comp["init_curve"], comp["explorer_final_curve"]):
        winner = "HTP" if t["htp_total_ms"] < t["cpu_total_ms"] else "CPU"
        print(
            f"  P={t['P']:3d} | Truth: CPU={t['cpu_total_ms']:6.0f}ms HTP={t['htp_total_ms']:6.0f}ms ({winner}) | "
            f"Init: CPU={ic['cc_pred_ms']:6.0f} HTP={ic['hc_pred_ms']:6.0f} | "
            f"Learned: CPU={fc['cc_pred_ms']:6.0f} HTP={fc['hc_pred_ms']:6.0f}"
        )
    last = comp["step_curve"][-1]
    print(
        f"\nFinal Closeness to Oracle: StaticTiny={last['static_tiny_closeness']*100:.2f}%, "
        f"Greedy={last['online_greedy_closeness']*100:.2f}%, "
        f"M1-Explorer={last['m1_explorer_closeness']*100:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
