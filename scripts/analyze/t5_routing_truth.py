#!/usr/bin/env python3
"""用 T5 三路配对真值生成 Fig. 7 切片与路由回放数据。"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from a2o.scheduling.m1_explorer import BayesianLinearComponent, M1Explorer
from scripts.collect.p_cost_truth_sweep import (
    PHASES, read, read_sample, sample_path, save, sha256, validate_truth,
)

PATHS = ("CC", "HC-Full", "HC-Compact")
POLICIES = ("Oracle", "Static-Threshold", "Offline-LUT", "Greedy-Online", "M1-LCB")
ROUTE_NAMES = dict(zip(("cc", "hc_full", "hc_compact"), PATHS))
DECODE_CONFIG = ROOT / "configs/qwen3-history-decode.json"
DECODE_CURVE = ROOT / "results/20261114-qwen3-history-decode/summary.json"
P_MIN = 32
SEED = 20261120


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_slice(directory):
    plan_path = directory / "plan.json"
    plan = read(plan_path)
    verify = plan["verify_kv_state"]
    if (read(directory / "status.json")["status"] != "passed"
            or plan["paths"] != list(PATHS) or plan["modes"] != ["shared"]
            or plan["reps"] < 3 or type(verify) is not bool):
        raise ValueError(f"T5 切片未完成或缺少三路验证配置: {directory}")
    h, d, k = (plan[key] for key in ("H", "D", "keep_tokens"))
    digest = sha256(plan_path)
    samples = {}
    for p in plan["p_points"]:
        for rep in range(plan["reps"]):
            path = sample_path(directory, "shared", p, rep)
            row = read_sample(path, digest, "shared", h, p, d, rep)
            gap = plan["metadata"]["gap_ms"]
            validate_truth(row, PATHS, plan["decode_ids"], gap, verify)
            truth = row["paired_truth"]
            compact = truth["HC-Compact"]
            if (not compact["collect_prior"] or row["hc_compact_kept_count"] != k
                    or row["hc_compact_evicted_count"] != h + p - k):
                raise ValueError(f"实际裁剪与冻结预算不符: {path}")
            if (truth["CC"]["handoff_ms"] != 0 or truth["CC"]["compact_ms"] != 0
                    or truth["HC-Full"]["compact_ms"] != 0
                    or min(truth[label]["handoff_ms"] for label in PATHS[1:]) <= 0):
                raise ValueError(f"三路交接或裁剪阶段不符: {path}")
            samples[h, p, rep] = row

    logs = []
    for health_path in sorted(directory.glob("*/health.json")):
        health = read(health_path)
        launch = read(health_path.with_name("manifest.json"))
        log_path = health_path.parent / "device/runtime.log"
        log = log_path.read_text(errors="replace")
        metadata = plan["metadata"]
        runtime_meta = [json.loads(line.removeprefix("A2O_META ")) for line in log.splitlines()
                        if line.startswith("A2O_META ")]
        if (len(runtime_meta) != 1
                or any(runtime_meta[0].get(key) != value for key, value in metadata.items())
                or any(health.get(key) != value for key, value in metadata.items())
                or launch["runtime_overlay"] != plan["build"]["artifacts"]
                or launch["hexagon_sha256"] != plan["build"]["hexagon_sha256"]
                or health["context_size"] != plan["context_size"]
                or health["htp_kv_offload"] is not False
                or health["verify_kv_state"] is not verify
                or (not verify and health.get("paired_truth_version", 1) < 2)
                or health["kv_keep_tokens"] != k
                or not re.search(r"llama_prepare_model_devices: using device HTP0", log)
                or not re.search(r"layer\s+\d+ assigned to device HTP0", log)
                or "CPU-only scheduler, shared packed weights" not in log):
            raise ValueError(f"服务日志与 CPU/HTP 或封版配置不符: {log_path}")
        readbacks = re.findall(r"^TRANSFER verified_bytes=([1-9]\d*)$", log, re.MULTILINE)
        graphs = len(re.findall(r"^sched_reserve:\s+HTP0 compute buffer size =", log, re.MULTILINE))
        logs.append(dict(path=str(log_path.relative_to(ROOT)), sha256=sha256(log_path),
                         verified_transfers=len(readbacks), htp_graph_reservations=graphs,
                         cpu_threads=health["cpu_threads"],
                         htp_host_threads=health["htp_host_threads"]))
    checks = ["htp_graph_reservations"]
    if verify:
        checks.append("verified_transfers")
    for check in checks:
        if sum(log[check] for log in logs) < 2 * len(samples):
            raise ValueError(f"{check} 少于每个样本的两条 HTP 路径: {directory}")
    return plan, digest, samples, logs


def costs(row):
    return {path: row["paired_truth"][path]["post_return_ms"] for path in PATHS}


def curve_decode_ms(term, active, tokens):
    mean_tokens = active + (tokens - 1) / 2
    return tokens * (term["intercept_ms"] + term["slope_ms_per_token"] * mean_tokens)


def slices(samples, k, d, decode_term):
    groups = defaultdict(list)
    for (h, p, _), row in samples.items():
        groups[h, p].append(row["paired_truth"])
    measured, projected = [], []
    for (h, p), truths in sorted(groups.items()):
        medians = {}
        row = dict(H=h, P=p, D=d, K=k, pairs=len(truths))
        for path in PATHS:
            key = path.lower().replace("-", "_")
            for phase in (*PHASES, "post_return_ms"):
                row[f"{key}_{phase}"] = statistics.median(t[path][phase] for t in truths)
            medians[path] = row[f"{key}_post_return_ms"]
            row[f"{key}_wins"] = sum(min(PATHS, key=lambda label: t[label]["post_return_ms"]) == path
                                   for t in truths)
        ranked = sorted(PATHS, key=medians.get)
        measured.append(dict(row, winner=ranked[0],
                             margin_ms=medians[ranked[1]] - medians[ranked[0]]))
        for target_d in (16, 64, 96):
            values = {}
            for path in PATHS:
                active = k if path == "HC-Compact" else h + p
                key = path.lower().replace("-", "_")
                scale = curve_decode_ms(decode_term, active, target_d) / curve_decode_ms(decode_term, active, d)
                values[path] = medians[path] + row[f"{key}_decode_ms"] * (scale - 1)
            projected.append(dict(H=h, P=p, D=target_d, anchor_D=d, K=k,
                                  winner=min(values, key=values.get),
                                  **{f"{path.lower().replace('-', '_')}_model_ms": v
                                     for path, v in values.items()}))
    return measured, projected


def observe(router, h, p, d, k, path, row):
    phases = row["paired_truth"][path]
    router.observe(h=h, p=p, route="cpu" if path == "CC" else "htp",
                   h_active=k if path == "HC-Compact" else h + p,
                   restore_ms=phases["restore_ms"], prepare_ms=phases["prepare_ms"],
                   prefill_ms=phases["prefill_ms"], handoff_ms=phases["handoff_ms"],
                   decode_ms=phases["decode_ms"], predicted_n=d,
                   collect_prior=phases.get("collect_prior", False))


def replay(samples, k, d, decode_term, min_evict):
    training = {(h, p): row for (h, p, rep), row in samples.items() if rep == 0}
    histories = sorted({h for h, _ in training})
    points = sorted({p for _, p in training})
    bootstrap = [(histories[0], points[1]), (histories[-1], points[len(points) // 2])]
    lut = {cell: min(PATHS, key=costs(row).get) for cell, row in training.items()}
    prior_taxes, evict_costs = [], []
    for cell in bootstrap:
        truth = training[cell]["paired_truth"]
        prior_taxes.append(truth["HC-Compact"]["prefill_ms"] - truth["HC-Full"]["prefill_ms"])
        evict_costs.append(truth["HC-Compact"]["compact_ms"])
    kv = dict(quality_k_min=k, quality_keep_ratio_min=0.25, align_pad=256,
              min_evict_savings_tokens=min_evict, min_gain_ms=100.0, candidates=[k],
              decode_per_tok=decode_term,
              htp_prior_tax=dict(intercept_ms=max(0.0, statistics.median(prior_taxes))),
              evict=dict(intercept_ms=statistics.median(evict_costs)))

    routers = {"Greedy-Online": M1Explorer(exploration_enabled=False), "M1-LCB": M1Explorer()}
    for router in routers.values():
        router.decode_per_tok = BayesianLinearComponent(
            dim=2, prior_mean=[decode_term["intercept_ms"], decode_term["slope_ms_per_token"]],
            prior_var=[2500.0, 0.0004], obs_noise_std=10.0,
        )
        router.bootstrap_probes(dict(H=h, P=p, **training[h, p]) for h, p in bootstrap)

    stream = sorted((h, p, rep) for h, p, rep in samples if rep > 0)
    random.Random(SEED).shuffle(stream)
    steps = []
    total = Counter()
    choices = defaultdict(Counter)
    for step, (h, p, rep) in enumerate(stream, 1):
        row = samples[h, p, rep]
        truth = costs(row)
        selected = {
            "Oracle": min(PATHS, key=truth.get),
            "Static-Threshold": "CC" if p < P_MIN else "HC-Full",
            "Offline-LUT": lut[h, p],
        }
        reasons = {}
        for name, router in routers.items():
            decision = router.select_joint_route(h, p, d, kv, compact_keep_tokens=k)
            selected[name] = ROUTE_NAMES[decision.path]
            reasons[name] = decision.reason
            observe(router, h, p, d, k, selected[name], row)
        for name, path in selected.items():
            total[name] += truth[path]
            choices[name][path] += 1
        for name, path in selected.items():
            steps.append(dict(step=step, H=h, P=p, D=d, rep=rep, policy=name, path=path,
                              reason=reasons.get(name, ""), matched_oracle=path == selected["Oracle"],
                              cost_ms=truth[path], oracle_ms=truth[selected["Oracle"]],
                              cumulative_cost_ms=total[name], cumulative_ratio=total["Oracle"] / total[name]))
    summary = [dict(policy=name, samples=len(stream), cumulative_cost_ms=total[name],
                    oracle_ratio=total["Oracle"] / total[name], regret_ms=total[name] - total["Oracle"],
                    **{path: choices[name][path] for path in PATHS}) for name in POLICIES]
    return steps, summary, bootstrap, kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("slices", type=Path, nargs="+")
    parser.add_argument("--supplement", type=Path, help="追加短历史切片，仅用于 Fig. 7a")
    args = parser.parse_args()
    samples, plans = {}, []
    reference = None
    for path in args.slices:
        plan, digest, rows, logs = load_slice(path.resolve())
        common = {key: value for key, value in plan.items() if key != "H"}
        if (reference is not None and common != reference) or (samples.keys() & rows.keys()):
            raise ValueError(f"切片设置不一致或 H/P/rep 重复: {path}")
        reference = common
        samples.update(rows)
        plans.append(dict(H=plan["H"], plan_sha256=digest, paired_samples=len(rows), logs=logs))
    d, k = reference["D"], reference["keep_tokens"]
    curve = read(DECODE_CURVE)
    if not curve["complete"] or curve["model"] != reference["model"]["model_id"]:
        raise ValueError("历史 Decode 曲线未完成或模型不一致")
    fit = curve["descriptive_fits"]["cpu"]["linear_H"]
    term = dict(intercept_ms=fit["intercept"], slope_ms_per_token=fit["slope"])
    boundary_samples = samples.copy()
    if args.supplement:
        plan, digest, rows, logs = load_slice(args.supplement.resolve())
        comparable = {key: value for key, value in plan.items() if key not in ("H", "p_points")}
        if comparable != {key: value for key, value in reference.items() if key != "p_points"}:
            raise ValueError("补测切片与主网格的构建、夹具或计时设置不一致")
        if any(h == plan["H"] for h, _, _ in samples):
            raise ValueError("补测必须使用独立历史长度")
        boundary_samples.update(rows)
        plans.append(dict(H=plan["H"], plan_sha256=digest, paired_samples=len(rows), logs=logs,
                          boundary_only=True))
    measured, projected = slices(boundary_samples, k, d, term)
    steps, strategies, bootstrap, kv = replay(samples, k, d, term, reference["metadata"]["kv_min_evict_tokens"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fig7a-truth.csv", measured)
    write_csv(args.output / "fig7a-decode-projections.csv", projected)
    write_csv(args.output / "fig7b-steps.csv", steps)
    write_csv(args.output / "fig7b-ratios.csv", strategies)
    winners = Counter(row["winner"] for row in measured)
    stable = Counter(row["winner"] for row in measured
                     if row[f"{row['winner'].lower().replace('-', '_')}_wins"] == row["pairs"])
    save(args.output / "summary.json", dict(
        slices=plans, build=reference["build"], model=reference["model"], D=d, K=k,
        paired_samples=len(boundary_samples), grid_samples=len(samples), shuffle_seed=SEED,
        verify_kv_state=reference["verify_kv_state"],
        paired_truth_version=next(iter(samples.values()))["paired_truth"]["version"],
        training_rep=0, bootstrap_cells=bootstrap,
        static_p_min=P_MIN, m1_routing_p_min=M1Explorer().routing_p_min, kv_model=kv, winners=dict(winners), unanimous_winners=dict(stable),
        all_paths_have_unanimous_win=all(stable[path] for path in PATHS),
        decode_config_sha256=sha256(DECODE_CONFIG), decode_curve_sha256=sha256(DECODE_CURVE),
        analyzer_sha256=sha256(Path(__file__)),
        router_sha256=sha256(ROOT / "src/a2o/scheduling/m1_explorer.py"),
        ratios=strategies,
    ))
    for row in strategies:
        print(f"{row['policy']:<16} {row['oracle_ratio']:.2%}  {row['cumulative_cost_ms'] / 1000:.2f} s")


if __name__ == "__main__":
    main()
