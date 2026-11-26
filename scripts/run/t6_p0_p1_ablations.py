#!/usr/bin/env python3
"""Queued P0 (pre- vs post-handoff KV compaction) and P1 (rho, stride, 8B gap) ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.phone_preflight import require_attached_phone_preflight
from scripts.collect.tau2_e2e_worker import _agent_chat_resilient
from scripts.run.frozen_ablation import (
    COST_ARMS,
    CPU_REPLAY_ARMS,
    STATIC_HC_ARMS,
    digest,
    kv_cost_payload,
    read,
    request,
    save,
)
import scripts.run.t6_mobile_gui_table1 as mg
from scripts.run.t6_remaining_4arm_queue import pid_alive

DEFAULT_BUILD = ROOT / "results/mainline-eval/t6-p0p1-ablation-build"
DEFAULT_OUT = ROOT / "results/mainline-eval/t6-8b-p0-p1-ablations"
DEFAULT_STATUS = ROOT / "results/mainline-eval/t6-8b-p0-p1-ablations-status.json"
DEFAULT_WAIT_PID_FILE = ROOT / "results/mainline-eval/t6-8b-remaining-4arm-queue.pid"
DEFAULT_ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"
DEFAULT_COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
DEFAULT_PRIORS = ROOT / "results/mainline-eval/t6-8b-table1-revised/t5-kv-priors.json"
AC_FIXTURES = ROOT / "results/20261124-aclow-10cases/fixtures.json"
TS_FIXTURES = ROOT / "results/20261124-toolsandbox-10cases/fixtures.json"

P0_CASES = ("ac_ep_9464", "ac_ep_4451")
P1_AC_CASES = ("ac_ep_9464", "ac_ep_4451", "ac_ep_6119")
P1_GAP_CASES = (
    "find_current_city_low_battery_mode",
    "modify_contact_with_message_recency_alt",
    "ac_ep_2405",
    "ac_ep_6119",
)
P1_GAPS_MS = (0, 500, 1000)

ABLATION_ARM_SPECS = {
    "pre_handoff_htp": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.50,
        stride=4,
        force_path="hc_auto",
        post_handoff_kv_reduce=False,
    ),
    "post_handoff_cpu": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.50,
        stride=4,
        force_path="hc_auto",
        post_handoff_kv_reduce=True,
    ),
    "rho_0.50": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.50,
        stride=4,
        force_path="hc_auto",
        post_handoff_kv_reduce=False,
    ),
    "rho_0.75": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.75,
        stride=4,
        force_path="hc_auto",
        post_handoff_kv_reduce=False,
    ),
    "rho_1.00": dict(
        condition="route",
        shared=True,
        kv_reduce=0,
        kv_policy=None,
        kv_keep_ratio=-1.0,
        stride=4,
        force_path="hc_full",
        post_handoff_kv_reduce=False,
    ),
    "stride_1": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.50,
        stride=1,
        force_path="hc_auto",
        post_handoff_kv_reduce=False,
    ),
    "stride_4": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.50,
        stride=4,
        force_path="hc_auto",
        post_handoff_kv_reduce=False,
    ),
    "stride_8": dict(
        condition="route",
        shared=True,
        kv_reduce=1,
        kv_policy="prefill_prior",
        kv_keep_ratio=0.50,
        stride=8,
        force_path="hc_auto",
        post_handoff_kv_reduce=False,
    ),
}


def build_subset_fixtures(out_path: Path, case_ids: tuple[str, ...]) -> Path:
    pool = {}
    for src in (AC_FIXTURES, TS_FIXTURES):
        for case in read(src)["cases"]:
            pool[case["id"]] = case
    missing = [cid for cid in case_ids if cid not in pool]
    if missing:
        raise ValueError(f"missing fixture cases: {missing}")
    selected = [pool[cid] for cid in case_ids]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(out_path, {"cases": selected, "calibration_cases": []})
    return out_path


def ablation_arm_chat_overrides(arm: str, turn_index: int) -> dict:
    spec = ABLATION_ARM_SPECS.get(arm)
    if spec is None or turn_index == 0:
        return {}
    body = {"frozen_force_path": spec["force_path"]}
    if 0.0 < spec["kv_keep_ratio"] < 1.0:
        body["frozen_kv_keep_ratio"] = spec["kv_keep_ratio"]
    if spec["post_handoff_kv_reduce"]:
        body["frozen_post_handoff_kv_reduce"] = True
    return body


class AblationCampaign(mg.MobileCampaign):
    def service(self, action, arm, directory):
        spec = ABLATION_ARM_SPECS.get(arm)
        if spec is None:
            return super().service(action, arm, directory)
        condition = spec["condition"]
        shared = spec["shared"]
        command = [
            sys.executable,
            str(ROOT / "scripts/run/route_phone.py"),
            action,
            "--condition",
            condition,
            "--output",
            str(directory),
            "--logs-only",
            "--retry-network",
        ]
        if action == "start":
            m = self.model
            command += [
                "--runtime-overlay",
                str(self.args.build),
                "--model",
                m["model_path"],
                "--model-id",
                m["model_id"],
                "--model-sha256",
                m["model_sha256"],
                "--context-size",
                str(self.context),
                "--cpu-repack",
                "0",
                "--kv-hwm-trim",
                str(self.args.kv_hwm_trim),
                "--device",
                self.args.device,
                "--gap-ms",
                str(self.args.gap_ms),
                "--verify-kv-state",
                str(self.args.verify_kv_state),
                "--cpu-threads",
                str(self.args.cpu_threads),
                "--opbatch",
                str(self.args.opbatch),
                "--opqueue",
                str(self.args.opqueue),
                "--htp-big-cores",
                str(self.args.htp_big_cores),
                "--max-quiet-c",
                str(self.args.max_quiet_c),
                "--min-prime-khz",
                str(self.args.min_prime_khz),
                "--cooldown-timeout-s",
                str(self.args.cooldown_timeout_s),
                "--min-mem-avail-kib",
                "4000000",
                "--memory-ledger",
            ]
            if shared:
                command += ["--shared-weights"]
            if spec["kv_reduce"]:
                command += [
                    "--kv-reduce",
                    "1",
                    "--kv-policy",
                    spec["kv_policy"],
                    "--kv-keep-ratio",
                    str(spec["kv_keep_ratio"]),
                    "--kv-recent-tokens",
                    "32",
                    "--kv-tail-tokens",
                    "16",
                    "--compact-anchor",
                    str(self.args.compact_anchor),
                    "--kv-layer-stride",
                    str(spec["stride"]),
                    "--kv-min-evict-tokens",
                    str(self.args.kv_min_evict_tokens),
                ]
        subprocess.run(command, cwd=ROOT, check=True)
        if action == "start":
            health = read(directory / "health.json")
            if (
                health["condition"] != condition
                or health["shared_weights"] != shared
                or health["kv_reduce"] != bool(spec["kv_reduce"])
                or health["context_size"] != self.context
                or health["frozen_decode_version"] != 1
                or health.get("build_id") != self.build_id
                or health.get("cpu_threads") != self.args.cpu_threads
                or (spec["kv_reduce"] and health.get("kv_layer_stride") != spec["stride"])
                or read(directory / "manifest.json")["runtime_overlay"] != self.build
            ):
                raise RuntimeError(f"mismatched ablation phone condition {arm}")

    def replay_turns(self, arm, case, path, turns, routed, force_path):
        if arm not in ABLATION_ARM_SPECS:
            yield from super().replay_turns(arm, case, path, turns, routed, force_path)
            return
        require_attached_phone_preflight(
            ROOT,
            path,
            max_quiet_c=self.args.max_quiet_c,
            min_prime_khz=self.args.min_prime_khz,
            min_mem_avail_kib=4000000,
            cooldown_timeout_s=self.args.cooldown_timeout_s,
        )
        reset = request("/a2o/reset", {})
        save(path / "reset.json", reset)
        spec = ABLATION_ARM_SPECS[arm]
        warm_compactions = 0
        for i, turn in enumerate(turns):
            self.state(
                stage="replay",
                condition=arm,
                case=case["id"],
                turn=i,
                H=turn["H"],
                P=turn["P"],
                D=turn["D"],
            )
            prep, gap = None, None
            if i:
                begin = time.monotonic()
                prep = request("/a2o/tool-start", {}, timeout=12)
                time.sleep(max(0.0, self.args.gap_ms / 1000.0 - (time.monotonic() - begin)))
                gap = (time.monotonic() - begin) * 1000.0
            body = dict(
                turn["request"],
                max_tokens=turn["D"],
                expected_decode_tokens=turn["D"],
                frozen_graph_profile=True,
                **ablation_arm_chat_overrides(arm, i),
            )
            rid = f"{path.parent.parent.name}:{arm}:{case['id']}:{i}"
            response = _agent_chat_resilient(
                "http://127.0.0.1:18080", body, rid, total_timeout=2400
            )
            t = response["timings"]
            profile, kv = t["graph_profile"], t["kv_reduce"]
            prior = bool(kv and kv["policy"] == "prefill_prior" and kv["prior_ranked_tokens"] > 0)
            if (
                profile["decode"]["callback_graphs"] != 0
                or profile["prefill"]["callback_graphs"] != int(prior)
                or (t["prior_ms"] > 0) != prior
                or (not i and kv)
            ):
                raise RuntimeError("callback or cold compaction contract failed")
            if i:
                if t["route"] != "htp" or t["handoff_bytes"] <= 0 or not t["deferred_hc_anchor"]:
                    raise RuntimeError(f"ablation {arm} turn {i} lost HTP handoff or deferred anchor")
                if spec["force_path"] == "hc_full" and kv is not None:
                    raise RuntimeError(f"ablation {arm} turn {i} unexpectedly compacted")
                if kv is not None:
                    if not kv.get("compacted") or kv.get("evicted_count", 0) <= 0:
                        raise RuntimeError(f"ablation {arm} turn {i} invalid compaction: {kv}")
                    expected_stage = "post_handoff_cpu" if spec["post_handoff_kv_reduce"] else "pre_handoff_htp"
                    if kv.get("kv_reduce_stage") != expected_stage:
                        raise RuntimeError(
                            f"ablation {arm} turn {i} kv_reduce_stage={kv.get('kv_reduce_stage')} != {expected_stage}"
                        )
                    warm_compactions += 1
            record = dict(
                response=response,
                prepare=prep,
                tool_gap_ms=gap,
                observed_source_gap_ms=turn["observed_gap_ms"],
            )
            save(path / f"turn-{i:02d}.json", record)
            yield record
        if spec["kv_reduce"] and warm_compactions == 0:
            raise RuntimeError(f"ablation {arm} {case['id']} did not trigger any warm KV compaction")

    def replay(self, arm, case, path, *, recorded=False):
        if arm not in ABLATION_ARM_SPECS:
            return super().replay(arm, case, path, recorded=recorded)
        return super().replay("shared_prior", case, path, recorded=recorded)

    def run_cell(self, case, arm, target, stage, audit=False):
        if (target / "summary.json").is_file():
            return read(target / "summary.json")
        if (target / "FAILED.json").is_file():
            return None
        attempt = self.output / "services" / f"{stage}-{case['id']}-{arm}-{time.time_ns()}"
        raw = attempt / "capture"
        self.state(status="running", stage=f"{stage}_startup", case=case["id"], condition=arm, turn=None)
        try:
            try:
                self.service("start", arm, attempt)
                raw.mkdir()
                health = read(attempt / "health.json")
                if (
                    health["git_commit"] != self.git_commit
                    or health["gap_ms"] != int(self.args.gap_ms)
                    or health["verify_kv_state"]
                    or health["kv_hwm_trim"]
                    or health.get("cpu_threads") != self.args.cpu_threads
                    or health.get("htp_host_threads", 4) != 4
                ):
                    raise RuntimeError(f"phone health mismatch on {arm}: {health}")
                if audit:
                    cleared = {c["id"]: self.verify_renderer(c) for c in self.cases}
                    save(self.output / "renderer-audit.json", cleared)
                    if any(cleared.values()):
                        raise RuntimeError("phone renderer disagrees with frozen fixtures")
                if arm not in ABLATION_ARM_SPECS and (
                    arm in COST_ARMS
                    or (
                        self.cold_costs is not None
                        and arm not in CPU_REPLAY_ARMS
                        and arm not in STATIC_HC_ARMS
                        and arm != "full_htp"
                    )
                ):
                    payload = read(self.args.route_costs) if arm in COST_ARMS else {}
                    if self.cold_costs is not None:
                        payload["cold_model"] = read(self.args.cold_costs)
                    if arm in COST_ARMS:
                        payload["kv_model"] = kv_cost_payload(self.args.kv_costs)
                    if arm in ("shared_m1", "shiftserve_joint"):
                        payload["m1_explorer"] = {"enabled": True}
                    save(attempt / "cost-request.json", payload)
                    save(attempt / "cost-response.json", request("/a2o/costs", payload))
                if arm in ABLATION_ARM_SPECS:
                    # Override arm inside summary metadata without invoking shared_prior service flags.
                    self._ablation_replay(arm, case, raw)
                else:
                    self.replay(arm, case, raw)
            finally:
                try:
                    self.service("stop", arm, attempt)
                except Exception as stop_exc:
                    print(f"[STOP CLEANUP FALLBACK] arm={arm} case={case['id']} err={stop_exc}", flush=True)
                    self.cleanup_failed_phone_cell(attempt.name)
            return self.finalize_cell(attempt, target)
        except Exception as exc:
            self.cleanup_failed_phone_cell(attempt.name)
            return self.record_cell_failure(case, arm, target, attempt, exc)

    def _ablation_replay(self, arm, case, path):
        # Run replay_turns under custom ablation validation and reuse Frozen/Mobile summary schema.
        spec = ABLATION_ARM_SPECS[arm]
        records = list(self.replay_turns(arm, case, path, case["turns"], True, spec["force_path"]))
        from scripts.run.frozen_ablation import post_return_phases
        from a2o.profiling.memory_ledger import validate_turn

        results, samples = [], []
        evicted_from_anchor = 0
        for i, (turn, record) in enumerate(zip(case["turns"], records)):
            timing = record["response"]["timings"]
            res_msg = record["response"]
            if (
                res_msg["__verbose"]["tokens"] != turn["request"]["frozen_decode_ids"]
                or res_msg["__verbose"]["stop_type"] != "forced"
                or (timing["H_anchor"], timing["P_evaluated"], timing["predicted_n"])
                != (turn["H"], turn["P"], turn["D"])
            ):
                raise RuntimeError("frozen shape or decode tokens changed")
            expected_h_eff = turn["H"] - evicted_from_anchor
            if timing["H_effective"] != expected_h_eff:
                raise RuntimeError(f"{case['id']} turn {i}: effective {timing['H_effective']} != {expected_h_eff}")
            kv = timing["kv_reduce"]
            if kv and kv.get("compact_anchor") and i + 1 < len(case["turns"]):
                next_h = case["turns"][i + 1]["H"]
                next_live = kv["anchor_live_tokens"]
                if not 0 < next_live <= next_h - evicted_from_anchor:
                    raise RuntimeError(f"{case['id']} turn {i}: invalid compact anchor length {next_live}")
                evicted_from_anchor = next_h - next_live
            validate_turn(timing, False)
            if record["prepare"] and record["prepare"].get("memory_ledger"):
                samples.extend((i, s) for s in record["prepare"]["memory_ledger"]["samples"])
            samples.extend((i, s) for s in timing["memory_ledger"]["samples"])
            results.append(timing)
        warm = results[1:]
        post_return_ledger_ms = math.fsum(float((t.get("memory_ledger") or {}).get("collection_ms", 0.0)) for t in warm)
        inference_ledger_ms = math.fsum(float((t.get("memory_ledger") or {}).get("collection_ms", 0.0)) for t in results)
        post_return_ms = sum(t["total_ms"] for t in warm)
        inference_ms = sum(t["total_ms"] for t in results)
        summary = dict(
            benchmark=case["benchmark"],
            source_model=case["source_model"],
            id=case["id"],
            arm=arm,
            logical_rounds=len(results),
            warm_rounds=len(warm),
            warm_decode_tokens=sum(t["predicted_n"] for t in warm),
            post_return_ms=post_return_ms,
            inference_ms=inference_ms,
            decode_ms_per_token=sum(t["predicted_ms"] for t in warm) / sum(t["predicted_n"] for t in warm),
            htp_rounds=sum(t["route"] == "htp" for t in warm),
            prior_ranked_tokens=sum((t["kv_reduce"] or {}).get("prior_ranked_tokens", 0) for t in warm),
            max_sampled_rss_mib=max(t["decode_resources"][p]["VmRSS_kib"] / 1024 for t in results for p in ("before", "after")),
            max_sampled_swap_mib=max(t["decode_resources"][p]["VmSwap_kib"] / 1024 for t in results for p in ("before", "after")),
            handoff_bytes=sum(t["handoff_bytes"] for t in warm),
            post_return_ledger_ms=post_return_ledger_ms,
            post_return_excl_ledger_ms=post_return_ms - post_return_ledger_ms,
            inference_ledger_ms=inference_ledger_ms,
            inference_excl_ledger_ms=inference_ms - inference_ledger_ms,
            post_return_phases_ms=post_return_phases(warm),
        )
        turn_s, peak_s = max(samples, key=lambda item: item[1]["process"]["VmSwap_kib"])
        summary["memory_ledger_peak_swap"] = dict(turn=turn_s, stage=peak_s["stage"], swap_kib=peak_s["process"]["VmSwap_kib"])
        physical = [(i, s) for i, s in samples if s.get("pss_plus_identified_dma_kib") is not None]
        if physical:
            turn_p, peak_p = max(physical, key=lambda item: item[1]["pss_plus_identified_dma_kib"])
            summary["memory_ledger_peak_pss_plus_dma"] = dict(turn=turn_p, stage=peak_p["stage"], kib=peak_p["pss_plus_identified_dma_kib"])
        summary["memory_ledger_missing_pss_plus_dma_samples"] = len(samples) - len(physical)
        save(path / "summary.json", summary)


def make_args(
    *,
    build: Path,
    output: Path,
    fixtures: Path,
    conditions: list[str],
    gap_ms: int = 0,
    route_costs: Path | None = None,
    cold_costs: Path | None = None,
    kv_costs: Path | None = None,
    cpu_threads: int = 6,
    opbatch: int = 1024,
    opqueue: int = 16,
    htp_big_cores: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        build=build,
        output=output,
        fixtures=fixtures,
        model=ROOT / "configs/model-default.json",
        cpu_registry=ROOT / "shared/toolchains/htp/v73-libggml-cpu/libggml.so",
        route_costs=route_costs,
        cold_costs=cold_costs,
        kv_costs=kv_costs,
        context_size=16384,
        gap_ms=gap_ms,
        verify_kv_state=0,
        kv_hwm_trim=0,
        device="xiaomi14pro",
        memory_ledger=True,
        compact_anchor=1,
        kv_layer_stride=4,
        kv_min_evict_tokens=256,
        conditions=conditions,
        case=None,
        force_path="none",
        single_turn=False,
        m1_explorer=True,
        graph_profile=True,
        cpu_threads=cpu_threads,
        opbatch=opbatch,
        opqueue=opqueue,
        htp_big_cores=htp_big_cores,
        max_quiet_c=39.0,
        min_prime_khz=2000000,
        cooldown_timeout_s=360,
        continue_on_cell_error=True,
    )


def run_cohort(
    *,
    build: Path,
    output: Path,
    case_ids: tuple[str, ...],
    arms: list[str],
    gap_ms: int = 0,
    use_cost_models: bool = False,
) -> dict:
    fixtures = build_subset_fixtures(output / "fixtures.json", case_ids)
    args = make_args(
        build=build,
        output=output,
        fixtures=fixtures,
        conditions=arms,
        gap_ms=gap_ms,
        route_costs=DEFAULT_ROUTE_COSTS if use_cost_models else None,
        cold_costs=DEFAULT_COLD_COSTS if use_cost_models else None,
        kv_costs=DEFAULT_PRIORS if use_cost_models else None,
    )
    campaign = AblationCampaign(args, read(fixtures))
    order = mg.balanced_order(campaign.cases, arms=tuple(arms))
    completed = 0
    failed = 0
    audited = (output / "renderer-audit.json").is_file()
    for row in order:
        case = next(c for c in campaign.cases if c["id"] == row["case"])
        for arm in row["order"]:
            res = campaign.run_cell(case, arm, output / arm / case["id"], "formal", audit=not audited)
            audited = True
            if res is None:
                failed += 1
            else:
                completed += 1
    status = "passed" if failed == 0 else "passed_with_cell_failures"
    save(
        output / "cohort-status.json",
        {
            "status": status,
            "completed_cells": completed,
            "failed_cells": failed,
            "arms": arms,
            "cases": list(case_ids),
            "gap_ms": gap_ms,
        },
    )
    return {"status": status, "completed_cells": completed, "failed_cells": failed}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _cell_warm_turns(cell_dir: Path) -> list[dict]:
    paths = sorted(cell_dir.glob("turn-*.json"))
    return [read(p)["response"]["timings"] for p in paths[1:]]


def summarize_p0(out_dir: Path) -> dict:
    rows = []
    for case in P0_CASES:
        for arm in ("pre_handoff_htp", "post_handoff_cpu"):
            cell = out_dir / arm / case
            if not (cell / "summary.json").is_file():
                continue
            s = read(cell / "summary.json")
            warm = _cell_warm_turns(cell)
            kvs = [t["kv_reduce"] for t in warm if t.get("kv_reduce")]
            phases = s["post_return_phases_ms"]
            rows.append({
                "case": case,
                "arm": arm,
                "warm_rounds": s["warm_rounds"],
                "compact_rounds": len(kvs),
                "handoff_bytes": s["handoff_bytes"],
                "handoff_mib": round(s["handoff_bytes"] / (1024 * 1024), 2),
                "handoff_ms": round(phases["handoff"], 2),
                "evict_ms": round(sum(float(k.get("evict_ms", 0.0)) for k in kvs), 2),
                "compact_ms": round(sum(float(k.get("compact_ms", 0.0)) for k in kvs), 2),
                "kv_reduce_ms": round(phases["compact"], 2),
                "snapshot_ms": round(phases["snapshot"], 2),
                "prefill_ms": round(phases["prefill_including_prior"], 2),
                "decode_ms": round(phases["decode"], 2),
                "post_return_excl_ledger_ms": round(s["post_return_excl_ledger_ms"], 2),
                "peak_pss_plus_dma_mib": round(s["memory_ledger_peak_pss_plus_dma"]["kib"] / 1024, 2),
                "peak_vmswap_mib": round(s["memory_ledger_peak_swap"]["swap_kib"] / 1024, 2),
            })
    _write_csv(out_dir / "p0_summary.csv", rows)
    pairs = []
    by_pair = {(r["case"], r["arm"]): r for r in rows}
    for case in P0_CASES:
        pre, post = by_pair.get((case, "pre_handoff_htp")), by_pair.get((case, "post_handoff_cpu"))
        if pre and post:
            pairs.append({
                "case": case,
                "handoff_bytes_reduction_pct": round(100.0 * (1.0 - pre["handoff_bytes"] / post["handoff_bytes"]), 2),
                "handoff_speedup_post_over_pre": round(post["handoff_ms"] / pre["handoff_ms"], 3),
                "kv_reduce_overhead_delta_ms": round(post["kv_reduce_ms"] - pre["kv_reduce_ms"], 2),
                "post_return_delta_ms": round(post["post_return_excl_ledger_ms"] - pre["post_return_excl_ledger_ms"], 2),
                "peak_pss_plus_dma_delta_mib": round(post["peak_pss_plus_dma_mib"] - pre["peak_pss_plus_dma_mib"], 2),
            })
    summary = {"rows": rows, "pairs": pairs}
    save(out_dir / "p0_summary.json", summary)
    return summary


def summarize_p1_keep_ratio(out_dir: Path) -> dict:
    rho_map = {"rho_0.50": 0.50, "rho_0.75": 0.75, "rho_1.00": 1.00}
    rows = []
    for case in P1_AC_CASES:
        for arm, rho in rho_map.items():
            cell = out_dir / arm / case
            if not (cell / "summary.json").is_file():
                continue
            s = read(cell / "summary.json")
            warm = _cell_warm_turns(cell)
            phases = s["post_return_phases_ms"]
            rows.append({
                "arm": arm,
                "rho": rho,
                "case": case,
                "warm_rounds": s["warm_rounds"],
                "compact_rounds": sum(1 for t in warm if t.get("kv_reduce")),
                "handoff_mib": round(s["handoff_bytes"] / (1024 * 1024), 2),
                "handoff_ms": round(phases["handoff"], 2),
                "prior_ms": round(phases.get("prior", 0.0), 2),
                "compact_ms": round(phases["compact"], 2),
                "prefill_ms": round(phases["prefill_including_prior"], 2),
                "decode_ms": round(phases["decode"], 2),
                "post_return_excl_ledger_ms": round(s["post_return_excl_ledger_ms"], 2),
                "peak_pss_plus_dma_mib": round(s["memory_ledger_peak_pss_plus_dma"]["kib"] / 1024, 2),
                "peak_vmswap_mib": round(s["memory_ledger_peak_swap"]["swap_kib"] / 1024, 2),
            })
    _write_csv(out_dir / "p1_keep_ratio_summary.csv", rows)
    aggregates = []
    for arm, rho in rho_map.items():
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        aggregates.append({
            "arm": arm,
            "rho": rho,
            "cases": len(sub),
            "mean_post_return_excl_ledger_ms": round(statistics.mean(r["post_return_excl_ledger_ms"] for r in sub), 2),
            "mean_handoff_mib": round(statistics.mean(r["handoff_mib"] for r in sub), 2),
            "peak_pss_plus_dma_mib": max(r["peak_pss_plus_dma_mib"] for r in sub),
        })
    summary = {"rows": rows, "aggregates": aggregates}
    save(out_dir / "p1_keep_ratio_summary.json", summary)
    return summary


def summarize_p1_layer_stride(out_dir: Path) -> dict:
    stride_map = {"stride_1": 1, "stride_4": 4, "stride_8": 8}
    rows = []
    for case in P1_AC_CASES:
        for arm, stride in stride_map.items():
            cell = out_dir / arm / case
            if not (cell / "summary.json").is_file():
                continue
            s = read(cell / "summary.json")
            warm = _cell_warm_turns(cell)
            phases = s["post_return_phases_ms"]
            cb = sum(int((t.get("graph_profile") or {}).get("prefill", {}).get("callback_graphs", 0)) for t in warm)
            prefill = phases["prefill_including_prior"]
            prior_ms = phases.get("prior", 0.0)
            rows.append({
                "arm": arm,
                "stride": stride,
                "case": case,
                "warm_rounds": s["warm_rounds"],
                "compact_rounds": sum(1 for t in warm if t.get("kv_reduce")),
                "callback_graphs": cb,
                "prior_ms": round(prior_ms, 2),
                "prefill_including_prior_ms": round(prefill, 2),
                "prior_share_pct": round(100.0 * prior_ms / prefill, 3) if prefill > 0 else 0.0,
                "post_return_excl_ledger_ms": round(s["post_return_excl_ledger_ms"], 2),
            })
    _write_csv(out_dir / "p1_layer_stride_summary.csv", rows)
    aggregates = []
    for arm, stride in stride_map.items():
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        aggregates.append({
            "arm": arm,
            "stride": stride,
            "cases": len(sub),
            "mean_prior_ms": round(statistics.mean(r["prior_ms"] for r in sub), 2),
            "mean_prefill_ms": round(statistics.mean(r["prefill_including_prior_ms"] for r in sub), 2),
            "mean_prior_share_pct": round(statistics.mean(r["prior_share_pct"] for r in sub), 3),
            "mean_post_return_excl_ledger_ms": round(statistics.mean(r["post_return_excl_ledger_ms"] for r in sub), 2),
        })
    summary = {"rows": rows, "aggregates": aggregates}
    save(out_dir / "p1_layer_stride_summary.json", summary)
    return summary


def summarize_p1_gap(gap_root: Path) -> dict:
    rows = []
    for gap in P1_GAPS_MS:
        gdir = gap_root / f"gap-{gap}"
        for case in P1_GAP_CASES:
            for arm in ("full_cpu", "shared_static_hc", "shiftserve_joint"):
                cell = gdir / arm / case
                if not (cell / "summary.json").is_file():
                    continue
                s = read(cell / "summary.json")
                phases = s["post_return_phases_ms"]
                rows.append({
                    "gap_ms": gap,
                    "benchmark": s["benchmark"],
                    "case": case,
                    "arm": arm,
                    "warm_rounds": s["warm_rounds"],
                    "htp_rounds": s["htp_rounds"],
                    "prep_wait_ms": round(phases["prep_wait"], 2),
                    "prefill_ms": round(phases["prefill_including_prior"], 2),
                    "decode_ms": round(phases["decode"], 2),
                    "post_return_excl_ledger_ms": round(s["post_return_excl_ledger_ms"], 2),
                    "peak_pss_plus_dma_mib": round(s["memory_ledger_peak_pss_plus_dma"]["kib"] / 1024, 2),
                    "peak_vmswap_mib": round(s["memory_ledger_peak_swap"]["swap_kib"] / 1024, 2),
                })
    _write_csv(gap_root / "p1_gap_summary.csv", rows)
    by_gap = []
    indexed = {(r["gap_ms"], r["case"], r["arm"]): r for r in rows}
    for gap in P1_GAPS_MS:
        cpu_spd, shc_spd = [], []
        for case in P1_GAP_CASES:
            ss = indexed.get((gap, case, "shiftserve_joint"))
            fc = indexed.get((gap, case, "full_cpu"))
            sh = indexed.get((gap, case, "shared_static_hc"))
            if ss and fc:
                cpu_spd.append(fc["post_return_excl_ledger_ms"] / ss["post_return_excl_ledger_ms"])
            if ss and sh:
                shc_spd.append(sh["post_return_excl_ledger_ms"] / ss["post_return_excl_ledger_ms"])
        agg = {"gap_ms": gap, "paired_cases": len(cpu_spd)}
        for arm in ("full_cpu", "shared_static_hc", "shiftserve_joint"):
            sub = [r for r in rows if r["gap_ms"] == gap and r["arm"] == arm]
            if sub:
                agg[f"{arm}_mean_post_return_ms"] = round(statistics.mean(r["post_return_excl_ledger_ms"] for r in sub), 2)
                agg[f"{arm}_mean_prep_wait_ms"] = round(statistics.mean(r["prep_wait_ms"] for r in sub), 2)
        if cpu_spd:
            agg["shiftserve_geomean_speedup_vs_full_cpu"] = round(statistics.geometric_mean(cpu_spd), 3)
        if shc_spd:
            agg["shiftserve_geomean_speedup_vs_shared_static_hc"] = round(statistics.geometric_mean(shc_spd), 3)
        by_gap.append(agg)
    summary = {"rows": rows, "by_gap": by_gap}
    save(gap_root / "p1_gap_summary.json", summary)
    return summary


def save_status(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def wait_for_pid(status_file: Path, state: dict, wait_pid: int | None) -> None:
    if wait_pid is None or wait_pid <= 0 or not pid_alive(wait_pid):
        return
    state["status"] = "waiting_previous_queue"
    state["waiting_for_pid"] = wait_pid
    save_status(status_file, state)
    print(f"=== Waiting for queue PID {wait_pid} before starting P0+P1 ablations ===", flush=True)
    while pid_alive(wait_pid):
        time.sleep(10)
    time.sleep(3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    p.add_argument("--wait-pid", type=int, default=None)
    p.add_argument("--only", choices=["all", "p0", "p1_rho", "p1_stride", "p1_gap"], default="all")
    args = p.parse_args()

    wait_pid = args.wait_pid
    if wait_pid is None and DEFAULT_WAIT_PID_FILE.is_file():
        wait_pid = int(DEFAULT_WAIT_PID_FILE.read_text().strip())

    args.output.mkdir(parents=True, exist_ok=True)
    pid_file = ROOT / "results/mainline-eval/t6-8b-p0-p1-ablations.pid"
    pid_file.write_text(f"{os.getpid()}\n")

    state = {
        "status": "initializing",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "build": str(args.build),
        "output": str(args.output),
        "stages": {},
    }
    wait_for_pid(args.status_file, state, wait_pid)
    state.pop("waiting_for_pid", None)
    state["status"] = "running"
    save_status(args.status_file, state)

    run_all = args.only == "all"
    if run_all or args.only == "p0":
        p0_dir = args.output / "p0_pre_vs_post_handoff"
        state["current"] = "p0_pre_vs_post_handoff"
        save_status(args.status_file, state)
        res = run_cohort(
            build=args.build,
            output=p0_dir,
            case_ids=P0_CASES,
            arms=["pre_handoff_htp", "post_handoff_cpu"],
            gap_ms=0,
            use_cost_models=False,
        )
        res["summary"] = summarize_p0(p0_dir)["pairs"]
        state["stages"]["p0_pre_vs_post_handoff"] = res
        save_status(args.status_file, state)

    if run_all or args.only == "p1_rho":
        rho_dir = args.output / "p1_keep_ratio"
        state["current"] = "p1_keep_ratio"
        save_status(args.status_file, state)
        res = run_cohort(
            build=args.build,
            output=rho_dir,
            case_ids=P1_AC_CASES,
            arms=["rho_0.50", "rho_0.75", "rho_1.00"],
            gap_ms=0,
            use_cost_models=False,
        )
        res["summary"] = summarize_p1_keep_ratio(rho_dir)["aggregates"]
        state["stages"]["p1_keep_ratio"] = res
        save_status(args.status_file, state)

    if run_all or args.only == "p1_stride":
        stride_dir = args.output / "p1_layer_stride"
        state["current"] = "p1_layer_stride"
        save_status(args.status_file, state)
        res = run_cohort(
            build=args.build,
            output=stride_dir,
            case_ids=P1_AC_CASES,
            arms=["stride_1", "stride_4", "stride_8"],
            gap_ms=0,
            use_cost_models=False,
        )
        res["summary"] = summarize_p1_layer_stride(stride_dir)["aggregates"]
        state["stages"]["p1_layer_stride"] = res
        save_status(args.status_file, state)

    if run_all or args.only == "p1_gap":
        gap_root = args.output / "p1_gap_sensitivity_8b"
        state["current"] = "p1_gap_sensitivity_8b"
        save_status(args.status_file, state)
        gap_stats = {}
        for gap in P1_GAPS_MS:
            gap_stats[str(gap)] = run_cohort(
                build=args.build,
                output=gap_root / f"gap-{gap}",
                case_ids=P1_GAP_CASES,
                arms=["full_cpu", "shared_static_hc", "shiftserve_joint"],
                gap_ms=gap,
                use_cost_models=True,
            )
        state["stages"]["p1_gap_sensitivity_8b"] = {
            "gaps": gap_stats,
            "summary": summarize_p1_gap(gap_root)["by_gap"],
        }
        save_status(args.status_file, state)

    state["status"] = "completed"
    state["current"] = None
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_status(args.status_file, state)
    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import os
    main()
