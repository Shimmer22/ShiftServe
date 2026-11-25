#!/usr/bin/env python3
"""Teacher-force exact τ² prompt/output IDs through one live anchor chain per condition."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from a2o.runtime.phone_preflight import require_attached_phone_preflight
from a2o.workloads.toolsandbox import post
from scripts.collect.tau2_adaptive_kv_e2e import route_phone, sample_phone_memory
from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post_resilient

ENDPOINT = "http://10.29.131.50:18780"
CONDITIONS = ("route_m1_m2_compact", "cpu_baseline", "cpu_adaptive_compact", "route_m1_only")
CONTEXT_PROBE_CONDITIONS = ("route_m1_only_serial_context", "route_m1_only")
EMPTY_THINK = "<think>\n\n</think>\n\n"


def save(path: Path, obj) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def load_turns(trace: Path, indices: list[int]) -> list[dict]:
    events = [json.loads(line) for line in (trace / "events.jsonl").read_text().splitlines()]
    agent = [row for row in events if row["kind"] == "llm_response" and row["role"] == "agent"]
    start = next(i for i, row in enumerate(agent) if row["index"] == indices[0])
    selected = agent[start:start + len(indices)]
    if [row["index"] for row in selected] != indices:
        raise ValueError("Selected requests must be consecutive agent turns")
    turns = []
    for i, row in enumerate(selected):
        source = json.loads((trace / f"{row['index']:03d}-agent.request.json").read_text())
        recorded = json.loads((trace / f"{row['index']:03d}-agent.response.json").read_text())
        prompt_ids = recorded["prompt_token_ids"]
        decode_ids = recorded["choices"][0]["token_ids"]
        if not prompt_ids or not decode_ids or len(prompt_ids) != row["prompt_tokens"] or len(decode_ids) != row["D"]:
            raise ValueError(f"Source turn {row['index']} has inconsistent frozen token IDs")
        body = {key: source[key] for key in ("messages", "tools") if key in source}
        body.update(frozen_prompt_ids=prompt_ids, frozen_decode_ids=decode_ids, max_tokens=len(decode_ids))
        gap_ms = None
        if i:
            previous = selected[i - 1]
            between = [e for e in events if previous["response_ns"] < e["monotonic_ns"] < row["request_ns"]]
            tools = [e for e in between if e["kind"] == "tool_result" and e["call"]["requestor"] == "assistant"]
            if not tools or any(e["kind"] == "llm_response" and e["role"] == "user" for e in between):
                raise ValueError(f"Source turn {row['index']} has no isolated assistant tool boundary")
            gap_ms = (max(e["end_ns"] for e in tools) - min(e["start_ns"] for e in tools)) / 1e6
        turns.append({"source_index": row["index"], "source_d": row["D"], "gap_ms": gap_ms, "request": body})
    return turns


def render_chain(turns: list[dict]) -> list[dict]:
    previous = []
    rows = []
    for turn in turns:
        prompt = _http_post_resilient(ENDPOINT + "/apply-template", turn["request"], timeout=30, retries=4)["prompt"]
        ids = _http_post_resilient(ENDPOINT + "/tokenize", {"content": prompt}, timeout=30, retries=4)["tokens"]
        source = turn["request"]["frozen_prompt_ids"]
        if ids != source:
            common = next((i for i, (x, y) in enumerate(zip(ids, source)) if x != y), min(len(ids), len(source)))
            raise RuntimeError(f"Source turn {turn['source_index']} phone/source token mismatch: "
                               f"common={common}, phone={len(ids)}, source={len(source)}")
        if previous and (len(ids) <= len(previous) or ids[:len(previous)] != previous):
            raise RuntimeError(f"Source turn {turn['source_index']} would miss the live anchor")
        rows.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "H": len(previous), "P": len(ids) - len(previous),
                     "D": len(turn["request"]["frozen_decode_ids"])})
        anchor = ids
        if prompt.endswith(EMPTY_THINK):
            candidate = _http_post_resilient(ENDPOINT + "/tokenize", {"content": prompt[:-len(EMPTY_THINK)]},
                                            timeout=30, retries=4)["tokens"]
            if len(previous) < len(candidate) < len(ids) and ids[:len(candidate)] == candidate:
                anchor = candidate
        previous = anchor
    return rows


def run_condition(cfg: dict, cond: dict, turns: list[dict], out: Path, reference: list | None,
                  gap_override_ms: float | None) -> list:
    name = cond["name"]
    force_hc = cond.get("frozen_force_hc", False)
    serial_context = cond.get("frozen_serial_context", False)
    svc_dir = out / f"{out.name}-{name}"
    svc_log = out / f"{name}.log"
    route_phone("start", cfg, cond, svc_dir, svc_log)
    try:
        with urllib.request.urlopen(ENDPOINT + "/health", timeout=20) as response:
            health = json.load(response)
        expected = dict(condition=cond["condition"], model=cfg["model_id"], frozen_decode_version=1,
                        kv_reduce=bool(cond["kv_reduce"]), compact_anchor=bool(cond["compact_anchor"]),
                        kv_cost_loaded=cond["use_kv_costs"], single_resident=cond["single_resident"])
        if any(health[key] != value for key, value in expected.items()):
            raise RuntimeError(f"Unexpected phone configuration for {name}: {health}")
        save(svc_dir / "verified-health.json", health)
        require_attached_phone_preflight(ROOT, svc_dir, expected_owner=svc_dir.name)
        plan = render_chain(turns)
        if reference is None:
            save(out / "rendered-chain.json", plan)
        elif plan != reference:
            raise RuntimeError(f"Rendered requests changed in {name}")
        post(ENDPOINT, "/a2o/reset", {}, 60)
        for i, (turn, shape) in enumerate(zip(turns, plan)):
            prep = None
            actual_gap_ms = None
            target_gap_ms = turn["gap_ms"]
            if i:
                if gap_override_ms is not None:
                    target_gap_ms = gap_override_ms
                gap_start = time.monotonic()
                if cond["condition"] == "route":
                    prep = post(ENDPOINT, "/a2o/tool-start", {"frozen_serial_context": serial_context}, 30)
                    if not prep["started"]:
                        raise RuntimeError(f"Preparation failed to start for {name} turn {i}: {prep}")
                    if force_hc and prep.get("serial_context") != serial_context:
                        raise RuntimeError(f"Unexpected preparation context mode for {name} turn {i}: {prep}")
                time.sleep(max(0, target_gap_ms / 1000 - (time.monotonic() - gap_start)))
                actual_gap_ms = (time.monotonic() - gap_start) * 1000
            print(f"{name} turn {i}: submitting {shape}, tool gap {actual_gap_ms} ms", flush=True)
            body = {**turn["request"], "model": cfg["model_id"],
                    "expected_decode_tokens": cfg["expected_decode_tokens"]}
            if i and force_hc:
                body["frozen_force_hc"] = True
            response = _agent_chat_resilient(ENDPOINT, body, f"{out.name}:{name}:{i}")
            record = {"turn": i, "source_index": turn["source_index"], "source_tool_ms": turn["gap_ms"],
                      "target_gap_ms": target_gap_ms,
                      "actual_gap_ms": actual_gap_ms, "prepare": prep, "response": response}
            destination = svc_dir / f"turn-{i:02d}.json"
            save(destination, record)
            prompt = response["__verbose"]["prompt"]
            timing = response["timings"]
            if hashlib.sha256(prompt.encode()).hexdigest() != shape["prompt_sha256"]:
                raise RuntimeError(f"Unexpected prompt in measured {name} turn {i}")
            if (timing["H_anchor"], timing["P_evaluated"], timing["predicted_n"]) != (shape["H"], shape["P"], shape["D"]):
                raise RuntimeError(f"Unexpected measured anchor or suffix in {name} turn {i}: {timing}")
            if (response["__verbose"]["stop_type"] != "forced"
                    or response["__verbose"]["tokens"] != turn["request"]["frozen_decode_ids"]):
                raise RuntimeError(f"Unexpected evaluated token sequence in {name} turn {i}")
            if i and force_hc and (timing["route_reason"] != "frozen_forced_hc"
                                   or timing["context_probe"]["serial_context"] != serial_context):
                raise RuntimeError(f"Unexpected HC execution context mode for {name} turn {i}")
            record["memory"] = sample_phone_memory(svc_dir)
            record["decode_ms_per_token"] = timing["predicted_ms"] / shape["D"]
            resources = timing["decode_resources"]
            record["decode_major_faults"] = resources["after"]["major_faults"] - resources["before"]["major_faults"]
            save(destination, record)
            print(f"{name} turn {i}: {timing['route']} / {timing['route_reason']}, "
                  f"H={timing['H_anchor']}/{timing['H_effective']} P={timing['P_evaluated']} "
                  f"D={timing['predicted_n']} decode={record['decode_ms_per_token']:.1f} ms/token "
                  f"decode_majflt={record['decode_major_faults']} "
                  f"total={timing['total_ms']:.1f} ms", flush=True)
        return plan
    finally:
        route_phone("stop", cfg, cond, svc_dir, svc_log)


def run_campaign(out: Path, manifest: dict, resume: bool) -> None:
    conditions = manifest["conditions"]
    completed = []
    metrics = {}
    plan = None
    status_path = out / "status.json"
    manifest_path = out / "manifest.json"
    if resume:
        status = json.loads(status_path.read_text())
        if status["status"] != "failed" or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Resume requires the failed campaign's exact fixture, configuration and binary")
        completed = status["completed"]
        if completed != conditions[:len(completed)] or len(completed) >= len(conditions):
            raise RuntimeError("Invalid completed-condition checkpoint")
        summary_path = out / "summary.json"
        metrics = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if set(metrics) != set(completed):
            raise RuntimeError("Summary does not match completed-condition checkpoint")
        plan_path = out / "rendered-chain.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
        failed_name = conditions[len(completed)]
        archive = out / "failed-attempts" / str(time.time_ns())
        archive.mkdir(parents=True)
        save(archive / "status.json", status)
        for old in (out / f"{out.name}-{failed_name}", out / f"{failed_name}.log"):
            if old.exists():
                old.rename(archive / old.name)
    else:
        save(manifest_path, manifest)
    config = manifest["config"]
    turns = manifest["turns"]
    try:
        for name in conditions[len(completed):]:
            save(status_path, {"status": "running", "condition": name, "completed": completed})
            cond = next(c for c in config["conditions"] if c["name"] == name)
            plan = run_condition(config, cond, turns, out, plan, manifest["gap_override_ms"])
            svc_dir = out / f"{out.name}-{name}"
            measured = [json.loads(p.read_text())["response"]["timings"]
                        for p in sorted(svc_dir.glob("turn-*.json"))]
            warm = measured[1:]
            decode_steps = sum(t["predicted_n"] for t in warm)
            metrics[name] = {"warm_decode_steps": decode_steps,
                             "warm_decode_ms_per_token": sum(t["predicted_ms"] for t in warm) / decode_steps,
                             "warm_decode_major_faults": sum(t["decode_resources"]["after"]["major_faults"]
                                                             - t["decode_resources"]["before"]["major_faults"] for t in warm),
                             "warm_service_ms": sum(t["total_ms"] for t in warm)}
            save(out / "summary.json", metrics)
            completed.append(name)
    except BaseException as exc:
        save(status_path, {"status": "failed", "completed": completed, "error": str(exc)})
        raise
    save(status_path, {"status": "complete", "completed": completed})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-ms", type=float, help="Use the same controlled tool gap in every warm turn")
    parser.add_argument("--resume", action="store_true", help="Re-run only unfinished conditions of a failed campaign")
    parser.add_argument("--context-probe", action="store_true",
                        help="Compare forced full-KV HC with serial and overlapping contexts")
    parser.add_argument("--probe-cc", action="store_true",
                        help="Include a single-resident CPU baseline in the context probe")
    args = parser.parse_args()
    if args.probe_cc and not args.context_probe:
        parser.error("--probe-cc requires --context-probe")
    if len(args.indices) < (2 if args.context_probe else 3) or (args.gap_ms is not None and not 0 <= args.gap_ms < 120000):
        parser.error("provide a consecutive trajectory with a warm boundary and a valid gap")
    config = json.loads(args.config.read_text())
    config["binary"] = str(args.binary.resolve())
    conditions = CONTEXT_PROBE_CONDITIONS if args.context_probe else CONDITIONS
    if args.context_probe:
        m1 = next(c for c in config["conditions"] if c["name"] == "route_m1_only")
        probe_conditions = [dict(m1, frozen_force_hc=True),
                            dict(m1, name=conditions[0], frozen_force_hc=True, frozen_serial_context=True)]
        if args.probe_cc:
            conditions = ("cpu_baseline", *conditions)
            probe_conditions.append(next(c for c in config["conditions"] if c["name"] == "cpu_baseline"))
        config["conditions"] = probe_conditions
    turns = load_turns(args.trace, args.indices)
    out = args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    manifest = {"config": config, "trace": str(args.trace.resolve()), "turns": turns,
                "conditions": list(conditions), "gap_override_ms": args.gap_ms,
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "scope": "exact prompt and teacher-forced output IDs; one live anchor chain per condition"}
    with (out / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_campaign(out, manifest, args.resume)


if __name__ == "__main__":
    main()
