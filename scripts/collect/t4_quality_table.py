"""从 T4 逐题记录生成质量表与 KV 执行审计。"""

import csv
import json
from pathlib import Path
import re

from a2o.profiling.host_accuracy import same_run


def read(path):
    return json.loads(path.read_text())


def task_observations(bench, directory, summary, task):
    if bench == "bfcl":
        calls = read(directory / "tasks" / task["id"] / "calls.json")
        quality = task["quality"]
        return (calls, task["trace_complete"], float(bool(quality.get("valid"))),
                task["error"], bool(quality.get("checker_error")))
    if bench == "tau2":
        calls = [call for call in task["calls"] if call["is_agent"]]
        return calls, task["complete"], task["reward"], task["error"], task["reward"] is None

    folder = directory / summary["attempt"] / "run" / task["name"]
    responses = (read(path) for path in sorted(folder.glob("*.response.json")))
    calls = [{"timings": response.get("timings"), "finish_reason": response["choices"][0]["finish_reason"]}
             for response in responses]
    evaluation = task["evaluation"]
    return (calls, task["trace_complete"], evaluation["similarity"] if evaluation else None,
            task["error"], evaluation is None)


def validate_task_ids(bench, summary, expected, directory):
    tasks = summary["results"]
    if bench == "tau2":
        ids = [task["task_id"] for task in tasks]
        keys = [(task["domain"], task["task_id"]) for task in tasks]
    else:
        id_field = "id" if bench == "bfcl" else "name"
        ids = [task[id_field] for task in tasks]
        keys = ids
    if len(keys) != expected or len(set(keys)) != expected or ids != summary["selected_task_ids"]:
        raise ValueError(f"题集数量或唯一性校验失败 {directory}")
    return keys


def calculate_row(bench, model, condition, directory, summary, expected):
    tasks = summary["results"]
    server_dir = directory / summary["attempt"] if bench == "toolsandbox" else directory
    server_log = (server_dir / "agent-server.log").read_text(encoding="utf-8", errors="replace")
    offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", server_log)
    if not offload or int(offload[1]) != int(offload[2]) or "CUDA0 KV buffer size" not in server_log:
        raise ValueError(f"缺少模型层和 KV 的 CUDA offload 证据 {server_dir}")

    scores = []
    stats = dict(trace_complete=0, truncated_tasks=0, context_overflow_tasks=0,
                 evaluation_failure_tasks=0, absent_score_tasks=0, request_error_tasks=0,
                 agent_calls=0, compact_turns=0, prior_turns=0, recent_turns=0,
                 compact_anchor_reuse_turns=0, before_tokens_on_compacted_turns=0,
                 logical_tokens_on_compacted_turns=0, kept_tokens_on_compacted_turns=0,
                 evicted_tokens=0)
    for task in tasks:
        calls, complete, score, error, evaluation_failed = task_observations(bench, directory, summary, task)
        error = (error or "").lower()
        scores.append(float(score or 0.0))
        stats["trace_complete"] += int(complete)
        stats["truncated_tasks"] += int(any(call["finish_reason"] == "length" for call in calls))
        stats["context_overflow_tasks"] += int("context overflow" in error or "context size" in error)
        stats["evaluation_failure_tasks"] += int(evaluation_failed)
        stats["absent_score_tasks"] += int(score is None)
        stats["request_error_tasks"] += int(bool(re.search(r"http \d{3}|urlerror|timeouterror|async chat failed", error)))
        stats["agent_calls"] += len(calls)
        for call in calls:
            timing = call["timings"]
            if not timing:
                raise ValueError(f"缺少服务执行记录 {directory}")
            stats["compact_anchor_reuse_turns"] += int(timing["H_anchor"] > timing["H_effective"])
            kv = timing.get("kv_reduce")
            if kv and kv["evicted_count"] > 0:
                if condition == "full_kv" or not kv.get("compact_anchor"):
                    raise ValueError(f"条件与实际裁剪或锚点状态不符 {directory}")
                stats["compact_turns"] += 1
                stats["prior_turns"] += int(kv["policy"] == "prefill_prior" and kv["prior_ranked_tokens"] > 0)
                stats["recent_turns"] += int(kv["policy"] == "recent")
                stats["before_tokens_on_compacted_turns"] += kv["before_pos_count"]
                stats["logical_tokens_on_compacted_turns"] += timing["H_anchor"] + timing["P_evaluated"]
                stats["kept_tokens_on_compacted_turns"] += kv["kept_count"]
                stats["evicted_tokens"] += kv["evicted_count"]
    return {
        "benchmark": bench, "model": model, "condition": condition,
        "metric": {"bfcl": "accuracy", "toolsandbox": "mean_similarity", "tau2": "mean_reward"}[bench],
        "expected_tasks": expected, "scored_as_full_success": sum(score == 1 for score in scores),
        "quality": sum(scores) / expected, **stats,
        "evicted_fraction_on_compacted_turns": (
            stats["evicted_tokens"] / stats["before_tokens_on_compacted_turns"]
            if stats["before_tokens_on_compacted_turns"] else None),
        "cum_evicted_fraction_vs_logical": (
            1.0 - stats["kept_tokens_on_compacted_turns"] / stats["logical_tokens_on_compacted_turns"]
            if stats["logical_tokens_on_compacted_turns"] else None),
    }


def write_table(out: Path, conditions, *, smoke=False):
    rows, missing, checks = [], [], []
    for model in ("8b", "4b"):
        for bench, full_count in (("bfcl", 60), ("toolsandbox", 28), ("tau2", 36)):
            stage = out / f"{out.name}-{bench}-{model}"
            plan_path = stage / "run-metadata.json"
            plans = read(plan_path) if plan_path.exists() else {}
            reference = None
            for condition in conditions:
                directory = stage / "conditions" / condition
                path = directory / "worker-summary.json"
                key = f"{bench}/{model}/{condition}"
                if not path.exists():
                    missing.append(key)
                    continue
                summary = read(path)
                if condition not in plans or not same_run(summary, plans[condition]):
                    raise ValueError(f"汇总身份与计划不符 {path}")
                selected = validate_task_ids(bench, summary, 1 if smoke else full_count, directory)
                if reference is not None and selected != reference:
                    raise ValueError(f"条件题集不一致 {stage}")
                reference = selected
                row = calculate_row(bench, model, condition, directory, summary, 1 if smoke else full_count)
                rows.append(row)
                if condition != "full_kv" and (
                    not row["compact_anchor_reuse_turns"]
                    or (bench != "toolsandbox" and not row["prior_turns"])
                ):
                    checks.append(key)

    obj = dict(status="smoke" if smoke else "partial" if missing else "needs_kv_review" if checks else "complete",
               denominator="all selected tasks; missing reward/similarity contributes zero",
               missing=missing, conditions_without_prior_or_anchor_reuse=checks, rows=rows)
    with (out / "table3.json.tmp").open("w") as stream:
        json.dump(obj, stream, indent=2)
        stream.write("\n")
    (out / "table3.json.tmp").replace(out / "table3.json")
    if rows:
        with (out / "table3.csv.tmp").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "table3.csv.tmp").replace(out / "table3.csv")
