#!/usr/bin/env python3
"""Analyze multi-turn tau2-bench E2E quality and latency for Mechanism 2 adaptive KV reduce."""
import argparse
import json
from pathlib import Path
import statistics


def _mean(vals):
    return statistics.mean(vals) if vals else 0.0


def summarize_condition(cond_dir: Path) -> dict:
    worker = json.loads((cond_dir / "worker-summary.json").read_text())
    mem_path = cond_dir / "phone-memory.json"
    mem = json.loads(mem_path.read_text()) if mem_path.exists() else {}

    tasks_summary = []
    all_agent_turns = []
    long_turns = []
    short_turns = []

    for task_rec in worker["results"]:
        agent_calls = [c for c in task_rec.get("calls", []) if c.get("is_agent")]
        all_agent_turns.extend(agent_calls)

        admitted = 0
        rejected = 0
        htp_turns = 0
        cpu_turns = 0
        prompt_ms_sum = 0.0
        decode_ms_sum = 0.0
        restore_ms_sum = 0.0
        handoff_ms_sum = 0.0
        wait_ms_sum = 0.0
        kv_reduce_ms_sum = 0.0
        agent_total_ms_sum = 0.0
        agent_http_ms_sum = 0.0
        pred_tokens_sum = 0
        max_h_anchor = 0
        max_h_eff = 0
        max_prompt_tokens = 0
        keep_targets = []

        for c in agent_calls:
            t = c.get("timings") or {}
            u = c.get("usage") or {}
            pt = int(u.get("prompt_tokens", 0))
            max_prompt_tokens = max(max_prompt_tokens, pt)
            max_h_anchor = max(max_h_anchor, int(t.get("H_anchor", 0)))
            max_h_eff = max(max_h_eff, int(t.get("H_effective", 0)))
            prompt_ms_sum += float(t.get("prompt_ms", 0.0))
            decode_ms_sum += float(t.get("predicted_ms", 0.0))
            restore_ms_sum += float(t.get("restore_ms", 0.0))
            handoff_ms_sum += float(t.get("handoff_ms", 0.0))
            wait_ms_sum += float(t.get("wait_ms", 0.0))
            kv_reduce_ms_sum += float(t.get("kv_reduce_ms", 0.0))
            if t.get("route") == "htp":
                htp_turns += 1
            else:
                cpu_turns += 1
            agent_total_ms_sum += float(t.get("total_ms", 0.0))
            agent_http_ms_sum += float(c.get("wall_ms", 0.0))
            pred_tokens_sum += int(t.get("predicted_n", 0))

            dec = t.get("kv_cost_decision")
            kvr = t.get("kv_reduce")
            if isinstance(dec, dict):
                if dec.get("admit"):
                    admitted += 1
                    keep_targets.append(int(dec.get("keep_tokens", 0)))
                else:
                    rejected += 1
            elif isinstance(kvr, dict):
                admitted += 1

            row_entry = {
                "prompt_tokens": pt,
                "H_anchor": t.get("H_anchor", 0),
                "H_effective": t.get("H_effective", 0),
                "P_evaluated": t.get("P_evaluated", 0),
                "predicted_n": t.get("predicted_n", 0),
                "prompt_ms": t.get("prompt_ms", 0.0),
                "predicted_ms": t.get("predicted_ms", 0.0),
                "restore_ms": t.get("restore_ms", 0.0),
                "handoff_ms": t.get("handoff_ms", 0.0),
                "kv_reduce_ms": t.get("kv_reduce_ms", 0.0),
                "total_ms": t.get("total_ms", 0.0),
                "route": t.get("route", "cpu"),
                "admitted": bool((dec or {}).get("admit")) if isinstance(dec, dict) else bool(kvr),
            }
            if pt >= 4096:
                long_turns.append(row_entry)
            else:
                short_turns.append(row_entry)

        tasks_summary.append({
            "domain": task_rec["domain"],
            "task_id": task_rec["task_id"],
            "complete": task_rec["complete"],
            "reward": task_rec["reward"],
            "termination_reason": task_rec.get("termination_reason"),
            "duration_s": task_rec["duration_s"],
            "agent_turns": len(agent_calls),
            "tool_calls": task_rec.get("tool_calls", 0),
            "max_prompt_tokens": max_prompt_tokens,
            "max_H_anchor": max_h_anchor,
            "max_H_effective": max_h_eff,
            "kv_admitted_turns": admitted,
            "kv_rejected_turns": rejected,
            "htp_routed_turns": htp_turns,
            "cpu_routed_turns": cpu_turns,
            "keep_targets": keep_targets,
            "agent_wall_s": agent_http_ms_sum / 1000.0,
            "agent_service_s": agent_total_ms_sum / 1000.0,
            "prompt_s": prompt_ms_sum / 1000.0,
            "decode_s": decode_ms_sum / 1000.0,
            "restore_s": restore_ms_sum / 1000.0,
            "handoff_s": handoff_ms_sum / 1000.0,
            "wait_s": wait_ms_sum / 1000.0,
            "kv_reduce_s": kv_reduce_ms_sum / 1000.0,
            "predicted_tokens": pred_tokens_sum,
            "decode_ms_per_tok": (decode_ms_sum / pred_tokens_sum) if pred_tokens_sum else 0.0,
            "error": task_rec.get("error"),
        })

    def _agg_turns(turns):
        n_tok = sum(r["predicted_n"] for r in turns)
        dec_ms = sum(r["predicted_ms"] for r in turns)
        return {
            "count": len(turns),
            "mean_prompt_tokens": _mean([r["prompt_tokens"] for r in turns]),
            "mean_restore_ms": _mean([r["restore_ms"] for r in turns]),
            "mean_handoff_ms": _mean([r["handoff_ms"] for r in turns]),
            "mean_prompt_ms": _mean([r["prompt_ms"] for r in turns]),
            "mean_kv_reduce_ms": _mean([r["kv_reduce_ms"] for r in turns]),
            "mean_total_ms": _mean([r["total_ms"] for r in turns]),
            "decode_ms_per_tok": (dec_ms / n_tok) if n_tok else 0.0,
            "htp_routed_count": sum(1 for r in turns if r["route"] == "htp"),
            "admitted_count": sum(1 for r in turns if r["admitted"]),
        }

    total_pred_toks = sum(t["predicted_tokens"] for t in tasks_summary)
    total_decode_s = sum(t["decode_s"] for t in tasks_summary)
    return {
        "condition": worker["condition"],
        "total_tasks": worker["total_tasks"],
        "completed_tasks": worker["completed_tasks"],
        "reward_sum": worker["reward_sum"],
        "reward_mean": worker["reward_mean"],
        "total_duration_s": sum(t["duration_s"] for t in tasks_summary),
        "total_agent_service_s": sum(t["agent_service_s"] for t in tasks_summary),
        "total_prompt_s": sum(t["prompt_s"] for t in tasks_summary),
        "total_decode_s": total_decode_s,
        "total_restore_s": sum(t["restore_s"] for t in tasks_summary),
        "total_handoff_s": sum(t["handoff_s"] for t in tasks_summary),
        "total_wait_s": sum(t["wait_s"] for t in tasks_summary),
        "total_kv_reduce_s": sum(t["kv_reduce_s"] for t in tasks_summary),
        "htp_routed_turns": sum(t["htp_routed_turns"] for t in tasks_summary),
        "cpu_routed_turns": sum(t["cpu_routed_turns"] for t in tasks_summary),
        "total_predicted_tokens": total_pred_toks,
        "overall_decode_ms_per_tok": (total_decode_s * 1000.0 / total_pred_toks) if total_pred_toks else 0.0,
        "short_turns_lt4k": _agg_turns(short_turns),
        "long_turns_ge4k": _agg_turns(long_turns),
        "phone_memory": mem,
        "tasks": tasks_summary,
    }


def summarize_campaign(out_dir: Path, cond_names: list[str]) -> dict:
    conditions = {}
    for name in cond_names:
        cdir = out_dir / "conditions" / name
        if (cdir / "worker-summary.json").exists():
            conditions[name] = summarize_condition(cdir)
    base = conditions.get(cond_names[0]) if cond_names else None
    comparisons = {}
    if base:
        for name, c in conditions.items():
            if name == base["condition"]:
                continue
            comparisons[f"{name}_vs_{base['condition']}"] = {
                "reward_diff": c["reward_sum"] - base["reward_sum"],
                "agent_service_speedup": (base["total_agent_service_s"] / c["total_agent_service_s"]) if c["total_agent_service_s"] else 0.0,
                "decode_speedup": (base["overall_decode_ms_per_tok"] / c["overall_decode_ms_per_tok"]) if c["overall_decode_ms_per_tok"] else 0.0,
                "long_turn_decode_speedup": (base["long_turns_ge4k"]["decode_ms_per_tok"] / c["long_turns_ge4k"]["decode_ms_per_tok"]) if c["long_turns_ge4k"]["decode_ms_per_tok"] else 0.0,
                "long_turn_restore_speedup": (base["long_turns_ge4k"]["mean_restore_ms"] / c["long_turns_ge4k"]["mean_restore_ms"]) if c["long_turns_ge4k"]["mean_restore_ms"] else 0.0,
            }
    return {"conditions": conditions, "comparisons": comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", default=["cpu_baseline", "cpu_adaptive_full", "cpu_adaptive_compact"])
    args = p.parse_args()
    rep = summarize_campaign(args.input.resolve(), args.conditions)
    (args.input.resolve() / "summary.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
