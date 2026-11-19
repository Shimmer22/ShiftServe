#!/usr/bin/env python3
"""Host BFCL Multi-Turn worker calling phone_service_host (/v1/chat/completions + /a2o/reset)."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from a2o.workloads.bfcl import dump, load_entries, parse_calls, post, prompt_messages, sandbox_child


def run_case(entry: dict, cfg: dict, agent_base: str, out_dir: Path) -> dict:
    case_dir = out_dir / "tasks" / entry["id"]
    res_path = case_dir / "result.json"
    if res_path.exists():
        cached = json.loads(res_path.read_text())
        if cached.get("trace_complete"):
            return cached
    case_dir.mkdir(parents=True, exist_ok=True)
    post(agent_base, "/a2o/reset", {}, timeout=30)

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    sandbox = ctx.Process(target=sandbox_child, args=(child, entry))
    sandbox.start()

    messages = prompt_messages(entry)
    decoded = []
    allowed = {d["name"] for d in entry["function"]}
    complete = False
    error = None
    calls_log = []

    try:
        if not parent.poll(cfg["tool_timeout_s"]) or not parent.recv().get("ready"):
            raise TimeoutError("sandbox initialization")

        for turn, user_messages in enumerate(entry["question"]):
            if str(turn) in entry["held_docs"]:
                newdocs = entry["held_docs"][str(turn)]
                allowed.update(d["name"] for d in newdocs)
                user_messages = [{
                    "role": "user",
                    "content": json.dumps(newdocs) + "\nI have updated some more functions you can choose from. What about now?",
                }]
            messages.extend(user_messages)
            steps = []
            decoded.append(steps)

            for step in range(cfg["max_steps"]):
                req_body = {
                    "model": cfg["model_id"],
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": cfg["max_tokens"],
                    "seed": cfg["seed"],
                    "omit_verbose_prompt": True,
                }
                t0 = time.monotonic()
                resp = post(agent_base, "/v1/chat/completions", req_body, timeout=cfg["request_timeout_s"])
                wall_ms = (time.monotonic() - t0) * 1000.0

                choice = resp["choices"][0]
                content = (choice["message"].get("content") or "").strip()
                finish_reason = choice.get("finish_reason")
                calls_log.append({
                    "turn": turn,
                    "step": step,
                    "wall_ms": wall_ms,
                    "finish_reason": finish_reason,
                    "usage": resp.get("usage"),
                    "timings": resp.get("timings"),
                })
                if finish_reason == "length":
                    raise ValueError("output token limit exceeded")

                calls = parse_calls(content, allowed)
                messages.append({"role": "assistant", "content": content})
                steps.append([c["canonical"] for c in calls])
                if not calls:
                    break

                parent.send(calls)
                if not parent.poll(cfg["tool_timeout_s"]):
                    raise TimeoutError("tool batch exceeded timeout")
                values = parent.recv()
                messages.append({"role": "tool", "content": json.dumps([v["result"] for v in values])})
            else:
                raise ValueError("maximum steps exceeded")
        complete = True
    except Exception:
        error = traceback.format_exc()
    finally:
        if sandbox.is_alive():
            try:
                parent.send(None)
            except (BrokenPipeError, OSError):
                pass
            sandbox.join(1)
            if sandbox.is_alive():
                sandbox.terminate()
                sandbox.join()
        parent.close()
        child.close()

    dump(case_dir / "decoded.json", decoded)
    dump(case_dir / "calls.json", calls_log)
    quality = {"valid": False, "not_run": True}
    if complete:
        try:
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
            root = Path(cfg["bfcl_root"])
            category = entry["id"].rsplit("_", 1)[0]
            truthfile = root / "bfcl_eval/data/possible_answer" / f"BFCL_v4_{category}.json"
            truth = next(
                json.loads(s)["ground_truth"]
                for s in truthfile.read_text().splitlines()
                if json.loads(s)["id"] == entry["id"]
            )
            quality = multi_turn_checker(decoded, truth, entry, category, f"host_{entry['id']}")
        except Exception:
            quality = {"valid": False, "checker_error": traceback.format_exc()}

    result = {
        "id": entry["id"],
        "category": entry["id"].rsplit("_", 1)[0],
        "trace_complete": complete,
        "quality": quality,
        "error": error,
    }
    dump(res_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--condition-name", required=True)
    p.add_argument("--agent-base", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    bfcl_root = Path(cfg["bfcl_root"])
    sys.path.insert(0, str(bfcl_root))

    all_entries = load_entries(bfcl_root, cfg)
    selected = []
    per_cat = int(cfg.get("cases_per_category", 15))
    for cat in cfg["categories"]:
        choices = [e for e in all_entries if e["id"].rsplit("_", 1)[0] == cat]
        random.Random(cfg["seed"]).shuffle(choices)
        selected.extend(choices[:per_cat])
    if args.limit_tasks > 0:
        selected = selected[: args.limit_tasks]

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, entry in enumerate(selected):
        res = run_case(copy.deepcopy(entry), cfg, args.agent_base.rstrip("/"), args.output)
        results.append(res)
        passed = bool((res.get("quality") or {}).get("valid"))
        print(
            f"[{args.condition_name}] {idx+1}/{len(selected)} {res['id']} "
            f"complete={res['trace_complete']} valid={passed}",
            flush=True,
        )

    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"total": 0, "complete": 0, "valid": 0})
        by_cat[cat]["total"] += 1
        by_cat[cat]["complete"] += int(bool(r["trace_complete"]))
        by_cat[cat]["valid"] += int(bool((r.get("quality") or {}).get("valid")))

    summary = {
        "condition": args.condition_name,
        "total_tasks": len(results),
        "complete": sum(1 for r in results if r["trace_complete"]),
        "valid": sum(1 for r in results if (r.get("quality") or {}).get("valid")),
        "accuracy": (sum(1 for r in results if (r.get("quality") or {}).get("valid")) / len(results)) if results else 0.0,
        "by_category": by_cat,
        "results": results,
    }
    dump(args.output / "worker-summary.json", summary)


if __name__ == "__main__":
    main()
