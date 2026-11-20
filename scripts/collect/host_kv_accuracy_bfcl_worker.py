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
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from a2o.profiling.host_accuracy import run_metadata, same_run, select_tasks
from a2o.workloads.bfcl import dump, load_entries, post, sandbox_child


def _fix_schema(obj):
    if isinstance(obj, dict):
        t = obj.get("type")
        if t == "dict":
            obj["type"] = "object"
        elif t == "float":
            obj["type"] = "number"
        elif t == "tuple":
            obj["type"] = "array"
        for v in obj.values():
            _fix_schema(v)
    elif isinstance(obj, list):
        for v in obj:
            _fix_schema(v)


def _to_oai_tools(docs: list[dict]) -> list[dict]:
    out = []
    for d in docs:
        params = copy.deepcopy(d.get("parameters") or {"type": "object", "properties": {}})
        _fix_schema(params)
        out.append({
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": params,
            },
        })
    return out


_FC_SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. "
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
    "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. "
    "At each turn, try your best to complete the tasks requested by the user within the current turn."
)


PROTOCOL_VERSION = "bfcl-fc-v2"


def run_case(entry: dict, cfg: dict, agent_base: str, out_dir: Path, meta: dict) -> dict:
    case_dir = out_dir / "tasks" / entry["id"]
    res_path = case_dir / "result.json"
    if res_path.exists():
        cached = json.loads(res_path.read_text())
        if (
            cached.get("trace_complete")
            and same_run(cached, meta)
        ):
            return cached
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        post(agent_base, "/a2o/reset", {}, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    sandbox = ctx.Process(target=sandbox_child, args=(child, entry))
    sandbox.start()

    messages = [{"role": "system", "content": _FC_SYSTEM_PROMPT}]
    tools = _to_oai_tools(entry["function"])
    decoded = []
    allowed = {d["name"] for d in entry["function"]}
    enable_thinking = bool(cfg.get("enable_thinking", False))
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
                tools.extend(_to_oai_tools(newdocs))
                user_messages = [{
                    "role": "user",
                    "content": "I have updated some more functions you can choose from. What about now?",
                }]
            messages.extend(user_messages)
            steps = []
            decoded.append(steps)

            for step in range(cfg["max_steps"]):
                req_body = {
                    "model": cfg["model_id"],
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0.0,
                    "max_tokens": cfg["max_tokens"],
                    "seed": cfg["seed"],
                    "omit_verbose_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                }
                t0 = time.monotonic()
                resp = post(agent_base, "/v1/chat/completions", req_body, timeout=cfg["request_timeout_s"])
                wall_ms = (time.monotonic() - t0) * 1000.0

                choice = resp["choices"][0]
                msg = choice["message"]
                content = (msg.get("content") or "").strip()
                tcs = msg.get("tool_calls") or []
                finish_reason = choice.get("finish_reason")
                calls_log.append({
                    "turn": turn,
                    "step": step,
                    "wall_ms": wall_ms,
                    "finish_reason": finish_reason,
                    "message": msg,
                    "usage": resp.get("usage"),
                    "timings": resp.get("timings"),
                })
                if finish_reason == "length":
                    raise ValueError("output token limit exceeded")

                if not tcs:
                    messages.append({"role": "assistant", "content": content})
                    break

                calls = []
                malformed_errors = []
                for idx, tc in enumerate(tcs):
                    cid = tc.get("id") or f"call_{turn}_{step}_{idx}"
                    tc["id"] = cid
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    parse_err = None
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError as exc:
                        args = None
                        parse_err = f"invalid JSON arguments for {name}: {exc}"
                    if parse_err is None and not isinstance(args, dict):
                        parse_err = f"non-object arguments for {name}: {type(args).__name__}"
                        args = None
                    if parse_err is not None:
                        malformed_errors.append(parse_err)
                        calls.append({"name": name, "args": [], "kwargs": None, "canonical": "", "id": cid, "error": parse_err})
                    else:
                        canon = f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
                        calls.append({"name": name, "args": [], "kwargs": args, "canonical": canon, "id": cid, "error": None})

                if malformed_errors:
                    calls_log[-1]["malformed_tool_calls"] = malformed_errors
                messages.append({"role": "assistant", "content": content, "tool_calls": tcs})
                valid_calls = [c for c in calls if c["error"] is None and c["name"] in allowed]
                if valid_calls:
                    steps.append([c["canonical"] for c in valid_calls])
                    parent.send(valid_calls)
                    if not parent.poll(cfg["tool_timeout_s"]):
                        raise TimeoutError("tool batch exceeded timeout")
                    values = parent.recv()
                else:
                    values = []

                vi = 0
                for c in calls:
                    if c["error"] is not None:
                        res_str = f"Error: {c['error']}"
                    elif c["name"] in allowed:
                        res_str = values[vi]["result"]
                        vi += 1
                    else:
                        res_str = f"Error: unknown function {c['name']}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "name": c["name"],
                        "content": res_str,
                    })
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
        **meta,
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
    p.add_argument("--task-id", action="append", default=[])
    p.add_argument("--runtime-identity", type=json.loads, default={})
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    meta = run_metadata(PROTOCOL_VERSION, cfg, args.condition_name, args.limit_tasks,
                        args.task_id, args.runtime_identity)
    bfcl_root = Path(cfg["bfcl_root"])
    sys.path.insert(0, str(bfcl_root))

    all_entries = load_entries(bfcl_root, cfg)
    selected = []
    per_cat = int(cfg.get("cases_per_category", 15))
    for cat in cfg["categories"]:
        choices = [e for e in all_entries if e["id"].rsplit("_", 1)[0] == cat]
        random.Random(cfg["seed"]).shuffle(choices)
        selected.extend(choices[:per_cat])
    selected = select_tasks(selected, lambda entry: entry["id"], args.task_id, args.limit_tasks)

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, entry in enumerate(selected):
        res = run_case(copy.deepcopy(entry), cfg, args.agent_base.rstrip("/"), args.output, meta)
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
        **meta,
        "selected_task_ids": [entry["id"] for entry in selected],
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
