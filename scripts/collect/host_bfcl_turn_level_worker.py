#!/usr/bin/env python3
"""Teacher-Forced Turn-Level Host BFCL worker calling phone_service_host."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time
import traceback
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

from a2o.profiling.host_accuracy import run_metadata, same_run, select_tasks
from a2o.workloads.bfcl import dump, load_entries, post, sandbox_child
from host_kv_accuracy_bfcl_worker import _FC_SYSTEM_PROMPT, _to_oai_tools

PROTOCOL_VERSION = "bfcl-turn-tf-v1"


def _parse_gold_calls(gt_calls: list[str], param_map: dict[str, list[str]], turn: int) -> tuple[list[dict], list[dict]]:
    sandbox_calls = []
    oai_tool_calls = []
    for idx, call_str in enumerate(gt_calls):
        node = ast.parse(call_str, mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"Unexpected gold call AST: {call_str}")
        name = node.func.id
        args = [ast.literal_eval(a) for a in node.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        full_kwargs = dict(kwargs)
        if args:
            pnames = param_map.get(name, [])
            if len(pnames) < len(args):
                raise ValueError(f"Schema params shorter than positional args for {name}: {pnames} vs {args}")
            for pname, val in zip(pnames, args):
                full_kwargs[pname] = val
        cid = f"gold_{turn}_{idx}"
        sandbox_calls.append({
            "name": name,
            "args": args,
            "kwargs": kwargs,
            "canonical": call_str,
            "id": cid,
        })
        oai_tool_calls.append({
            "id": cid,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(full_kwargs, ensure_ascii=False),
            },
        })
    return sandbox_calls, oai_tool_calls


def _build_gold_turns(entry: dict, ground_truth: list[list[str]], timeout_s: float) -> tuple[list[dict], dict]:
    param_map: dict[str, list[str]] = {}
    for fn in entry["function"]:
        param_map[fn["name"]] = list((fn.get("parameters", {}).get("properties") or {}).keys())
    for flist in entry.get("held_docs", {}).values():
        for fn in flist:
            param_map[fn["name"]] = list((fn.get("parameters", {}).get("properties") or {}).keys())

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    sandbox = ctx.Process(target=sandbox_child, args=(child, entry))
    sandbox.start()
    turns_data = []
    messages = [{"role": "system", "content": _FC_SYSTEM_PROMPT}]
    tools = _to_oai_tools(entry["function"])
    allowed = {d["name"] for d in entry["function"]}

    try:
        if not parent.poll(timeout_s) or not parent.recv().get("ready"):
            raise TimeoutError("gold sandbox initialization")
        for turn, user_messages in enumerate(entry["question"]):
            if str(turn) in entry["held_docs"]:
                newdocs = entry["held_docs"][str(turn)]
                allowed.update(d["name"] for d in newdocs)
                tools.extend(_to_oai_tools(newdocs))
                user_messages = [{
                    "role": "user",
                    "content": "I have updated some more functions you can choose from. What about now?",
                }]
            messages.extend(copy.deepcopy(user_messages))
            gt_calls = ground_truth[turn]
            sandbox_calls, oai_tcs = _parse_gold_calls(gt_calls, param_map, turn)
            turns_data.append({
                "turn": turn,
                "messages_before": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "allowed": set(allowed),
                "gold_sandbox_calls": sandbox_calls,
                "gold_gt_calls": gt_calls,
            })
            if sandbox_calls:
                parent.send(sandbox_calls)
                if not parent.poll(timeout_s):
                    raise TimeoutError("gold sandbox execution")
                values = parent.recv()
                messages.append({"role": "assistant", "content": "", "tool_calls": oai_tcs})
                for sc, val in zip(sandbox_calls, values):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": sc["id"],
                        "name": sc["name"],
                        "content": val["result"],
                    })
                messages.append({"role": "assistant", "content": "Done."})
            else:
                messages.append({
                    "role": "assistant",
                    "content": "I cannot fulfill this request with the available functions and provided parameters.",
                })
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

    return turns_data, param_map


def _eval_single_turn(entry: dict, ground_truth: list[list[str]], turn: int, steps: list[list[str]]) -> dict:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import response_checker, state_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )

    gt_turn = ground_truth[turn]
    if not gt_turn:
        if is_empty_execute_response(steps):
            return {"valid": True}
        return {
            "valid": False,
            "error_type": "multi_turn:irrelevance_error:decoder_success",
            "model_decoded": steps,
        }

    if not steps or is_empty_execute_response(steps):
        return {
            "valid": False,
            "error_type": "multi_turn:empty_turn_model_response",
        }

    cat = entry["id"].rsplit("_", 1)[0]
    lc = "long_context" in cat or "composite" in cat
    initial_config = entry["initial_config"]
    involved_classes = entry["involved_classes"]
    tid = entry["id"]

    # Execute gold turns 0..turn-1 to get exact pre-turn ground truth execution results.
    gold_exec_before: list = []
    for k in range(turn):
        if ground_truth[k]:
            se, _ = execute_multi_turn_func_call(
                func_call_list=ground_truth[k],
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=f"tf_gt_{tid}_t{turn}",
                test_entry_id=tid,
                long_context=lc,
                is_evaL_run=True,
            )
            gold_exec_before.extend(se)

    # Execute gold turn t
    gt_exec_t, gt_inst_t = execute_multi_turn_func_call(
        func_call_list=gt_turn,
        initial_config=initial_config,
        involved_classes=involved_classes,
        model_name=f"tf_gt_{tid}_t{turn}",
        test_entry_id=tid,
        long_context=lc,
        is_evaL_run=True,
    )

    # Execute model steps on top of gold prefix 0..turn-1 in separate model namespace
    m_inst_t = {}
    for k in range(turn):
        if ground_truth[k]:
            _, m_inst_t = execute_multi_turn_func_call(
                func_call_list=ground_truth[k],
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=f"tf_m_{tid}_t{turn}",
                test_entry_id=tid,
                long_context=lc,
                is_evaL_run=True,
            )

    m_exec_t = []
    for step_calls in steps:
        se, m_inst_t = execute_multi_turn_func_call(
            func_call_list=step_calls,
            initial_config=initial_config,
            involved_classes=involved_classes,
            model_name=f"tf_m_{tid}_t{turn}",
            test_entry_id=tid,
            long_context=lc,
            is_evaL_run=True,
        )
        m_exec_t.extend(se)

    sc = state_checker(m_inst_t, gt_inst_t)
    if not sc["valid"]:
        return sc
    rc = response_checker(gold_exec_before + m_exec_t, gt_exec_t, turn)
    if not rc["valid"]:
        return rc
    return {"valid": True}


def run_case(entry: dict, truth: list[list[str]], cfg: dict, agent_base: str, out_dir: Path, meta: dict) -> dict:
    case_dir = out_dir / "tasks" / entry["id"]
    res_path = case_dir / "result.json"
    if res_path.exists():
        cached = json.loads(res_path.read_text())
        if cached.get("trace_complete") and same_run(cached, meta):
            return cached
    case_dir.mkdir(parents=True, exist_ok=True)

    turns_data, _ = _build_gold_turns(entry, truth, cfg["tool_timeout_s"])
    enable_thinking = bool(cfg.get("enable_thinking", False))
    turn_results = []
    all_calls_log = []

    for tinfo in turns_data:
        turn = tinfo["turn"]
        try:
            post(agent_base, "/a2o/reset", {}, timeout=30)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

        # Replay gold turns 0..turn-1 with max_tokens=0 so M2 compacts cross-turn KV naturally.
        prefix_calls = []
        steps: list[list[str]] = []
        turn_calls_log = []
        turn_complete = False
        turn_error = None

        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        sandbox = ctx.Process(target=sandbox_child, args=(child, entry))
        sandbox.start()
        messages = copy.deepcopy(tinfo["messages_before"])
        tools = tinfo["tools"]
        allowed = tinfo["allowed"]

        try:
            for k in range(turn):
                kinfo = turns_data[k]
                pre_body = {
                    "model": cfg["model_id"],
                    "messages": kinfo["messages_before"],
                    "tools": kinfo["tools"],
                    "tool_choice": "auto",
                    "temperature": 0.0,
                    "max_tokens": 0,
                    "seed": cfg["seed"],
                    "omit_verbose_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                }
                pre_resp = post(agent_base, "/v1/chat/completions", pre_body, timeout=cfg["request_timeout_s"])
                prefix_calls.append({"turn": k, "timings": pre_resp.get("timings")})

            if not parent.poll(cfg["tool_timeout_s"]) or not parent.recv().get("ready"):
                raise TimeoutError("turn sandbox initialization")
            for k in range(turn):
                gcalls = turns_data[k]["gold_sandbox_calls"]
                if gcalls:
                    parent.send(gcalls)
                    if not parent.poll(cfg["tool_timeout_s"]):
                        raise TimeoutError("turn sandbox prefix replay")
                    parent.recv()

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
                turn_calls_log.append({
                    "eval_turn": turn,
                    "step": step,
                    "wall_ms": wall_ms,
                    "finish_reason": finish_reason,
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
                    turn_calls_log[-1]["malformed_tool_calls"] = malformed_errors
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
            turn_complete = True
        except Exception:
            turn_error = traceback.format_exc()
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

        quality = {"valid": False, "not_run": True}
        if turn_complete:
            try:
                quality = _eval_single_turn(entry, truth, turn, steps)
            except Exception:
                quality = {"valid": False, "checker_error": traceback.format_exc()}

        turn_results.append({
            "turn": turn,
            "is_empty_gt": len(truth[turn]) == 0,
            "complete": turn_complete,
            "valid": bool(quality.get("valid")),
            "quality": quality,
            "error": turn_error,
            "decoded": steps,
            "prefix_calls": prefix_calls,
            "eval_calls": turn_calls_log,
        })
        all_calls_log.extend(prefix_calls)
        all_calls_log.extend(turn_calls_log)

    dump(case_dir / "turns.json", turn_results)
    n_turns = len(turn_results)
    n_complete = sum(1 for r in turn_results if r["complete"])
    n_valid = sum(1 for r in turn_results if r["valid"])
    result = {
        "id": entry["id"],
        "category": entry["id"].rsplit("_", 1)[0],
        **meta,
        "trace_complete": n_complete == n_turns,
        "total_turns": n_turns,
        "complete_turns": n_complete,
        "valid_turns": n_valid,
        "turn_accuracy": (n_valid / n_turns) if n_turns else 0.0,
        "all_turns_valid": n_valid == n_turns,
        "turn_results": [{k: v for k, v in r.items() if k not in ("prefix_calls", "eval_calls")} for r in turn_results],
        "kv_stats": _summarize_kv(all_calls_log),
    }
    dump(res_path, result)
    return result


def _summarize_kv(calls: list[dict]) -> dict:
    compact_turns = prior_turns = recent_turns = reuse_turns = 0
    before_toks = logical_toks = kept_toks = evicted_toks = 0
    for c in calls:
        t = c.get("timings") or {}
        if not t:
            continue
        if t.get("H_anchor", 0) > t.get("H_effective", 0):
            reuse_turns += 1
        kv = t.get("kv_reduce")
        if kv and kv.get("evicted_count", 0) > 0:
            compact_turns += 1
            prior_turns += int(kv.get("policy") == "prefill_prior" and kv.get("prior_ranked_tokens", 0) > 0)
            recent_turns += int(kv.get("policy") == "recent")
            before_toks += int(kv.get("before_pos_count", 0))
            logical_toks += int(t.get("H_anchor", 0)) + int(t.get("P_evaluated", 0))
            kept_toks += int(kv.get("kept_count", 0))
            evicted_toks += int(kv.get("evicted_count", 0))
    return {
        "compact_turns": compact_turns,
        "prior_turns": prior_turns,
        "recent_turns": recent_turns,
        "reuse_turns": reuse_turns,
        "before_tokens": before_toks,
        "logical_tokens": logical_toks,
        "kept_tokens": kept_toks,
        "evicted_tokens": evicted_toks,
    }


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
    truths = {}
    for cat in cfg["categories"]:
        tfile = bfcl_root / "bfcl_eval/data/possible_answer" / f"BFCL_v4_{cat}.json"
        for line in tfile.read_text().splitlines():
            row = json.loads(line)
            truths[row["id"]] = row["ground_truth"]

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
        res = run_case(copy.deepcopy(entry), truths[entry["id"]], cfg,
                       args.agent_base.rstrip("/"), args.output, meta)
        results.append(res)
        print(
            f"[{args.condition_name}] {idx+1}/{len(selected)} {res['id']} "
            f"turns={res['valid_turns']}/{res['total_turns']} complete={res['complete_turns']}/{res['total_turns']}",
            flush=True,
        )

    tot_turns = sum(r["total_turns"] for r in results)
    tot_complete = sum(r["complete_turns"] for r in results)
    tot_valid = sum(r["valid_turns"] for r in results)
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"tasks": 0, "turns": 0, "complete_turns": 0, "valid_turns": 0, "all_turns_valid_tasks": 0})
        by_cat[cat]["tasks"] += 1
        by_cat[cat]["turns"] += r["total_turns"]
        by_cat[cat]["complete_turns"] += r["complete_turns"]
        by_cat[cat]["valid_turns"] += r["valid_turns"]
        by_cat[cat]["all_turns_valid_tasks"] += int(r["all_turns_valid"])
    for st in by_cat.values():
        st["turn_accuracy"] = (st["valid_turns"] / st["turns"]) if st["turns"] else 0.0

    summary = {
        **meta,
        "selected_task_ids": [entry["id"] for entry in selected],
        "total_tasks": len(results),
        "total_turns": tot_turns,
        "complete_turns": tot_complete,
        "valid_turns": tot_valid,
        "turn_accuracy": (tot_valid / tot_turns) if tot_turns else 0.0,
        "all_turns_valid_tasks": sum(1 for r in results if r["all_turns_valid"]),
        "by_category": by_cat,
        "results": results,
    }
    dump(args.output / "worker-summary.json", summary)


if __name__ == "__main__":
    main()
