#!/usr/bin/env python3
"""Teacher-Forced Multi-Turn DroidTask worker calling phone_service_host."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
import time
import traceback
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.profiling.host_accuracy import run_metadata, same_run, select_tasks
from a2o.workloads.bfcl import dump, post
sys.path.insert(0, str(ROOT / "scripts/collect"))
from host_bfcl_turn_level_worker import _summarize_kv

PROTOCOL_VERSION = "droidtask-turn-tf-v4"

_SYSTEM_PROMPT = (
    "You are a smartphone GUI assistant helping a user complete tasks across mobile app screens.\n"
    "At each step you receive the current screen's UI elements with numeric `id` attributes.\n"
    "Previous steps and earlier UI states remain in the conversation history.\n"
    "Respond ONLY with a single JSON object with keys:\n"
    '{"Finished": "Yes" or "No", "id": <integer element id or -1 if finished>, '
    '"action": "tap" or "input" or "N/A", "input_text": "<text to type>" or "N/A"}\n'
    'Rules: If the task is already completed by previous actions, set "Finished": "Yes", "id": -1, '
    '"action": "N/A", "input_text": "N/A". For clicking/checking buttons or items, set "action": "tap" '
    'and "input_text": "N/A". For typing text into an input field, set "action": "input" and '
    '"input_text" to the exact string to type.'
)


def _load_tasks(dataset_path: Path, min_cum_chars: int) -> list[dict]:
    raw = json.loads(dataset_path.read_text())
    items = []
    for app, app_tasks in raw.items():
        for thash, entry in app_tasks.items():
            steps = entry.get("profile") or []
            cum_chars = sum(len(s.get("state", "")) for s in steps) + len(entry.get("task", ""))
            if cum_chars < min_cum_chars:
                continue
            items.append({
                "id": f"{app}_{thash[:12]}",
                "app": app,
                "task_hash": thash,
                "task": entry["task"],
                "steps": steps,
                "cum_chars": cum_chars,
            })
    items.sort(key=lambda x: (-x["cum_chars"], x["app"], x["task_hash"]))
    return items


def _gold_action_json(label: list) -> str:
    gid = int(label[0])
    ginp = label[1]
    if gid == -1:
        obj = {"Finished": "Yes", "id": -1, "action": "N/A", "input_text": "N/A"}
    elif ginp not in ("null", None):
        obj = {"Finished": "No", "id": gid, "action": "input", "input_text": str(ginp)}
    else:
        obj = {"Finished": "No", "id": gid, "action": "tap", "input_text": "N/A"}
    return json.dumps(obj, ensure_ascii=False)


def _parse_prediction(raw_content: str) -> tuple[dict | None, str | None]:
    text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.S).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj, None
            except json.JSONDecodeError:
                continue
    return None, "no valid JSON object in model output"


def _normalize_pred(obj: dict) -> tuple[int | None, str, str]:
    fin_raw = str(obj.get("Finished", obj.get("finished", ""))).strip().lower()
    finished = fin_raw in {"yes", "y", "true"} or "finished" in fin_raw
    if finished:
        return -1, "N/A", "N/A"
    raw_id = obj.get("id")
    if raw_id in ("N/A", None, ""):
        pred_id = -1
    else:
        try:
            pred_id = int(raw_id)
        except (TypeError, ValueError):
            pred_id = None
    act = str(obj.get("action", "")).strip().lower()
    if any(k in act for k in ("tap", "click", "check", "choose", "select", "press")):
        norm_act = "tap"
    elif any(k in act for k in ("input", "type", "enter", "write")):
        norm_act = "input"
    elif act in ("none", "n/a", ""):
        norm_act = "N/A"
    else:
        norm_act = act
    inp = obj.get("input_text", "N/A")
    inp_str = "N/A" if inp is None else str(inp)
    if pred_id == -1:
        return -1, "N/A", "N/A"
    return pred_id, norm_act, inp_str


def _eval_step(label: list, raw_content: str) -> dict:
    gid = int(label[0])
    ginp = label[1]
    expected_act = "N/A" if gid == -1 else ("input" if ginp not in ("null", None) else "tap")
    expected_inp = "N/A" if ginp in ("null", None) else str(ginp)

    obj, err = _parse_prediction(raw_content)
    if obj is None:
        return {
            "valid": False,
            "id_valid": False,
            "error_type": "parse_error",
            "error": err,
            "expected": {"id": gid, "action": expected_act, "input_text": expected_inp},
            "predicted": None,
        }
    pred_id, pred_act, pred_inp = _normalize_pred(obj)
    if gid == -1:
        ok = pred_id == -1
        return {
            "valid": ok,
            "id_valid": ok,
            "error_type": None if ok else "wrong_finish",
            "expected": {"id": gid, "action": expected_act, "input_text": expected_inp},
            "predicted": {"id": pred_id, "action": pred_act, "input_text": pred_inp},
        }

    id_ok = pred_id == gid
    if expected_act == "tap":
        inp_ok = True
        act_ok = pred_act == "tap"
    else:
        inp_ok = pred_inp.strip().lower() == expected_inp.strip().lower()
        act_ok = pred_act == "input"
    return {
        "valid": bool(id_ok and act_ok and inp_ok),
        "id_valid": bool(id_ok and inp_ok),
        "error_type": None if (id_ok and act_ok and inp_ok) else (
            "wrong_id" if not id_ok else ("wrong_input" if not inp_ok else "wrong_action")
        ),
        "expected": {"id": gid, "action": expected_act, "input_text": expected_inp},
        "predicted": {"id": pred_id, "action": pred_act, "input_text": pred_inp},
    }


def run_task(task: dict, cfg: dict, agent_base: str, out_dir: Path, meta: dict) -> dict:
    tdir = out_dir / "tasks" / task["id"]
    res_path = tdir / "result.json"
    if res_path.exists():
        cached = json.loads(res_path.read_text())
        if cached.get("trace_complete") and same_run(cached, meta):
            return cached
    tdir.mkdir(parents=True, exist_ok=True)

    enable_thinking = bool(cfg.get("enable_thinking", False))
    steps = task["steps"]
    gold_prefixes: list[list[dict]] = []
    # Pin task goal in system prompt so recent/KV compaction never evicts the objective.
    sys_msg = f"{_SYSTEM_PROMPT}\n\nApp: {task['app']}\nTask: {task['task']}"
    running_msgs = [{"role": "system", "content": sys_msg}]
    for sidx, step in enumerate(steps):
        uc = f"Step {sidx} UI state:\n{step['state']}"
        running_msgs.append({"role": "user", "content": uc})
        gold_prefixes.append(copy.deepcopy(running_msgs))
        running_msgs.append({"role": "assistant", "content": _gold_action_json(step["label"])})

    turn_results = []
    all_calls = []
    try:
        post(agent_base, "/a2o/reset", {}, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    need_replay = False
    for sidx, step in enumerate(steps):
        t_calls = []
        turn_complete = False
        turn_error = None
        content = ""
        try:
            if need_replay:
                try:
                    post(agent_base, "/a2o/reset", {}, timeout=30)
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                for k in range(sidx):
                    pre_body = {
                        "model": cfg["model_id"],
                        "messages": gold_prefixes[k],
                        "temperature": 0.0,
                        "max_tokens": 0,
                        "seed": cfg["seed"],
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    }
                    pre_resp = post(agent_base, "/v1/chat/completions", pre_body, timeout=cfg["request_timeout_s"])
                    t_calls.append({"eval_turn": sidx, "phase": f"prefix_{k}", "timings": pre_resp.get("timings"), "usage": pre_resp.get("usage")})
                need_replay = False

            req_body = {
                "model": cfg["model_id"],
                "messages": gold_prefixes[sidx],
                "temperature": 0.0,
                "max_tokens": cfg["max_tokens"],
                "seed": cfg["seed"],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            t0 = time.monotonic()
            resp = post(agent_base, "/v1/chat/completions", req_body, timeout=cfg["request_timeout_s"])
            wall_ms = (time.monotonic() - t0) * 1000.0
            choice = resp["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = (choice["message"].get("content") or "").strip()
            t_calls.append({
                "eval_turn": sidx,
                "phase": "eval",
                "wall_ms": wall_ms,
                "finish_reason": finish_reason,
                "usage": resp.get("usage"),
                "timings": resp.get("timings"),
            })
            if finish_reason == "length":
                turn_error = "ValueError: output token limit exceeded"
                eval_res = {"valid": False, "id_valid": False, "error_type": "runtime_error"}
            else:
                eval_res = _eval_step(step["label"], content)
                turn_complete = True
        except Exception:
            turn_error = traceback.format_exc()
            eval_res = {"valid": False, "id_valid": False, "error_type": "runtime_error"}
            need_replay = True

        all_calls.extend(t_calls)
        turn_results.append({
            "turn": sidx,
            "complete": turn_complete,
            "valid": bool(turn_complete and eval_res["valid"]),
            "id_valid": bool(turn_complete and eval_res.get("id_valid", False)),
            "eval": eval_res,
            "prompt_tokens": (t_calls[-1].get("usage") or {}).get("prompt_tokens") if t_calls else None,
            "completion_tokens": (t_calls[-1].get("usage") or {}).get("completion_tokens") if t_calls else None,
            "error": turn_error,
            "raw_content": content,
        })

    n_turns = len(turn_results)
    n_complete = sum(1 for r in turn_results if r["complete"])
    n_valid = sum(1 for r in turn_results if r["valid"])
    n_id_valid = sum(1 for r in turn_results if r["id_valid"])
    res = {
        "id": task["id"],
        "app": task["app"],
        "task_hash": task["task_hash"],
        "task": task["task"],
        "cum_chars": task["cum_chars"],
        **meta,
        "trace_complete": n_complete == n_turns,
        "total_turns": n_turns,
        "complete_turns": n_complete,
        "valid_turns": n_valid,
        "id_valid_turns": n_id_valid,
        "turn_accuracy": n_valid / n_turns if n_turns else 0.0,
        "id_turn_accuracy": n_id_valid / n_turns if n_turns else 0.0,
        "all_turns_valid": n_valid == n_turns,
        "max_prompt_tokens": max((r["prompt_tokens"] or 0) for r in turn_results),
        "turn_results": turn_results,
        "kv_stats": _summarize_kv(all_calls),
    }
    dump(tdir / "calls.json", all_calls)
    dump(res_path, res)
    return res


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
    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    tasks = _load_tasks(dataset_path, int(cfg.get("min_cum_chars", 0)))
    selected = select_tasks(tasks, lambda t: t["id"], args.task_id, args.limit_tasks)

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, task in enumerate(selected):
        res = run_task(task, cfg, args.agent_base.rstrip("/"), args.output, meta)
        results.append(res)
        ks = res["kv_stats"]
        print(
            f"[{args.condition_name}] {idx+1}/{len(selected)} {res['id']} "
            f"turns={res['valid_turns']}/{res['total_turns']} id_ok={res['id_valid_turns']}/{res['total_turns']} "
            f"max_ptok={res['max_prompt_tokens']} evict={ks['evicted_tokens']}/{ks['logical_tokens']}",
            flush=True,
        )

    tot_turns = sum(r["total_turns"] for r in results)
    tot_complete = sum(r["complete_turns"] for r in results)
    tot_valid = sum(r["valid_turns"] for r in results)
    tot_id_valid = sum(r["id_valid_turns"] for r in results)
    summary = {
        **meta,
        "selected_task_ids": [t["id"] for t in selected],
        "total_tasks": len(results),
        "total_turns": tot_turns,
        "complete_turns": tot_complete,
        "valid_turns": tot_valid,
        "id_valid_turns": tot_id_valid,
        "turn_accuracy": tot_valid / tot_turns if tot_turns else 0.0,
        "id_turn_accuracy": tot_id_valid / tot_turns if tot_turns else 0.0,
        "all_turns_valid_tasks": sum(1 for r in results if r["all_turns_valid"]),
        "results": results,
    }
    dump(args.output / "worker-summary.json", summary)


if __name__ == "__main__":
    main()
