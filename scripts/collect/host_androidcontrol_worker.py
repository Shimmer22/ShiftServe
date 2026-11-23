#!/usr/bin/env python3
"""Teacher-Forced Multi-Turn AndroidControl worker calling phone_service_host."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import random
import struct
import sys
import time
import traceback
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

from a2o.profiling.host_accuracy import run_metadata, same_run, select_tasks
from a2o.workloads.bfcl import dump, post
from host_bfcl_turn_level_worker import _summarize_kv
from host_droidtask_worker import _parse_prediction

PROTOCOL_VERSION = "androidcontrol-turn-tf-v2"

_SYSTEM_PROMPT = (
    "You are a smartphone GUI assistant helping a user complete tasks across real Android apps.\n"
    "At each step you receive the current screen's visible UI tree elements with numeric `[id]` indices "
    "and `box=[left,top,right,bottom]` coordinates.\n"
    "Previous steps and earlier UI trees remain in the conversation history.\n"
    "Respond ONLY with a single JSON object matching one of these action schemas:\n"
    '- Click element: {"action_type": "click", "id": <integer element id>}\n'
    '- Long press element: {"action_type": "long_press", "id": <integer element id>}\n'
    '- Type text (after input field is focused): {"action_type": "input_text", "text": "<text>"}\n'
    '- Open app: {"action_type": "open_app", "app_name": "<app name>"}\n'
    '- Scroll screen: {"action_type": "scroll", "direction": "up"|"down"|"left"|"right"}\n'
    '- Navigate back: {"action_type": "navigate_back"}\n'
    '- Navigate home: {"action_type": "navigate_home"}\n'
    '- Wait: {"action_type": "wait"}'
)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    res = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, pos
        shift += 7
    raise EOFError("truncated varint")


def _parse_pb(buf: bytes) -> list[tuple[int, int, int | bytes]]:
    pos = 0
    out = []
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
        elif wt == 2:
            ln, pos = _read_varint(buf, pos)
            val = buf[pos:pos + ln]
            pos += ln
        elif wt == 5:
            val = buf[pos:pos + 4]
            pos += 4
        elif wt == 1:
            val = buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported wire_type={wt}")
        out.append((fn, wt, val))
    return out


def _dec_varint_list(buf: bytes) -> list[int]:
    pos = 0
    vals = []
    while pos < len(buf):
        v, pos = _read_varint(buf, pos)
        vals.append(v)
    return vals


def _to_i32(val: int) -> int:
    return val if val < (1 << 31) else val - (1 << 32)


def parse_tree_nodes(tree_pb: bytes, width: int, height: int) -> list[dict]:
    nodes = []
    screen_area = max(1, width * height)
    for fn, wt, win_pb in _parse_pb(tree_pb):
        if fn != 1 or wt != 2:
            continue
        tmsgs = [v for f2, w2, v in _parse_pb(win_pb) if f2 == 11 and w2 == 2]
        if not tmsgs:
            continue
        for f3, w3, node_pb in _parse_pb(tmsgs[0]):
            if f3 != 1 or w3 != 2:
                continue
            d = {
                "bounds": (0, 0, 0, 0), "vis": 0, "clk": 0, "edt": 0, "scr": 0, "chk": 0,
                "cls": "", "cd": "", "txt": "", "res": "", "pkg": "",
            }
            for kf, kw, kv in _parse_pb(node_pb):
                if kf == 2 and kw == 2:
                    bm = {b1: _to_i32(b2) for b1, _, b2 in _parse_pb(kv)}
                    d["bounds"] = (bm.get(1, 0), bm.get(2, 0), bm.get(3, 0), bm.get(4, 0))
                elif kf == 3 and kw == 2:
                    d["cls"] = kv.decode("utf-8", "replace").split(".")[-1]
                elif kf == 4 and kw == 2:
                    d["cd"] = kv.decode("utf-8", "replace").strip()
                elif kf == 5 and kw == 2:
                    d["txt"] = kv.decode("utf-8", "replace").strip()
                elif kf == 6 and kw == 2:
                    d["pkg"] = kv.decode("utf-8", "replace").strip()
                elif kf == 10 and kw == 2:
                    d["res"] = kv.decode("utf-8", "replace").split("/")[-1]
                elif kf == 17 and kw == 0:
                    d["clk"] = kv
                elif kf == 19 and kw == 0:
                    d["edt"] = kv
                elif kf == 22 and kw == 0:
                    d["scr"] = kv
                elif kf == 23 and kw == 0:
                    d["vis"] = kv
                elif kf == 15 and kw == 0:
                    d["chk"] = kv
            l, t, r, b = d["bounds"]
            area = max(0, r - l) * max(0, b - t)
            if not d["vis"] or r <= l or b <= t or d["pkg"] == "com.android.systemui":
                continue
            if area >= 0.5 * screen_area and not d["txt"] and not d["cd"]:
                continue
            if d["txt"] or d["cd"] or d["edt"] or d["scr"] or d["clk"]:
                d["id"] = len(nodes)
                nodes.append(d)
    return nodes


def render_nodes(nodes: list[dict]) -> str:
    lines = []
    for n in nodes:
        flags = [f for f in ("clk", "edt", "scr", "chk") if n[f]]
        parts = [f"[{n['id']}] {n['cls']}"]
        if n["txt"]:
            parts.append(f"text={n['txt'][:120]!r}")
        if n["cd"] and n["cd"] != n["txt"]:
            parts.append(f"desc={n['cd'][:120]!r}")
        if n["res"]:
            parts.append(f"res={n['res']}")
        if flags:
            parts.append(",".join(flags))
        parts.append(f"box={list(n['bounds'])}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _matching_node_ids(nodes: list[dict], x: int, y: int) -> list[int]:
    hits = []
    for n in nodes:
        l, t, r, b = n["bounds"]
        if l <= x <= r and t <= y <= b:
            area = max(1, (r - l) * (b - t))
            hits.append((area, n["id"]))
    hits.sort()
    return [nid for _, nid in hits]


def load_episodes(
    dataset_paths: Path | list[Path],
    splits_path: Path,
    min_steps: int,
    sample_seed: int | None = None,
) -> list[dict]:
    paths = [dataset_paths] if isinstance(dataset_paths, Path) else list(dataset_paths)
    test_ids = set(json.loads(splits_path.read_text())["test"])
    seen_eids: set[int] = set()
    episodes = []
    for dataset_path in paths:
        with gzip.open(dataset_path, "rb") as g:
            while True:
                hdr = g.read(8)
                if not hdr:
                    break
                ln = struct.unpack("<Q", hdr)[0]
                g.read(4)
                data = g.read(ln)
                g.read(4)
                feats = {}
                for _, _, entry in _parse_pb(_parse_pb(data)[0][2]):
                    kv = dict((fn, val) for fn, _, val in _parse_pb(entry))
                    k = kv[1].decode()
                    feat = _parse_pb(kv[2])
                    if not feat:
                        continue
                    fn_f, _, fv = feat[0]
                    if fn_f == 1:
                        feats[k] = [v for _, _, v in _parse_pb(fv)]
                    elif fn_f == 3:
                        sub = _parse_pb(fv)
                        feats[k] = (
                            _dec_varint_list(sub[0][2])
                            if sub and sub[0][0] == 1 and sub[0][1] == 2
                            else [v for _, _, v in sub]
                        )
                eid = int(feats["episode_id"][0])
                if eid not in test_ids or eid in seen_eids:
                    continue
                acts = [json.loads(a.decode()) for a in feats.get("actions", [])]
                if len(acts) < min_steps:
                    continue
                W = int(feats.get("screenshot_widths", [1080])[0])
                H = int(feats.get("screenshot_heights", [2400])[0])
                trees = feats.get("accessibility_trees", [])
                step_insts = [s.decode("utf-8", "replace").strip() for s in feats.get("step_instructions", [])]
                steps = []
                all_clicks_covered = True
                cum_chars = 0
                for si, act in enumerate(acts):
                    nodes = parse_tree_nodes(trees[si], W, H)
                    rendered = render_nodes(nodes)
                    cum_chars += len(rendered)
                    hits = []
                    if act["action_type"] in ("click", "long_press"):
                        hits = _matching_node_ids(nodes, int(act["x"]), int(act["y"]))
                        if not hits:
                            all_clicks_covered = False
                    steps.append({
                        "step": si,
                        "instruction": step_insts[si] if si < len(step_insts) else "",
                        "nodes": [{"id": n["id"], "bounds": n["bounds"], "txt": n["txt"], "cd": n["cd"]} for n in nodes],
                        "rendered": rendered,
                        "action": act,
                        "hit_ids": hits,
                    })
                if not all_clicks_covered:
                    continue
                seen_eids.add(eid)
                episodes.append({
                    "id": f"ac_ep_{eid}",
                    "episode_id": eid,
                    "goal": feats["goal"][0].decode("utf-8", "replace"),
                    "cum_chars": cum_chars,
                    "steps": steps,
                })
    if sample_seed is not None:
        episodes.sort(key=lambda e: e["episode_id"])
        random.Random(int(sample_seed)).shuffle(episodes)
    else:
        episodes.sort(key=lambda e: (-e["cum_chars"], e["episode_id"]))
    return episodes


def gold_action_json(step: dict) -> str:
    act = step["action"]
    at = act["action_type"]
    if at in ("click", "long_press"):
        return json.dumps({"action_type": at, "id": step["hit_ids"][0]}, ensure_ascii=False)
    return json.dumps(act, ensure_ascii=False)


def eval_step(step: dict, raw_content: str) -> dict:
    gt = step["action"]
    gt_type = gt["action_type"]
    obj, err = _parse_prediction(raw_content)
    if obj is None:
        return {"valid": False, "error_type": "parse_error", "error": err, "expected": gt, "predicted": None}
    pred_type = str(obj.get("action_type", "")).strip().lower()
    if pred_type != gt_type:
        return {"valid": False, "error_type": "wrong_action_type", "expected": gt, "predicted": obj}

    if gt_type in ("click", "long_press"):
        try:
            pid = int(obj.get("id"))
        except (TypeError, ValueError):
            return {"valid": False, "error_type": "invalid_node_id", "expected": gt, "predicted": obj}
        hit_set = set(step["hit_ids"])
        ok = pid in hit_set
        strict_ok = bool(step["hit_ids"] and pid == step["hit_ids"][0])
        return {
            "valid": ok,
            "strict_valid": strict_ok,
            "error_type": None if ok else "wrong_target",
            "expected": {"action_type": gt_type, "hit_ids": step["hit_ids"][:5]},
            "predicted": obj,
        }
    if gt_type == "input_text":
        ptxt = str(obj.get("text", "")).strip().lower()
        gtxt = str(gt.get("text", "")).strip().lower()
        ok = ptxt == gtxt
        return {"valid": ok, "strict_valid": ok, "error_type": None if ok else "wrong_text", "expected": gt, "predicted": obj}
    if gt_type == "open_app":
        papp = str(obj.get("app_name", "")).strip().lower()
        gapp = str(gt.get("app_name", "")).strip().lower()
        ok = papp == gapp
        return {"valid": ok, "strict_valid": ok, "error_type": None if ok else "wrong_app", "expected": gt, "predicted": obj}
    if gt_type == "scroll":
        pdir = str(obj.get("direction", "")).strip().lower()
        gdir = str(gt.get("direction", "")).strip().lower()
        ok = pdir == gdir
        return {"valid": ok, "strict_valid": ok, "error_type": None if ok else "wrong_direction", "expected": gt, "predicted": obj}
    return {"valid": True, "strict_valid": True, "error_type": None, "expected": gt, "predicted": obj}


def run_episode(ep: dict, cfg: dict, agent_base: str, out_dir: Path, meta: dict) -> dict:
    edir = out_dir / "tasks" / ep["id"]
    res_path = edir / "result.json"
    if res_path.exists():
        cached = json.loads(res_path.read_text())
        if cached.get("trace_complete") and same_run(cached, meta):
            return cached
    edir.mkdir(parents=True, exist_ok=True)

    enable_thinking = bool(cfg.get("enable_thinking", False))
    prompt_mode = str(cfg.get("prompt_mode", "high_level"))
    steps = ep["steps"]
    sys_msg = f"{_SYSTEM_PROMPT}\n\nGoal: {ep['goal']}"
    gold_prefixes: list[list[dict]] = []
    running_msgs = [{"role": "system", "content": sys_msg}]
    for sidx, step in enumerate(steps):
        if prompt_mode == "low_level" and step.get("instruction"):
            ucontent = f"Step {sidx} instruction: {step['instruction']}\nStep {sidx} UI tree:\n{step['rendered']}"
        else:
            ucontent = f"Step {sidx} UI tree:\n{step['rendered']}"
        running_msgs.append({"role": "user", "content": ucontent})
        gold_prefixes.append(copy.deepcopy(running_msgs))
        running_msgs.append({"role": "assistant", "content": gold_action_json(step)})

    try:
        post(agent_base, "/a2o/reset", {}, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    turn_results = []
    all_calls = []
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
                # HTTP call succeeded and anchor was already saved cleanly; do not trigger prefix replay.
                turn_error = "ValueError: output token limit exceeded"
                eval_res = {"valid": False, "strict_valid": False, "error_type": "runtime_error"}
            else:
                eval_res = eval_step(step, content)
                turn_complete = True
        except Exception:
            turn_error = traceback.format_exc()
            eval_res = {"valid": False, "strict_valid": False, "error_type": "runtime_error"}
            need_replay = True

        all_calls.extend(t_calls)
        turn_results.append({
            "turn": sidx,
            "complete": turn_complete,
            "valid": bool(turn_complete and eval_res["valid"]),
            "strict_valid": bool(turn_complete and eval_res.get("strict_valid", eval_res["valid"])),
            "eval": eval_res,
            "prompt_tokens": (t_calls[-1].get("usage") or {}).get("prompt_tokens") if t_calls else None,
            "completion_tokens": (t_calls[-1].get("usage") or {}).get("completion_tokens") if t_calls else None,
            "error": turn_error,
            "raw_content": content,
        })

    n_turns = len(turn_results)
    n_complete = sum(1 for r in turn_results if r["complete"])
    n_valid = sum(1 for r in turn_results if r["valid"])
    n_strict = sum(1 for r in turn_results if r["strict_valid"])
    res = {
        "id": ep["id"],
        "episode_id": ep["episode_id"],
        "goal": ep["goal"],
        "cum_chars": ep["cum_chars"],
        **meta,
        "trace_complete": n_complete == n_turns,
        "total_turns": n_turns,
        "complete_turns": n_complete,
        "valid_turns": n_valid,
        "strict_valid_turns": n_strict,
        "turn_accuracy": n_valid / n_turns if n_turns else 0.0,
        "strict_turn_accuracy": n_strict / n_turns if n_turns else 0.0,
        "all_turns_valid": n_valid == n_turns,
        "max_prompt_tokens": max((r["prompt_tokens"] or 0) for r in turn_results),
        "turn_results": turn_results,
        "kv_stats": _summarize_kv(all_calls),
    }
    dump(edir / "calls.json", all_calls)
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
    raw_paths = cfg.get("dataset_paths") or [cfg["dataset_path"]]
    episodes = load_episodes(
        [Path(p) for p in raw_paths],
        Path(cfg["splits_path"]),
        int(cfg.get("min_steps", 4)),
        sample_seed=cfg.get("sample_seed"),
    )
    selected = select_tasks(episodes, lambda e: e["id"], args.task_id, args.limit_tasks)

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, ep in enumerate(selected):
        res = run_episode(ep, cfg, args.agent_base.rstrip("/"), args.output, meta)
        results.append(res)
        ks = res["kv_stats"]
        print(
            f"[{args.condition_name}] {idx+1}/{len(selected)} {res['id']} "
            f"turns={res['valid_turns']}/{res['total_turns']} strict={res['strict_valid_turns']}/{res['total_turns']} "
            f"max_ptok={res['max_prompt_tokens']} prior={ks['prior_turns']}/{ks['compact_turns']} "
            f"evict={ks['evicted_tokens']}/{ks['logical_tokens']}",
            flush=True,
        )

    tot_turns = sum(r["total_turns"] for r in results)
    tot_complete = sum(r["complete_turns"] for r in results)
    tot_valid = sum(r["valid_turns"] for r in results)
    tot_strict = sum(r["strict_valid_turns"] for r in results)
    summary = {
        **meta,
        "selected_task_ids": [e["id"] for e in selected],
        "total_tasks": len(results),
        "total_turns": tot_turns,
        "complete_turns": tot_complete,
        "valid_turns": tot_valid,
        "strict_valid_turns": tot_strict,
        "turn_accuracy": tot_valid / tot_turns if tot_turns else 0.0,
        "strict_turn_accuracy": tot_strict / tot_turns if tot_turns else 0.0,
        "all_turns_valid_tasks": sum(1 for r in results if r["all_turns_valid"]),
        "results": results,
    }
    dump(args.output / "worker-summary.json", summary)


if __name__ == "__main__":
    main()
