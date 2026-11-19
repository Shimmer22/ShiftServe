#!/usr/bin/env python3
"""Container-side official tau2-bench Host KV-accuracy worker.

Routes:
  - agent -> phone_service_host (/a2o/chat-start + /a2o/chat-wait)
  - user  -> shared llama-server (/v1/chat/completions)
  - judge -> external JUDGE_BASE_URL (gpt-6.1-terra)
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import urllib.error
import urllib.request


def _http_post(url: str, body: dict, timeout: int = 1800, api_key: str = "dummy") -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> HTTP {exc.code}: {detail}") from exc


def _http_post_resilient(
    url: str, body: dict, timeout: int = 25, retries: int = 120, api_key: str = "dummy"
) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            return _http_post(url, body, timeout=timeout, api_key=api_key)
        except (http.client.HTTPException, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            time.sleep(min(4.0, 0.5 + 0.25 * attempt))
        except RuntimeError as exc:
            if any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")) and attempt + 1 < retries:
                last_exc = exc
                time.sleep(min(8.0, 1.0 + 0.5 * attempt))
                continue
            raise
    raise RuntimeError(f"POST {url} failed after {retries} network retries: {last_exc}") from last_exc


def _agent_chat_resilient(agent_base: str, req_body: dict, request_id: str, total_timeout: float = 1800.0) -> dict:
    base = agent_base.rstrip("/")
    _http_post_resilient(f"{base}/a2o/chat-start", {**req_body, "request_id": request_id}, timeout=20, retries=120)
    deadline = time.monotonic() + total_timeout
    while time.monotonic() < deadline:
        poll = _http_post_resilient(
            f"{base}/a2o/chat-wait", {"request_id": request_id, "wait_ms": 15000}, timeout=25, retries=120
        )
        status = poll.get("status")
        if status == "done":
            return poll["result"]
        if status == "error":
            raise RuntimeError(f"Agent async chat failed ({request_id}): {poll.get('error')}")
        if status == "unknown":
            _http_post_resilient(f"{base}/a2o/chat-start", {**req_body, "request_id": request_id}, timeout=20, retries=120)
    raise TimeoutError(f"Agent async chat timed out after {total_timeout}s ({request_id})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--condition-name", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--agent-base", required=True)
    p.add_argument("--user-base", required=True)
    p.add_argument("--tau2-src", type=Path, default=Path("/workspace/references/tau2-bench-v0.1.0-source/src"))
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()

    judge_base = os.environ["JUDGE_BASE_URL"].rstrip("/")
    judge_key = os.environ["JUDGE_API_KEY"]
    judge_model = os.environ["JUDGE_MODEL"]

    sys.path.insert(0, str(args.tau2_src))
    cfg = json.loads(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    import tau2.utils.llm_utils as llm_utils
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.environment.environment import Environment

    agent_model_tag = f"host-{cfg['model_id']}"
    user_model_tag = cfg["user_model"]
    active_trace = {"calls": [], "tools": [], "task_key": "init"}

    def patched_generate(model, messages, tools=None, tool_choice=None, **kwargs):
        litellm_messages = llm_utils.to_litellm_messages(messages)
        for m in litellm_messages:
            if m.get("content") is None:
                m["content"] = ""
        oai_tools = [tool.openai_schema for tool in tools] if tools else None
        is_agent = model == agent_model_tag
        is_user = model == user_model_tag

        if is_agent:
            req_body = {
                "model": cfg["model_id"],
                "messages": litellm_messages,
                "temperature": float(kwargs.get("temperature", 0.0)),
                "max_tokens": int(cfg["agent_max_tokens"]),
                "expected_decode_tokens": int(cfg.get("expected_decode_tokens", 96)),
                "omit_verbose_prompt": True,
            }
            if oai_tools:
                req_body["tools"] = oai_tools
                req_body["tool_choice"] = tool_choice or "auto"
        elif is_user:
            req_body = {
                "model": user_model_tag,
                "messages": litellm_messages,
                "temperature": float(kwargs.get("temperature", 0.0)),
                "max_tokens": int(cfg["user_max_tokens"]),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if oai_tools:
                req_body["tools"] = oai_tools
                req_body["tool_choice"] = tool_choice or "auto"
        else:
            req_body = {
                "model": judge_model,
                "messages": litellm_messages,
                "temperature": float(kwargs.get("temperature", 0.0)),
                "response_format": {"type": "json_object"},
            }
            if oai_tools:
                req_body["tools"] = oai_tools
                req_body["tool_choice"] = tool_choice or "auto"

        t0 = time.monotonic()
        if is_agent:
            turn_n = sum(1 for c in active_trace["calls"] if c["is_agent"])
            rid = f"{active_trace['task_key']}:t{turn_n}"
            resp = _agent_chat_resilient(args.agent_base, req_body, rid, total_timeout=1800.0)
        elif is_user:
            resp = _http_post_resilient(
                args.user_base.rstrip("/") + "/v1/chat/completions", req_body, timeout=180, retries=10
            )
        else:
            resp = _http_post_resilient(
                judge_base + "/chat/completions", req_body, timeout=120, retries=6, api_key=judge_key
            )
        wall_ms = (time.monotonic() - t0) * 1000.0

        choice = resp["choices"][0]
        msg = choice["message"]
        raw_content = msg.get("content")
        if not is_agent and (raw_content is None or not str(raw_content).strip()):
            rc = (msg.get("reasoning_content") or "").strip()
            if rc:
                raw_content = rc
        if not is_agent and isinstance(raw_content, str):
            stripped = raw_content.strip()
            m_fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.S)
            if m_fence:
                raw_content = m_fence.group(1)

        parsed_tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed_args = {"_raw_invalid_json": str(raw_args)}
            parsed_tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{len(parsed_tool_calls)}",
                    name=fn.get("name", ""),
                    arguments=parsed_args if isinstance(parsed_args, dict) else {"_value": parsed_args},
                )
            )
        tool_calls_out = parsed_tool_calls or None

        usage_raw = resp.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
        }
        rec = {
            "role": "agent" if is_agent else ("user" if is_user else "judge"),
            "model_arg": model,
            "wall_ms": wall_ms,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage_raw,
            "timings": resp.get("timings"),
            "content_len": len(raw_content or ""),
            "tool_calls_count": len(parsed_tool_calls),
            "turn_index": len([c for c in active_trace["calls"] if c["is_agent"] == is_agent]),
            "is_agent": is_agent,
        }
        active_trace["calls"].append(rec)

        return AssistantMessage(
            role="assistant",
            content=raw_content,
            tool_calls=tool_calls_out,
            cost=0.0,
            usage=usage,
            raw_data=resp,
        )

    llm_utils.generate = patched_generate

    orig_tool_call = Environment.make_tool_call

    def patched_tool_call(self, tool_name, requestor, **kwargs):
        t0 = time.monotonic()
        res = orig_tool_call(self, tool_name, requestor, **kwargs)
        dt_ms = (time.monotonic() - t0) * 1000.0
        active_trace["tools"].append({"tool_name": tool_name, "requestor": requestor, "ms": dt_ms})
        return res

    Environment.make_tool_call = patched_tool_call

    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_task

    task_specs = cfg["tasks"][: args.limit_tasks] if args.limit_tasks > 0 else cfg["tasks"]
    by_domain = {}
    for spec in task_specs:
        dom = spec["domain"]
        if dom not in by_domain:
            by_domain[dom] = {t.id: t for t in get_tasks(dom)}

    tasks_out = []
    traces_dir = out / "tasks"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for idx, spec in enumerate(task_specs):
        dom = spec["domain"]
        tid = spec["task_id"]
        key = f"{idx:02d}-{dom}-" + hashlib.sha256(tid.encode()).hexdigest()[:10]
        task_path = traces_dir / f"{key}.json"
        if task_path.exists():
            cached = json.loads(task_path.read_text())
            if cached.get("complete"):
                tasks_out.append(cached)
                continue
            fail_idx = 0
            while (traces_dir / f"{key}.failed-{fail_idx}.json").exists():
                fail_idx += 1
            task_path.rename(traces_dir / f"{key}.failed-{fail_idx}.json")

        task = by_domain[dom][tid]
        active_trace["calls"] = []
        active_trace["tools"] = []
        active_trace["task_key"] = key
        _http_post_resilient(args.agent_base.rstrip("/") + "/a2o/reset", {}, timeout=25, retries=120)

        t_start = time.monotonic()
        sim_dump = None
        err = None
        try:
            sim = run_task(
                domain=dom,
                task=task,
                agent="llm_agent",
                user="user_simulator",
                llm_agent=agent_model_tag,
                llm_args_agent={"temperature": 0.0},
                llm_user=user_model_tag,
                llm_args_user={"temperature": 0.0},
                max_steps=int(cfg["max_steps"]),
                max_errors=int(cfg["max_errors"]),
                evaluation_type=EvaluationType.ALL,
                seed=int(cfg["seed"]),
            )
            sim_dump = sim.model_dump(mode="json")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        elapsed_s = time.monotonic() - t_start

        reward_info = (sim_dump or {}).get("reward_info") or {}
        agent_calls = [c for c in active_trace["calls"] if c["is_agent"]]
        record = {
            "index": idx,
            "key": key,
            "condition": args.condition_name,
            "domain": dom,
            "task_id": tid,
            "complete": err is None and sim_dump is not None,
            "error": err,
            "duration_s": elapsed_s,
            "termination_reason": (sim_dump or {}).get("termination_reason"),
            "reward": reward_info.get("reward"),
            "reward_breakdown": reward_info.get("reward_breakdown"),
            "db_check": reward_info.get("db_check"),
            "action_checks": reward_info.get("action_checks"),
            "nl_assertions": reward_info.get("nl_assertions"),
            "agent_turns": len(agent_calls),
            "user_or_judge_calls": len(active_trace["calls"]) - len(agent_calls),
            "tool_calls": len(active_trace["tools"]),
            "calls": active_trace["calls"],
            "tools": active_trace["tools"],
            "messages": (sim_dump or {}).get("messages"),
        }
        task_path.write_text(json.dumps(record, indent=2) + "\n")
        tasks_out.append(record)
        print(
            f"[{args.condition_name}] {idx+1}/{len(task_specs)} {dom}:{tid[:28]} "
            f"complete={record['complete']} reward={record['reward']} "
            f"term={record['termination_reason']} agent_turns={record['agent_turns']} dur={elapsed_s:.1f}s",
            flush=True,
        )

    summary = {
        "condition": args.condition_name,
        "total_tasks": len(tasks_out),
        "completed_tasks": sum(1 for r in tasks_out if r["complete"]),
        "reward_sum": sum(float(r["reward"] or 0.0) for r in tasks_out),
        "reward_mean": (sum(float(r["reward"] or 0.0) for r in tasks_out) / len(tasks_out)) if tasks_out else 0.0,
        "results": tasks_out,
    }
    (out / "worker-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
