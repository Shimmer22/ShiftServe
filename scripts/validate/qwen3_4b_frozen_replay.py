#!/usr/bin/env python3
"""Paired frozen-trajectory gap-replay on Qwen3-4B-Q4_0 (v3) across 4 conditions.

Samples stratified warm turns from real tau2 E2E traces (cpu_baseline), tokenizes
real task tools/messages to match each turn's (H, P, D), and runs /a2o/gap-replay
on a single route service instance for:
  1. cpu_baseline          (cpu, kv_reduce=false)
  2. cpu_adaptive_compact  (cpu, kv_reduce=true, kv_reduce_on_cpu=true, adaptive_cost)
  3. route_m1_only         (route, kv_reduce=false)
  4. route_m1_m2_compact   (route, kv_reduce=true, adaptive_cost)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import post

ENDPOINT = "http://10.29.131.50:18780"
DEFAULT_E2E_CFG = ROOT / "configs/tau2-qwen3-4b-joint-v2-e2e.json"
DEFAULT_TRACES_DIR = ROOT / "results/20261116-tau2-qwen3-4b-joint-v2-e2e/conditions/cpu_baseline/tasks"
DEFAULT_OUT_DIR = ROOT / "results/20261117-qwen3-4b-frozen-replay"


def route_service(action: str, cfg: dict, svc_dir: Path, log_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run/route_phone.py"),
        action,
        "--condition", "route",
        "--output", str(svc_dir),
        "--retry-network",
    ]
    if action == "start":
        cmd += [
            "--binary", str((ROOT / cfg["binary"]).resolve()),
            "--registry", str((ROOT / cfg["registry"]).resolve()),
            "--model", cfg["model_path"],
            "--model-id", cfg["model_id"],
            "--model-sha256", cfg["model_sha256"],
            "--context-size", str(cfg["context_size"]),
            "--cpu-threads", str(cfg["cpu_threads"]),
            "--cpu-repack", str(cfg["cpu_repack"]),
            "--opbatch", str(cfg["opbatch"]),
            "--opqueue", str(cfg["opqueue"]),
            "--kv-offload", str(cfg["kv_offload"]),
            "--flash-attention", str(cfg["flash_attention"]),
            "--kv-reduce", "1",
            "--kv-policy", "adaptive_cost",
            "--kv-costs", str((ROOT / cfg["kv_costs"]).resolve()),
            "--kv-sink-tokens", "4",
            "--kv-recent-tokens", "256",
            "--kv-pool-window", "5",
            "--kv-tail-tokens", "64",
            "--compact-anchor", "1",
            "--expected-decode", str(cfg["expected_decode_tokens"]),
            "--replay-cache",
        ]
    else:
        cmd.append("--logs-only")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def install_costs(cfg: dict) -> None:
    route_model = json.loads((ROOT / cfg["route_costs"]).read_text())
    kv_model = json.loads((ROOT / cfg["kv_costs"]).read_text())
    payload = {
        "model": route_model.get("model", route_model),
        "kv_model": kv_model,
    }
    post(ENDPOINT, "/a2o/costs", payload, 60)


def eval_linear(term: dict, x: float) -> float:
    return max(0.0, float(term["intercept_ms"]) + float(term["slope_ms_per_token"]) * float(x))


def predict_costs(route_model: dict, h: int, p: int) -> dict:
    m = route_model.get("model", route_model)
    cpu_restore = eval_linear(m["cpu_restore"], h)
    cpu_prefill = eval_linear(m["cpu_prefill"], p)
    htp_prefill = eval_linear(m["htp_prefill"], p)
    htp_to_cpu = eval_linear(m["htp_to_cpu"], h + p)
    prepare_ms = eval_linear(m["cpu_to_htp"], h)
    return {
        "validated": True,
        "prepare_ms": prepare_ms,
        "cpu_restore_ms": cpu_restore,
        "cpu_prefill_ms": cpu_prefill,
        "htp_prefill_handoff_ms": htp_prefill + htp_to_cpu,
    }


def collect_warm_turns(traces_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(traces_dir.glob("*.json")):
        if ".failed-" in p.name:
            continue
        d = json.loads(p.read_text())
        calls = d.get("calls", [])
        for i, c in enumerate(calls):
            if not c.get("is_agent"):
                continue
            t = c.get("timings") or {}
            h = int(t.get("H_anchor", 0))
            if h <= 0:
                continue
            p_eval = int(t.get("P_evaluated", 0))
            d_gen = int(t.get("predicted_n", 0))
            if p_eval <= 0 or d_gen <= 0:
                continue
            gap_ms = 0.0
            j = i - 1
            while j >= 0 and not calls[j].get("is_agent"):
                gap_ms += float(calls[j].get("wall_ms", 0.0))
                j -= 1
            rows.append({
                "key": d["key"],
                "domain": d["domain"],
                "task_path": str(p),
                "turn_index": int(c["turn_index"]),
                "H": h,
                "P": p_eval,
                "D": d_gen,
                "observed_gap_ms": gap_ms,
                "pinned_tokens": int(t.get("pinned_tokens", 0)),
            })
    rows.sort(key=lambda r: r["P"])
    return rows


def select_stratified(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    idxs = [int(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)]
    return [rows[i] for i in idxs]


def tokenize_text(text: str) -> list[int]:
    resp = retry_network(lambda: post(ENDPOINT, "/tokenize", {"content": text}, 120))
    return list(resp["tokens"])


def fit_tokens(tokens: list[int], target_len: int, filler: list[int]) -> list[int]:
    if len(tokens) >= target_len:
        return tokens[:target_len]
    out = list(tokens)
    while len(out) < target_len:
        need = target_len - len(out)
        out.extend(filler[:need])
    return out


def build_sample_tokens(sample: dict) -> tuple[list[int], list[int], list[int]]:
    task_data = json.loads(Path(sample["task_path"]).read_text())
    tools_text = json.dumps(task_data.get("tools", []), ensure_ascii=False)
    msgs = task_data.get("messages", [])
    turn_idx = sample["turn_index"]

    # Partition real messages into history (before turn_idx), current user/tool input, and assistant reply
    hist_parts = [f"<|im_start|>system\nTools: {tools_text}<|im_end|>\n"]
    cur_in_parts = []
    cur_out_parts = []
    seen_agent = 0
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content") or ""
        tc = m.get("tool_calls") or []
        rendered = content + (f"\n{json.dumps(tc, ensure_ascii=False)}" if tc else "")
        block = f"<|im_start|>{role}\n{rendered}<|im_end|>\n"
        if role == "assistant":
            if seen_agent == turn_idx:
                cur_out_parts.append(rendered or "Acknowledged.")
                break
            seen_agent += 1
            hist_parts.append(block)
        else:
            if seen_agent == turn_idx:
                cur_in_parts.append(block)
            else:
                hist_parts.append(block)

    filler_text = "System policy and conversation history context block for tau2 agent verification.\n"
    filler_toks = tokenize_text(filler_text)
    prefix_raw = tokenize_text("".join(hist_parts))
    input_raw = tokenize_text("".join(cur_in_parts) if cur_in_parts else "Please proceed with the next tool step.")
    gen_raw = tokenize_text("".join(cur_out_parts) if cur_out_parts else "I will check that information for you now.")

    prefix_ids = fit_tokens(prefix_raw, sample["H"], filler_toks)
    input_ids = fit_tokens(input_raw, sample["P"], filler_toks)
    gen_ids = fit_tokens(gen_raw, sample["D"], filler_toks)
    return prefix_ids, input_ids, gen_ids


def run_gap_replay(
    prefix_ids: list[int],
    input_ids: list[int],
    gen_ids: list[int],
    gap_ms: float,
    prediction: dict,
    mode: str,
    kv_reduce: bool,
    kv_reduce_on_cpu: bool,
) -> dict:
    body = {
        "prefix_ids": prefix_ids,
        "input_ids": input_ids,
        "generated_ids": gen_ids,
        "order": [mode],
        "allow_partial_order": True,
        "gap_ms": gap_ms,
        "return_policy": "greedy",
        "prediction": prediction,
        "kv_reduce": kv_reduce,
        "kv_reduce_on_cpu": kv_reduce_on_cpu,
        "kv_reduce_policy": "adaptive_cost" if kv_reduce else "prefill_prior",
        "kv_keep_tokens": 1536,
        "kv_sink_tokens": 4,
        "kv_recent_tokens": 256,
        "kv_pool_window": 5,
        "kv_tail_tokens": 64,
        "expected_decode_tokens": 96,
    }
    resp = retry_network(lambda: post(ENDPOINT, "/a2o/gap-replay", body, 2400))
    return resp["results"][0]


def summarize(records: list[dict]) -> dict:
    conds = ["cpu_baseline", "cpu_adaptive_compact", "route_m1_only", "route_m1_m2_compact"]
    summary = {}
    for c in conds:
        rows = [r["conditions"][c] for r in records if c in r.get("conditions", {})]
        if not rows:
            continue
        tot_d = sum(x["D"] for x in rows)
        tot_dec = sum(x["decode_ms"] for x in rows)
        summary[c] = {
            "n": len(rows),
            "avg_post_return_ms": statistics.mean(x["post_return_ms"] for x in rows),
            "avg_prefill_ms": statistics.mean(x["prefill_ms"] for x in rows),
            "avg_restore_ms": statistics.mean(x["restore_ms"] for x in rows),
            "avg_evict_ms": statistics.mean(x["evict_ms"] for x in rows),
            "avg_handoff_ms": statistics.mean(x["handoff_ms"] for x in rows),
            "avg_decode_ms": statistics.mean(x["decode_ms"] for x in rows),
            "decode_ms_per_tok": tot_dec / max(1, tot_d),
            "avg_decode_n_kv": statistics.mean(x["decode_n_kv"] for x in rows),
            "max_rss_mib": max(x["resources_after"].get("vm_rss_kib", 0) / 1024.0 for x in rows),
            "max_swap_mib": max(x["resources_after"].get("vm_swap_kib", 0) / 1024.0 for x in rows),
            "htp_routes": sum(1 for x in rows if x["route"] == "htp"),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_E2E_CFG)
    ap.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--gap-ms", type=float, default=2500.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    route_model = json.loads((ROOT / cfg["route_costs"]).read_text())

    all_warm = collect_warm_turns(args.traces_dir)
    selected = select_stratified(all_warm, args.count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    svc_dir = args.out_dir / "service-route"
    svc_log = args.out_dir / "service-route.log"

    print(f"Selected {len(selected)} stratified warm turns (gap_ms={args.gap_ms}):", flush=True)
    for i, s in enumerate(selected):
        print(f"  [{i:02d}] {s['key']} t={s['turn_index']} H={s['H']} P={s['P']} D={s['D']}", flush=True)

    route_service("start", cfg, svc_dir, svc_log)
    try:
        install_costs(cfg)
        records = []
        for i, s in enumerate(selected):
            out_file = samples_dir / f"{i:02d}.json"
            if args.resume and out_file.exists():
                rec = json.loads(out_file.read_text())
                records.append(rec)
                print(f"[{i:02d}] RESUMED from {out_file.name}", flush=True)
                continue

            t0 = time.time()
            prefix_ids, input_ids, gen_ids = build_sample_tokens(s)
            pred = predict_costs(route_model, s["H"], s["P"])
            cond_specs = [
                ("cpu_baseline", "cpu", False, False),
                ("cpu_adaptive_compact", "cpu", True, True),
                ("route_m1_only", "route", False, False),
                ("route_m1_m2_compact", "route", True, False),
            ]
            cond_res = {}
            for cname, mode, kvr, kvr_cpu in cond_specs:
                res = run_gap_replay(prefix_ids, input_ids, gen_ids, args.gap_ms, pred, mode, kvr, kvr_cpu)
                cond_res[cname] = res
                ra = res["resources_after"]
                print(
                    f"  [{i:02d} {s['key']} H={s['H']} P={s['P']} D={s['D']}] {cname:21s} "
                    f"route={res['route']:3s} n_kv={res['decode_n_kv']:5d} "
                    f"post={res['post_return_ms']:7.0f}ms (prm={res['prefill_ms']:6.0f} ev={res['evict_ms']:4.0f} "
                    f"ho={res['handoff_ms']:4.0f} dec={res['decode_ms']:6.0f}) "
                    f"RSS={ra.get('vm_rss_kib',0)/1024:.0f}M Swap={ra.get('vm_swap_kib',0)/1024:.0f}M",
                    flush=True,
                )
            rec = {
                "index": i,
                "sample": s,
                "gap_ms": args.gap_ms,
                "prediction": pred,
                "elapsed_s": time.time() - t0,
                "conditions": cond_res,
            }
            out_file.write_text(json.dumps(rec, indent=2))
            records.append(rec)

            summary = {
                "completed": len(records),
                "total": len(selected),
                "gap_ms": args.gap_ms,
                "summary": summarize(records),
            }
            (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("\n=== FROZEN REPLAY SUMMARY ===", flush=True)
        print(json.dumps(summarize(records), indent=2), flush=True)
    finally:
        try:
            route_service("stop", cfg, svc_dir, svc_log)
        except Exception as exc:
            print(f"Service stop warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
