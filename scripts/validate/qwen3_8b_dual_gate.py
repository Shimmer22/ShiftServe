#!/usr/bin/env python3
"""Phase-1 gate probe: Qwen3-8B-Q4_0 dual-residency (cpu_repack=0) + compact_anchor RAM/latency."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import post

ENDPOINT = "http://10.29.131.50:18780"
SSH_CMD = [
    "ssh",
    "-p", "8022",
    "-i", str(Path.home() / ".ssh/a2o_xiaomi14pro_ed25519"),
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", f"UserKnownHostsFile={ROOT / 'configs/termux-known-hosts'}",
    "u0_a320@10.29.131.50",
]


def get_health():
    with urllib.request.urlopen(ENDPOINT + "/health", timeout=20) as resp:
        return json.loads(resp.read().decode())


def read_phone_proc_mem():
    out = subprocess.check_output(
        SSH_CMD + [
            "pid=$(for d in /proc/[0-9]*; do "
            "c=$(cat \"$d/comm\" 2>/dev/null || true); "
            "case \"$c\" in service|agent) basename \"$d\";; esac; "
            "done | tail -n1); "
            "if [ -n \"$pid\" ] && [ -r \"/proc/$pid/status\" ]; then "
            "echo \"service_pid=$pid\"; "
            "grep -E '^(VmSize|VmHWM|VmRSS|VmSwap):' /proc/$pid/status; "
            "awk '{print \"majflt=\"$12}' /proc/$pid/stat; "
            "fi; "
            "free -m | awk '/Mem:|Swap:/{print $1\" \"$2\" \"$3\" \"$4\" \"$7}'"
        ],
        text=True,
    )
    info = {"raw": out.strip()}
    for ln in out.splitlines():
        if ":" in ln and ln.split(":", 1)[0] in {"VmSize", "VmHWM", "VmRSS", "VmSwap"}:
            k, v = ln.split(":", 1)
            info[k.strip()] = int(v.strip().split()[0])
        elif ln.startswith("majflt="):
            info["major_faults"] = int(ln.split("=", 1)[1])
    return info


def route_service(action: str, cfg: dict, svc_dir: Path, log_path: Path, ctx_size: int, cost_path: Path):
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
            "--context-size", str(ctx_size),
            "--cpu-threads", "6",
            "--cpu-repack", "0",
            "--opbatch", "1024",
            "--opqueue", "16",
            "--kv-offload", "0",
            "--flash-attention", "0",
            "--kv-reduce", "1",
            "--kv-policy", "adaptive_cost",
            "--kv-costs", str(cost_path),
            "--kv-sink-tokens", "4",
            "--kv-recent-tokens", "256",
            "--kv-pool-window", "5",
            "--kv-tail-tokens", "64",
            "--compact-anchor", "1",
            "--expected-decode", "96",
        ]
    else:
        cmd.append("--logs-only")
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_stage(cfg: dict, out: Path, cost_path: Path, ctx_size: int, chat_turns: int, h_gap: int, p_list: list[int]):
    tag = f"ctx{ctx_size}"
    svc_dir = out / f"svc-{tag}"
    svc_log = out / f"svc-{tag}.log"
    result = {"ctx_size": ctx_size, "ok": False, "chat": [], "gap_replay": [], "error": None}
    try:
        route_service("start", cfg, svc_dir, svc_log, ctx_size, cost_path)
        health = get_health()
        result["health"] = health
        result["mem_after_init"] = read_phone_proc_mem()
        print(f"[{tag}] started; init_ms={health['init_ms']:.1f}, mem={result['mem_after_init']}", flush=True)

        # 1. Multi-turn chat growing context with compact_anchor=1 (staying under logical ctx_size until L161 is updated in Phase 2)
        post(ENDPOINT, "/a2o/reset", {}, 60)
        messages = [{"role": "system", "content": "You summarize telemetry concisely in one sentence."}]
        chunk = "Telemetry frame: temp=34C, volt=4120mV, ticket=#841, status=OK, carrier=LTE-A.\n" * 30
        for t in range(chat_turns):
            prompt = (
                f"Batch {t}:\n{chunk}\nReply in one short sentence citing batch {t} and ticket #841."
                if t != 2
                else "Which ticket number did batch 0 and batch 1 mention?"
            )
            messages.append({"role": "user", "content": prompt})
            resp = retry_network(lambda: post(ENDPOINT, "/v1/chat/completions", {
                "model": cfg["model_id"],
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 32,
                "expected_decode_tokens": 96,
            }, 900))
            choice = resp["choices"][0]["message"]
            messages.append({"role": "assistant", "content": choice.get("content", "")})
            a2o = resp.get("timings", {})
            mem = read_phone_proc_mem()
            row = {
                "turn": t,
                "H_anchor": a2o.get("H_anchor"),
                "H_effective": a2o.get("H_effective"),
                "P": a2o.get("prompt_n"),
                "route": a2o.get("route"),
                "restore_ms": a2o.get("restore_ms") or 0.0,
                "prompt_ms": a2o.get("prompt_ms") or 0.0,
                "kv_reduce_ms": a2o.get("kv_reduce_ms") or 0.0,
                "handoff_ms": a2o.get("handoff_ms") or 0.0,
                "predicted_n": a2o.get("predicted_n") or 0,
                "predicted_ms": a2o.get("predicted_ms") or 0.0,
                "kv_cost_decision": a2o.get("kv_cost_decision"),
                "kv_reduce": a2o.get("kv_reduce"),
                "text": choice.get("content", "")[:140],
                "mem": mem,
            }
            result["chat"].append(row)
            print(
                f"[{tag} chat t={t}] route={row['route']} H={row['H_anchor']}(eff={row['H_effective']}) P={row['P']} "
                f"prm={row['prompt_ms']:.0f}ms rst={row['restore_ms']:.0f}ms kvr={row['kv_reduce_ms']:.0f}ms "
                f"RSS={mem.get('VmRSS',0)/1024:.0f}MiB HWM={mem.get('VmHWM',0)/1024:.0f}MiB Swap={mem.get('VmSwap',0)/1024:.0f}MiB majflt={mem.get('major_faults')}",
                flush=True,
            )

        # 2. Gap-replay paired cpu (repack=0) vs route (kv=0) vs route (kv=1, K=1536)
        post(ENDPOINT, "/a2o/reset", {}, 60)
        unit_text = "System telemetry block: node=phone-1, queue=4, temp=33C, power=nominal.\n"
        unit_toks = post(ENDPOINT, "/tokenize", {"content": unit_text})["tokens"]
        q_toks = post(ENDPOINT, "/tokenize", {"content": "\nAcknowledge."})["tokens"]
        gen_toks = post(ENDPOINT, "/tokenize", {"content": "Acknowledged. Telemetry status nominal across all queues."})["tokens"] * 3
        gen_toks = gen_toks[:32]
        prefix = (unit_toks * max(1, h_gap // len(unit_toks)))[:h_gap]

        for p_len in p_list:
            input_ids = (unit_toks * max(1, p_len // len(unit_toks)))[: max(1, p_len - len(q_toks))] + q_toks
            input_ids = input_ids[:p_len]
            for kv_on in (0, 1):
                body = {
                    "prefix_ids": prefix,
                    "input_ids": input_ids,
                    "generated_ids": gen_toks,
                    "order": ["cpu", "route"] if kv_on == 0 else ["route"],
                    "allow_partial_order": True,
                    "gap_ms": 4000.0,
                    "return_policy": "greedy",
                    "prediction": {
                        "validated": True,
                        "prepare_ms": 500.0,
                        "cpu_prefill_ms": 50000.0,
                        "cpu_restore_ms": 2000.0,
                        "htp_prefill_handoff_ms": 500.0,
                    },
                    "kv_reduce": bool(kv_on),
                    "kv_reduce_policy": "prefill_prior",
                    "kv_keep_tokens": 1536,
                    "kv_sink_tokens": 4,
                    "kv_recent_tokens": 256,
                    "kv_pool_window": 5,
                    "kv_tail_tokens": 64,
                }
                resp = retry_network(lambda: post(ENDPOINT, "/a2o/gap-replay", body, 1800))
                for r in resp["results"]:
                    rb = r["resources_before"]
                    ra = r["resources_after"]
                    entry = {
                        "H": r["H"], "P": r["P"], "D": r["D"], "kv_on": kv_on,
                        "condition": r["condition"], "route": r["route"],
                        "decode_n_kv": r["decode_n_kv"],
                        "prepare_ms": r["prepare_ms"], "restore_ms": r["restore_ms"],
                        "prefill_ms": r["prefill_ms"], "evict_ms": r["evict_ms"],
                        "handoff_ms": r["handoff_ms"], "handoff_bytes": r["handoff_bytes"],
                        "decode_ms": r["decode_ms"], "decode_ms_per_tok": r["decode_ms"] / r["D"],
                        "post_return_ms": r["post_return_ms"],
                        "rss_after_mib": ra.get("vm_rss_kib", 0) / 1024.0,
                        "swap_after_mib": ra.get("vm_swap_kib", 0) / 1024.0,
                        "majflt_delta": (ra.get("major_faults", 0) - rb.get("major_faults", 0)),
                    }
                    result["gap_replay"].append(entry)
                    print(
                        f"[{tag} gap H={entry['H']} P={entry['P']} kv={kv_on} {entry['condition']}->{entry['route']}] "
                        f"prep={entry['prepare_ms']:.0f} prm={entry['prefill_ms']:.0f} ev={entry['evict_ms']:.1f} "
                        f"ho={entry['handoff_ms']:.0f}({entry['handoff_bytes']/1048576:.1f}MiB) "
                        f"dec={entry['decode_ms_per_tok']:.1f}ms/tok post={entry['post_return_ms']:.0f} "
                        f"RSS={entry['rss_after_mib']:.0f}MiB Swap={entry['swap_after_mib']:.0f}MiB Δmajflt={entry['majflt_delta']}",
                        flush=True,
                    )
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{tag}] FAILED: {result['error']}", flush=True)
    finally:
        try:
            route_service("stop", cfg, svc_dir, svc_log, ctx_size, cost_path)
        except Exception as stop_exc:
            print(f"[{tag}] stop warning: {stop_exc}", flush=True)
    return result


def main():
    out = ROOT / "results/20261115-qwen3-8b-dual-gate"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_id": "qwen3-8b-q4_0",
        "model_path": "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf",
        "model_sha256": "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066",
        "binary": "results/20261115-qwen3-1_7b-adaptive-kv/build-service/agent",
        "registry": "results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so",
    }
    # Permissive gate cost table so chat admits compaction once H_total >= 3072
    gate_cost = {
        "kind": "kv_reduce_cost_v1",
        "model_id": "qwen3-8b-q4_0",
        "expected_decode_tokens": 96,
        "quality_k_min": 1536,
        "quality_keep_ratio_min": 0.25,
        "align_pad": 256,
        "min_evict_savings_tokens": 512,
        "min_gain_ms": 10.0,
        "candidates": [1536, 2048, 2560, 3072],
        "decode_per_tok": {"intercept_ms": 120.0, "slope_ms_per_token": 0.015},
        "handoff": {"intercept_ms": 120.0, "slope_ms_per_token": 0.13},
        "evict": {"intercept_ms": 1.0, "slope_ms_per_token": 0.0005},
        "htp_prior_tax": {"intercept_ms": 700.0, "slope_ms_per_token": 0.0},
        "cpu_prior_tax": {"intercept_ms": 900.0, "slope_ms_per_token": 0.0},
    }
    cost_path = out / "gate-cost.json"
    cost_path.write_text(json.dumps(gate_cost, indent=2) + "\n")

    summary = {"stages": []}
    s_6k = run_stage(cfg, out, cost_path, ctx_size=6144, chat_turns=4, h_gap=3584, p_list=[128, 256, 512])
    summary["stages"].append(s_6k)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if s_6k["ok"]:
        s_8k = run_stage(cfg, out, cost_path, ctx_size=8192, chat_turns=4, h_gap=4096, p_list=[128, 256, 512])
        summary["stages"].append(s_8k)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
