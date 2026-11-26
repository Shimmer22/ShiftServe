#!/usr/bin/env python3
"""Collect leakage-free synthetic multi-turn infer() calibration samples under exact
t6-final-ob1024 production flags (htp_big_cores=1, opbatch=1024, opqueue=16, cpu_threads=6)
to calibrate htp_prefill, cpu_to_htp (prepare_ms), and htp_to_cpu (handoff_ms)."""
from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network
from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post_resilient

BASE = os.environ.get("A2O_PHONE_HTTP_BASE", "http://10.29.131.50:18780").rstrip("/")
BUILD = ROOT / "results/mainline-eval/t6-final-ob1024"
MODEL = "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf"
MODEL_ID = "qwen3-8b-q4_0"
MODEL_SHA = "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066"
COLD_COSTS = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
OUT = ROOT / "results/mainline-eval/20261126-ob1024-infer-hc-cal-6t"

SHORT_CHAINS = [
    ("short_h400", 400, [24, 32, 40, 48, 64, 24, 32, 40, 48, 64]),
    ("short_h900", 900, [24, 32, 40, 48, 64, 24, 32, 40, 48, 64]),
    ("short_h1800", 1800, [24, 32, 40, 48, 64, 24, 32, 40, 48, 64]),
    ("short_h3200", 3200, [24, 32, 40, 48, 64, 24, 32, 40, 48, 64]),
]

LONG_CHAINS = [
    ("long_h400", 400, [128, 256, 512, 1024]),
    ("long_h1200", 1200, [128, 256, 512, 1024]),
    ("long_h2400", 2400, [128, 256, 512, 1024]),
]


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_quiet_therm_c(transport: DeviceTransport) -> float | None:
    cmd = "cat /sys/class/thermal/thermal_zone64/temp 2>/dev/null || echo na"
    out = retry_network(lambda: transport.call("shell", cmd)).stdout.decode().strip()
    if out and out != "na":
        num = float(out)
        return num / 1000.0 if num > 200.0 else num
    return None


def wait_for_cooldown(transport: DeviceTransport, target_c: float, max_wait_s: int = 180, label: str = "") -> float | None:
    temp = read_quiet_therm_c(transport)
    if temp is None or temp <= target_c:
        return temp
    t0 = time.monotonic()
    print(f"[cooldown {label}] quiet-therm={temp:.2f}C > {target_c:.2f}C, waiting...", flush=True)
    while time.monotonic() - t0 < max_wait_s:
        time.sleep(5)
        temp = read_quiet_therm_c(transport)
        if temp is None or temp <= target_c:
            print(f"[cooldown {label}] reached quiet-therm={temp:.2f}C in {time.monotonic() - t0:.1f}s", flush=True)
            return temp
    print(f"[cooldown {label}] timeout after {max_wait_s}s at quiet-therm={temp:.2f}C", flush=True)
    return temp


def render_and_tokenize(messages: list[dict]) -> list[int]:
    rendered = _http_post_resilient(BASE + "/apply-template", {"messages": messages}, timeout=60, retries=5)["prompt"]
    return _http_post_resilient(BASE + "/tokenize", {"content": rendered}, timeout=60, retries=5)["tokens"]


def make_prose(n_words: int, seed_idx: int) -> str:
    vocab = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
        "yankee", "zulu", "river", "stone", "cloud", "forest", "bridge", "harbor",
    ]
    return " ".join(vocab[(seed_idx * 7 + i) % len(vocab)] for i in range(max(1, n_words)))


def build_turn_matching_p(prev_msgs: list[dict], prev_anchor_len: int, target_p: int, seed_idx: int) -> tuple[list[dict], list[int]]:
    """Binary search word count so (len(ids) - prev_anchor_len) is as close as possible to target_p."""
    lo, hi = 1, max(4, target_p * 2)
    best_msgs, best_ids, best_diff = None, None, 10**9
    while lo <= hi:
        mid = (lo + hi) // 2
        msgs = prev_msgs + [{"role": "user", "content": make_prose(mid, seed_idx)}]
        ids = render_and_tokenize(msgs)
        p = len(ids) - prev_anchor_len
        diff = abs(p - target_p)
        if diff < best_diff:
            best_diff, best_msgs, best_ids = diff, msgs, ids
        if p == target_p:
            break
        if p < target_p:
            lo = mid + 1
        else:
            hi = mid - 1
    assert best_msgs is not None and best_ids is not None
    return best_msgs, best_ids


def run_chain(transport: DeviceTransport, chain_name: str, h0: int, p_list: list[int], cooldown_c: float | None) -> list[dict]:
    chain_dir = OUT / chain_name
    chain_dir.mkdir(parents=True, exist_ok=True)
    summary_file = chain_dir / "chain-samples.json"
    if summary_file.is_file():
        print(f"[{chain_name}] already complete; skipping.", flush=True)
        return json.loads(summary_file.read_text())

    _http_post_resilient(BASE + "/a2o/reset", {}, timeout=60, retries=5)
    seed_base = abs(hash(chain_name)) % 1000

    # Turn 0
    msgs0, ids0 = build_turn_matching_p([], 0, h0, seed_base)
    assert ids0[-4:] == [151667, 271, 151668, 271], f"unexpected anchor tail: {ids0[-6:]}"
    resp0 = _agent_chat_resilient(
        BASE,
        {
            "messages": msgs0,
            "frozen_prompt_ids": ids0,
            "frozen_decode_ids": [198],
            "max_tokens": 1,
            "expected_decode_tokens": 1,
        },
        f"infer-cal:{chain_name}:t0",
        total_timeout=600,
    )
    save_json(chain_dir / "turn-00.json", resp0)
    prev_anchor = ids0[:-4]
    cur_msgs = msgs0 + [{"role": "assistant", "content": "OK"}]

    samples = []
    for ti, target_p in enumerate(p_list, 1):
        msgs_t, ids_t = build_turn_matching_p(cur_msgs, len(prev_anchor), target_p, seed_base + ti)
        assert ids_t[: len(prev_anchor)] == prev_anchor, f"{chain_name} T{ti} prefix mismatch!"
        assert ids_t[-4:] == [151667, 271, 151668, 271]

        qc = wait_for_cooldown(transport, cooldown_c, max_wait_s=180, label=f"{chain_name}-t{ti}") if cooldown_c else read_quiet_therm_c(transport)
        _http_post_resilient(BASE + "/a2o/tool-start", {}, timeout=30, retries=5)
        time.sleep(0.75)

        resp = _agent_chat_resilient(
            BASE,
            {
                "messages": msgs_t,
                "frozen_prompt_ids": ids_t,
                "frozen_decode_ids": [198],
                "max_tokens": 1,
                "expected_decode_tokens": 1,
                "frozen_force_path": "hc_full",
            },
            f"infer-cal:{chain_name}:t{ti}",
            total_timeout=600,
        )
        save_json(chain_dir / f"turn-{ti:02d}.json", resp)
        t = resp["timings"]
        assert t["route"] == "htp" and t["availability"] == "ready", f"{chain_name} T{ti} route={t['route']} avail={t['availability']}"
        assert t["H_effective"] == len(prev_anchor), f"H_eff={t['H_effective']} != {len(prev_anchor)}"
        assert t["prepare_ms"] > 0 and t["prompt_ms"] > 0 and t["handoff_ms"] > 0

        row = {
            "chain": chain_name,
            "turn": ti,
            "H": int(t["H_effective"]),
            "P": int(t["P_evaluated"]),
            "quiet_c": qc,
            "prepare_ms": float(t["prepare_ms"]),
            "wait_ms": float(t["wait_ms"]),
            "htp_prefill_ms": float(t["prompt_ms"]),
            "handoff_ms": float(t["handoff_ms"]),
        }
        samples.append(row)
        print(
            f"[{chain_name} T{ti:02d}] H={row['H']:4d} P={row['P']:4d} | "
            f"prep={row['prepare_ms']:6.1f}ms prefill={row['htp_prefill_ms']:7.1f}ms hnd={row['handoff_ms']:5.1f}ms T={qc}",
            flush=True,
        )
        prev_anchor = ids_t[:-4]
        cur_msgs = msgs_t + [{"role": "assistant", "content": "OK"}]

    save_json(summary_file, samples)
    return samples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_f = (OUT / "driver.lock").open("a")
    fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    transport = DeviceTransport(ROOT, "ssh", config=ROOT / "configs/termux-ssh.json")

    all_samples: list[dict] = []
    # Check if already collected
    if (OUT / "infer-hc-samples.csv").is_file():
        print("infer-hc-samples.csv already exists; skipping collection.", flush=True)
        return

    svc_dir = OUT / f"svc-{time.time_ns()}"
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), "start",
        "--condition", "route", "--shared-weights",
        "--output", str(svc_dir), "--logs-only", "--retry-network",
        "--runtime-overlay", str(BUILD),
        "--model", MODEL, "--model-id", MODEL_ID, "--model-sha256", MODEL_SHA,
        "--context-size", "9472", "--cpu-repack", "0", "--cpu-threads", "6",
        "--opbatch", "1024", "--opqueue", "16", "--htp-big-cores", "1",
        "--kv-hwm-trim", "0", "--verify-kv-state", "0", "--gap-ms", "0",
        "--cold-costs", str(COLD_COSTS),
        "--max-quiet-c", "38.0", "--min-prime-khz", "2000000",
        "--min-mem-avail-kib", "1500000", "--cooldown-timeout-s", "360",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    try:
        for cname, h0, ps in SHORT_CHAINS:
            all_samples.extend(run_chain(transport, cname, h0, ps, cooldown_c=38.5))
        for cname, h0, ps in LONG_CHAINS:
            all_samples.extend(run_chain(transport, cname, h0, ps, cooldown_c=38.5))
    finally:
        subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/run/route_phone.py"), "stop",
                "--condition", "route", "--output", str(svc_dir), "--logs-only", "--retry-network",
            ],
            cwd=ROOT, check=True,
        )

    csv_path = OUT / "infer-hc-samples.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_samples[0]))
        w.writeheader()
        w.writerows(all_samples)
    print(f"Wrote {csv_path} ({len(all_samples)} rows)", flush=True)


if __name__ == "__main__":
    main()
