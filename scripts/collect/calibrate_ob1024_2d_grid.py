#!/usr/bin/env python3
"""Collect a clean 2D warm calibration grid on t6-final-ob1024 (opbatch=1024, opqueue=16,
htp_big_cores=1, cpu_threads=6, htp_host_threads=4):
- Sub-grid A (short P, isolated session): H in {512, 1024, 2048, 4096} x P in {16, 24, 32, 40, 48, 64}, 4 rounds each.
- Sub-grid B (long P, separate cool session): H in {512, 1024, 2048, 4096} x P in {256, 512, 1024}, 2 rounds each.
"""
from __future__ import annotations

import collections
import csv
import fcntl
import http.client
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import is_network_error, retry_network

BASE = os.environ.get("A2O_PHONE_HTTP_BASE", "http://10.29.131.50:18780").rstrip("/")
BUILD = ROOT / "results/mainline-eval/t6-final-ob1024"
EXPECTED_COMMIT = "730ad251fcbe519de639a9af418705f66d992fed"
MODEL = "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf"
MODEL_ID = "qwen3-8b-q4_0"
MODEL_SHA = "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066"
PROMPT_SOURCE = ROOT / "results/mainline-eval/prefill-motivation-h4096-8b/fixture.json"
OUT = ROOT / "results/mainline-eval/20261125-ob1024-2d-cal-6t"

H_LEVELS = (512, 1024, 2048, 4096)
SHORT_P_LEVELS = (16, 24, 32, 40, 48, 64)
SHORT_ROUNDS = 4
LONG_P_LEVELS = (256, 512, 1024)
LONG_ROUNDS = 2


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


def http_get_json(path: str, timeout: int = 120) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as stream:
        return json.load(stream)


def post_resilient(path: str, body: dict, timeout: int, identity: dict, label: str) -> dict:
    while True:
        try:
            req = urllib.request.Request(
                BASE + path,
                json.dumps(body).encode("utf-8"),
                {"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as stream:
                return json.load(stream)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {path} ({label}) -> HTTP {exc.code}: {detail}") from exc
        except (http.client.HTTPException, urllib.error.URLError, TimeoutError, OSError) as exc:
            if not is_network_error(exc) and not isinstance(exc, (http.client.HTTPException, TimeoutError)):
                raise
            print(f"[network retry {label}] {exc}; checking /health...", flush=True)
            health = retry_network(lambda: http_get_json("/health", timeout=60))
            stable = {k: v for k, v in health.items() if k != "init_ms"}
            if stable != identity:
                raise RuntimeError(f"Service identity changed after reconnect: {stable} != {identity}")


def validate_stage_sweep(req: dict, resp: dict, *, require_cached: bool = False) -> None:
    if resp.get("mode") != "stage_sweep_v1":
        raise ValueError(f"Unexpected mode: {resp.get('mode')}")
    if resp["cache"]["prefix_tokens"] != len(req["prefix_ids"]):
        raise ValueError("prefix_tokens mismatch")
    if require_cached and resp["cache"]["source"] not in ("memory", "disk"):
        raise ValueError(f"Expected cached history, got source={resp['cache']['source']}")
    if [r["backend"] for r in resp["rows"]] != req["backend_order"]:
        raise ValueError("backend_order mismatch")
    for row in resp["rows"]:
        if (row["H"], row["P"], row["D"]) != (len(req["prefix_ids"]), len(req["input_ids"]), 0):
            raise ValueError("shape mismatch")
        if not (row["state_restore_verified"] and row["finite_logits"]):
            raise ValueError("restore or finite_logits failed")
        if not (math.isfinite(row["prefill_ms"]) and row["prefill_ms"] > 0):
            raise ValueError("invalid prefill_ms")
        expected_threads = 6 if row["backend"] == "cpu" else 4
        if row.get("n_threads") != expected_threads or row.get("n_threads_batch") != expected_threads:
            raise ValueError(f"thread mismatch for {row['backend']}: {row.get('n_threads')}/{row.get('n_threads_batch')}")
    comp = resp["logit_comparison"]
    if not (math.isfinite(comp["cosine"]) and math.isfinite(comp["nrmse"]) and comp["cosine"] >= 0.99 and comp["passed"]):
        raise ValueError(f"logit comparison failed: {comp}")


def start_service(svc_dir: Path) -> dict:
    cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), "start",
        "--condition", "route", "--shared-weights", "--replay-cache",
        "--output", str(svc_dir), "--logs-only", "--retry-network",
        "--runtime-overlay", str(BUILD),
        "--model", MODEL, "--model-id", MODEL_ID, "--model-sha256", MODEL_SHA,
        "--context-size", "9472", "--cpu-repack", "0", "--cpu-threads", "6",
        "--opbatch", "1024", "--opqueue", "16", "--htp-big-cores", "1",
        "--kv-hwm-trim", "0", "--verify-kv-state", "0", "--gap-ms", "0",
        "--max-quiet-c", "37.5", "--min-prime-khz", "2000000",
        "--min-mem-avail-kib", "1500000", "--cooldown-timeout-s", "360",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    health = json.loads((svc_dir / "health.json").read_text())
    assert health["git_commit"] == EXPECTED_COMMIT, health["git_commit"]
    assert health["shared_weights"] is True
    assert health["cpu_threads"] == 6
    assert health["htp_host_threads"] == 4
    assert health["htp_big_cores"] is True
    assert health["context_size"] == 9472
    assert health["stage_sweep_version"] == 1
    return {k: v for k, v in health.items() if k != "init_ms"}


def stop_service(svc_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/run/route_phone.py"), "stop",
            "--condition", "route", "--output", str(svc_dir), "--logs-only", "--retry-network",
        ],
        cwd=ROOT, check=True,
    )


def symlink_replay_cache(transport: DeviceTransport, src_svc: Path, dst_svc: Path) -> None:
    q = shlex.quote
    remote_dst = transport.remote_root + "/" + dst_svc.name
    cache_src = transport.remote_root + "/" + src_svc.name + "/replay-cache"
    cache_dst = remote_dst + "/replay-cache"
    guard = f'test "$(cat {q(transport.remote_root + "/model.lock/owner")})" = {q(dst_svc.name)}'
    cmd = (
        f"{guard} && test -d {q(cache_src)} && "
        f"if [ -e {q(cache_dst)} ]; then test \"$(readlink {q(cache_dst)})\" = {q(cache_src)}; "
        f"else ln -s {q(cache_src)} {q(cache_dst)}; fi"
    )
    retry_network(lambda: transport.call("shell", cmd))


def is_valid_file(path: Path, req: dict, *, require_cached: bool = False) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        if data.get("request") != req:
            return False
        validate_stage_sweep(req, data["response"], require_cached=require_cached)
        return True
    except Exception:
        return False


def aggregate_outputs() -> None:
    raw_rows = []
    grouped: dict[tuple[str, int, int], list[dict]] = collections.defaultdict(list)
    specs = [
        ("short", OUT / "short_p", H_LEVELS, SHORT_P_LEVELS, SHORT_ROUNDS),
        ("long", OUT / "long_p", H_LEVELS, LONG_P_LEVELS, LONG_ROUNDS),
    ]
    for grid_name, grid_dir, hs, ps, rounds in specs:
        for h in hs:
            for p in ps:
                for r in range(rounds):
                    fpath = grid_dir / f"H{h:04d}-P{p:04d}-r{r:02d}.json"
                    data = json.loads(fpath.read_text())
                    req = data["request"]
                    resp = data["response"]
                    validate_stage_sweep(req, resp)
                    by_b = {row["backend"]: row for row in resp["rows"]}
                    c_row, h_row = by_b["cpu"], by_b["htp"]
                    item = {
                        "grid": grid_name,
                        "H": h,
                        "P": p,
                        "round": r,
                        "backend_order": ",".join(req["backend_order"]),
                        "cache_source": resp["cache"]["source"],
                        "quiet_c_before": data.get("quiet_c_before"),
                        "cpu_restore_ms": c_row["restore_ms"],
                        "cpu_prefill_ms": c_row["prefill_ms"],
                        "htp_restore_ms": h_row["restore_ms"],
                        "htp_prefill_ms": h_row["prefill_ms"],
                        "cosine": resp["logit_comparison"]["cosine"],
                        "nrmse": resp["logit_comparison"]["nrmse"],
                    }
                    raw_rows.append(item)
                    grouped[(grid_name, h, p)].append(item)

    with (OUT / "raw-pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader()
        w.writerows(raw_rows)

    summary_rows = []
    for grid_name, _, hs, ps, _ in specs:
        for h in hs:
            for p in ps:
                items = grouped[(grid_name, h, p)]
                c_vals = [x["cpu_prefill_ms"] for x in items]
                h_vals = [x["htp_prefill_ms"] for x in items]
                cr_vals = [x["cpu_restore_ms"] for x in items]
                c_med = statistics.median(c_vals)
                h_med = statistics.median(h_vals)
                c_cv = (statistics.stdev(c_vals) / c_med * 100.0) if len(c_vals) > 1 else 0.0
                h_cv = (statistics.stdev(h_vals) / h_med * 100.0) if len(h_vals) > 1 else 0.0
                summary_rows.append({
                    "grid": grid_name,
                    "H": h,
                    "P": p,
                    "n": len(items),
                    "cpu_restore_median_ms": statistics.median(cr_vals),
                    "cpu_prefill_median_ms": c_med,
                    "cpu_prefill_mean_ms": statistics.mean(c_vals),
                    "cpu_cv_pct": c_cv,
                    "htp_prefill_median_ms": h_med,
                    "htp_prefill_mean_ms": statistics.mean(h_vals),
                    "htp_cv_pct": h_cv,
                    "paired_cpu_wins": sum(c < h_v for c, h_v in zip(c_vals, h_vals)),
                    "min_cosine": min(x["cosine"] for x in items),
                })

    with (OUT / "grid-summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {OUT / 'raw-pairs.csv'} ({len(raw_rows)} rows) and {OUT / 'grid-summary.csv'} ({len(summary_rows)} rows)", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_f = (OUT / "driver.lock").open("a")
    fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)

    prompt = json.loads(PROMPT_SOURCE.read_text())["choices"][0]["prompt_token_ids"]
    assert len(prompt) >= max(H_LEVELS) + max(LONG_P_LEVELS)
    transport = DeviceTransport(ROOT, "ssh", config=ROOT / "configs/termux-ssh.json")

    svc_short = OUT / "svc-short-p"
    svc_long = OUT / "svc-long-p"
    short_dir = OUT / "short_p"
    long_dir = OUT / "long_p"
    short_dir.mkdir(exist_ok=True)
    long_dir.mkdir(exist_ok=True)

    # Check if short_p is already completely collected (including remote-artifacts.json so replay-cache exists)
    short_jobs = []
    for h in H_LEVELS:
        for r in range(SHORT_ROUNDS):
            ps = list(SHORT_P_LEVELS)
            random.Random(20261125 + h + r).shuffle(ps)
            for p_idx, p in enumerate(ps):
                order = ["cpu", "htp"] if (r + p_idx) % 2 == 0 else ["htp", "cpu"]
                req = {
                    "prefix_ids": prompt[:h],
                    "input_ids": prompt[h : h + p],
                    "generated_ids": [],
                    "decode_points": [],
                    "backend_order": order,
                }
                short_jobs.append((h, p, r, req, short_dir / f"H{h:04d}-P{p:04d}-r{r:02d}.json"))

    need_short = not (svc_short / "remote-artifacts.json").is_file() or any(
        not is_valid_file(fp, req) for _, _, _, req, fp in short_jobs
    )

    if need_short:
        print("=== Session 1: Short-P 2D Grid (H=512..4096 x P=16..64) ===", flush=True)
        identity = start_service(svc_short)
        try:
            norm_path = OUT / "normal-response.json"
            if not norm_path.is_file():
                norm_resp = post_resilient(
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "Reply with exactly OK."}], "max_tokens": 16},
                    timeout=300, identity=identity, label="normal-response",
                )
                msg = norm_resp["choices"][0]["message"]
                assert "ok" in (msg.get("content") or "").lower(), msg
                assert norm_resp["usage"]["completion_tokens"] > 0
                assert not norm_resp["__verbose"]["truncated"]
                save_json(norm_path, norm_resp)
                post_resilient("/a2o/reset", {}, timeout=60, identity=identity, label="reset-after-normal")
                print("[Session 1] normal-response check passed.", flush=True)

            for h in H_LEVELS:
                boot_path = short_dir / f"bootstrap-H{h:04d}.json"
                boot_req = {
                    "prefix_ids": prompt[:h],
                    "input_ids": prompt[h : h + 16],
                    "generated_ids": [],
                    "decode_points": [],
                    "backend_order": ["htp", "cpu"],
                }
                if not is_valid_file(boot_path, boot_req):
                    print(f"[Session 1] Bootstrapping H={h}...", flush=True)
                    boot_resp = post_resilient("/a2o/stage-sweep", boot_req, timeout=1800, identity=identity, label=f"boot-H{h}")
                    validate_stage_sweep(boot_req, boot_resp)
                    save_json(boot_path, {"request": boot_req, "response": boot_resp})
                    wait_for_cooldown(transport, 38.0, max_wait_s=180, label=f"post-boot-H{h}")

                h_jobs = [j for j in short_jobs if j[0] == h]
                for _, p, r, req, fp in h_jobs:
                    if is_valid_file(fp, req):
                        continue
                    qc = wait_for_cooldown(transport, 38.5, max_wait_s=180, label=f"short-H{h}-P{p}-r{r}")
                    resp = post_resilient("/a2o/stage-sweep", req, timeout=300, identity=identity, label=f"short-H{h}-P{p}-r{r}")
                    validate_stage_sweep(req, resp)
                    save_json(fp, {"quiet_c_before": qc, "request": req, "response": resp})
                    by_b = {row["backend"]: row for row in resp["rows"]}
                    print(
                        f"[short H={h:4d} P={p:2d} r={r}] cpu={by_b['cpu']['prefill_ms']:6.1f}ms "
                        f"htp={by_b['htp']['prefill_ms']:6.1f}ms cos={resp['logit_comparison']['cosine']:.5f} "
                        f"src={resp['cache']['source']} T={qc}",
                        flush=True,
                    )
        finally:
            stop_service(svc_short)
    else:
        print("=== Session 1 already complete; skipping ===", flush=True)

    long_jobs = []
    for h in H_LEVELS:
        for r in range(LONG_ROUNDS):
            for p_idx, p in enumerate(LONG_P_LEVELS):
                order = ["cpu", "htp"] if (r + p_idx) % 2 == 0 else ["htp", "cpu"]
                req = {
                    "prefix_ids": prompt[:h],
                    "input_ids": prompt[h : h + p],
                    "generated_ids": [],
                    "decode_points": [],
                    "backend_order": order,
                }
                long_jobs.append((h, p, r, req, long_dir / f"H{h:04d}-P{p:04d}-r{r:02d}.json"))

    need_long = any(not is_valid_file(fp, req, require_cached=True) for _, _, _, req, fp in long_jobs)
    if need_long:
        print("=== Session 2: Long-P 2D Grid (H=512..4096 x P=256..1024) ===", flush=True)
        identity = start_service(svc_long)
        try:
            symlink_replay_cache(transport, svc_short, svc_long)
            for h, p, r, req, fp in long_jobs:
                if is_valid_file(fp, req, require_cached=True):
                    continue
                qc = wait_for_cooldown(transport, 38.5, max_wait_s=180, label=f"long-H{h}-P{p}-r{r}")
                resp = post_resilient("/a2o/stage-sweep", req, timeout=600, identity=identity, label=f"long-H{h}-P{p}-r{r}")
                validate_stage_sweep(req, resp, require_cached=True)
                save_json(fp, {"quiet_c_before": qc, "request": req, "response": resp})
                by_b = {row["backend"]: row for row in resp["rows"]}
                print(
                    f"[long  H={h:4d} P={p:4d} r={r}] cpu={by_b['cpu']['prefill_ms']:8.1f}ms "
                    f"htp={by_b['htp']['prefill_ms']:8.1f}ms cos={resp['logit_comparison']['cosine']:.5f} "
                    f"src={resp['cache']['source']} T={qc}",
                    flush=True,
                )
        finally:
            stop_service(svc_long)
    else:
        print("=== Session 2 already complete; skipping ===", flush=True)

    aggregate_outputs()


if __name__ == "__main__":
    main()
