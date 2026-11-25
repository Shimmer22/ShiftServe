#!/usr/bin/env python3
"""Measure 3 small-P warm points under opbatch=1024, opqueue=16, htp_big_cores=1, cpu_threads=6,
and build clean ob1024 cold & 7-point 2D WLS route cost tables."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.collect.tau2_e2e_worker import _http_post_resilient

BASE = "http://10.29.131.50:18780"
BUILD = ROOT / "results/mainline-eval/t6-opt-htp-prefill"
MODEL = "/data/data/com.termux/files/home/.local/a2o/models/Qwen3-8B-Q4_0.gguf"
MODEL_ID = "qwen3-8b-q4_0"
MODEL_SHA = "7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066"
AC_FIXTURES = ROOT / "results/20261124-aclow-10cases/fixtures.json"
BASE_ROUTE_COSTS = ROOT / "configs/qwen3-8b-route-costs.json"
CAL_DIR = ROOT / "results/mainline-eval/20261125-qwen3-8b-cold-cal-6t-opt"
VERIFY_WARM = ROOT / "results/mainline-eval/20261125-htp-opt-verify/armB_shared_hc_big0/cal-warm.json"
COLD_OUT = ROOT / "configs/qwen3-8b-cold-costs-6t-ob1024.json"
ROUTE_OUT = ROOT / "configs/qwen3-8b-route-costs-6t-ob1024.json"


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def fit_wls_2d(rows: list[tuple[int, int, float]]) -> tuple[float, float, float]:
    """Fit y = a + b*P + c*(H/1000)*P with 1/y^2 weights via weighted normal equations."""
    A = [[0.0, 0.0, 0.0] for _ in range(3)]
    b = [0.0, 0.0, 0.0]
    for H, P, y in rows:
        x = [1.0, float(P), (float(H) / 1000.0) * float(P)]
        w = 1.0 / (y * y)
        for i in range(3):
            b[i] += w * x[i] * y
            for j in range(3):
                A[i][j] += w * x[i] * x[j]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 3):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    beta = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        beta[i] = (b[i] - sum(A[i][j] * beta[j] for j in range(i + 1, 3))) / A[i][i]
    return max(0.0, beta[0]), max(0.0, beta[1]), max(0.0, beta[2])


def main() -> None:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(AC_FIXTURES.read_text())
    long_tokens: list[int] = []
    for c in corpus["cases"]:
        for t in reversed(c["turns"]):
            ids = t["request"]["frozen_prompt_ids"]
            if len(ids) > len(long_tokens):
                long_tokens = ids
    assert len(long_tokens) >= 4096 + 48

    svc_dir = CAL_DIR / f"svc-smallp-{time.time_ns()}"
    start_cmd = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"), "start",
        "--condition", "route", "--shared-weights",
        "--output", str(svc_dir), "--logs-only", "--retry-network",
        "--runtime-overlay", str(BUILD),
        "--model", MODEL, "--model-id", MODEL_ID, "--model-sha256", MODEL_SHA,
        "--context-size", "9472", "--cpu-repack", "0", "--cpu-threads", "6",
        "--opbatch", "1024", "--opqueue", "16", "--htp-big-cores", "1",
        "--kv-hwm-trim", "0", "--memory-ledger", "--gap-ms", "0",
        "--max-quiet-c", "38.0", "--min-prime-khz", "2000000",
        "--min-mem-avail-kib", "1500000", "--cooldown-timeout-s", "360",
    ]
    subprocess.run(start_cmd, cwd=ROOT, check=True)
    try:
        for H, P in ((1024, 24), (2048, 32), (4096, 48)):
            out_f = CAL_DIR / f"warm-H{H}-P{P}.json"
            res = _http_post_resilient(
                BASE + "/a2o/calibrate",
                {
                    "prefix_ids": long_tokens[:H],
                    "input_ids": long_tokens[H : H + P],
                    "decode_evaluations": 1,
                    "repetitions": 1,
                },
                timeout=1200,
                retries=2,
            )
            save(out_f, res)
            row = res["rows"][0]
            assert row["cosine"] >= 0.99, f"H={H} P={P} cosine={row['cosine']}"
            print(f"[warm ob1024 H={H:4d} P={P:3d}] cosine={row['cosine']:.5f} htp_ms={row['htp_prefill_ms']:.1f} cpu_ms={row['cpu_prefill_ms']:.1f}", flush=True)
    finally:
        subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/run/route_phone.py"), "stop",
                "--condition", "route", "--output", str(svc_dir), "--logs-only", "--retry-network",
            ],
            cwd=ROOT, check=True,
        )

    # Reconstruct cold_route_v1 from the 5 already-measured cold-P*.json files
    cold_points = []
    for p in (64, 256, 512, 1024, 2048):
        r = json.loads((CAL_DIR / f"cold-P{p}.json").read_text())["rows"][0]
        cold_points.append({"P": p, "cc_ms": r["cold_cc_ms"], "hc_ms": r["cold_hc_ms"]})
    save(COLD_OUT, {
        "kind": "cold_route_v1",
        "model_id": MODEL_ID,
        "shared_weights": True,
        "opbatch": 1024,
        "opqueue": 16,
        "htp_big_cores": True,
        "cpu_threads": 6,
        "points": cold_points,
    })

    # Fit 7-point 2D WLS for htp_prefill
    warm_specs = [
        (606, 42, VERIFY_WARM),
        (1024, 24, CAL_DIR / "warm-H1024-P24.json"),
        (2048, 32, CAL_DIR / "warm-H2048-P32.json"),
        (4096, 48, CAL_DIR / "warm-H4096-P48.json"),
        (1024, 128, CAL_DIR / "warm-H1024-P128.json"),
        (2048, 512, CAL_DIR / "warm-H2048-P512.json"),
        (4096, 1024, CAL_DIR / "warm-H4096-P1024.json"),
    ]
    warm_rows = []
    for H, P, fp in warm_specs:
        r = json.loads(fp.read_text())["rows"][0]
        warm_rows.append((H, P, r["htp_prefill_ms"]))

    a, b, c = fit_wls_2d(warm_rows)
    route_cfg = json.loads(BASE_ROUTE_COSTS.read_text())
    route_cfg["model"]["model_id"] = "qwen3-8b-q4_0-6t-ob1024-wls-2d"
    route_cfg["model"]["htp_prefill"] = {
        "intercept_ms": a,
        "slope_ms_per_token": b,
        "hp_slope_ms_per_1k_tokens": c,
        "weight": "1/y^2",
        "n": len(warm_rows),
        "samples": [{"H": H, "P": P, "htp_prefill_ms": y} for H, P, y in warm_rows],
    }
    save(ROUTE_OUT, route_cfg)
    print(f"Saved {COLD_OUT} and {ROUTE_OUT}: intercept={a:.2f}ms slope={b:.2f}ms/tok hp_slope={c:.2f}ms/1k*tok")


if __name__ == "__main__":
    main()
