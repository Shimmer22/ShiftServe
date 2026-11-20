#!/usr/bin/env python3
"""T1 DoD validation on phone: single frozen build, forced CC/HC-Full/HC-Compact, and 3-way M1-Explorer joint routing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post

BASE = "http://10.29.131.50:18780"
BUILD = ROOT / "results/20261120-t1-frozen-build-v4"
MODEL_CFG = json.loads((ROOT / "configs/model-qwen3-4b.json").read_text())
FIXTURES = json.loads((ROOT / "results/mainline-eval/main-4b-ablation/fixtures.json").read_text())
KV_COSTS = ROOT / "configs/qwen3-4b-kv-reduce-cost.json"
ROUTE_COSTS = ROOT / "configs/qwen3-4b-route-costs.json"
OUT = ROOT / "results/20261120-t1-dod-verify"


def pick_case_with_min_h(min_h: int):
    for c in FIXTURES["cases"]:
        if len(c["turns"]) >= 2 and c["turns"][1]["H"] >= min_h:
            return c
    raise RuntimeError(f"No case with H >= {min_h}")


def prime_and_prepare(case, gap_ms: int = 1000):
    _http_post(BASE + "/a2o/reset", {}, timeout=60)
    t0 = case["turns"][0]
    b0 = dict(t0["request"], max_tokens=t0["D"], expected_decode_tokens=t0["D"])
    _agent_chat_resilient(BASE, b0, f"t1-prime-{case['id']}", total_timeout=600)
    t_begin = time.monotonic()
    prep = _http_post(BASE + "/a2o/tool-start", {}, timeout=30)
    assert prep.get("started") is True, f"tool-start failed: {prep}"
    time.sleep(max(0.0, gap_ms / 1000.0 - (time.monotonic() - t_begin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--output", type=Path, default=OUT.with_name(f"{OUT.name}-{time.time_ns()}"))
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.relative_to(ROOT / "results")
    manifest = json.loads((build / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    svc_dir = output / "service"
    cmd_base = [
        sys.executable, str(ROOT / "scripts/run/route_phone.py"),
    ]
    start_cmd = cmd_base + [
        "start", "--condition", "route", "--output", str(svc_dir), "--logs-only", "--retry-network",
        "--runtime-overlay", str(build),
        "--model", MODEL_CFG["model_path"],
        "--model-id", MODEL_CFG["model_id"],
        "--model-sha256", MODEL_CFG["model_sha256"],
        "--context-size", "12288",
        "--cpu-repack", "0",
        "--shared-weights",
        "--kv-reduce", "1",
        "--kv-policy", "prefill_prior",
        "--kv-recent-tokens", "32",
        "--kv-tail-tokens", "16",
        "--compact-anchor", "1",
        "--kv-hwm-trim", "0",
        "--kv-layer-stride", "4",
        "--kv-min-evict-tokens", "256",
        "--m1-explorer", "1",
        "--kv-costs", str(KV_COSTS),
        "--route-costs", str(ROUTE_COSTS),
        "--device", "xiaomi14pro",
        "--gap-ms", "1000",
    ]
    stop_cmd = cmd_base + ["stop", "--condition", "route", "--output", str(svc_dir), "--logs-only", "--retry-network"]

    identity = {key: manifest[key] for key in ("build_id", "git_commit", "artifacts")}
    results = {"build": identity, "forced": {}, "joint": {}}
    try:
        subprocess.run(start_cmd, cwd=ROOT, check=True)
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            health = json.load(r)
        launch = json.loads((svc_dir / "manifest.json").read_text())
        assert health["build_id"] == identity["build_id"], health
        assert health["git_commit"] == identity["git_commit"], health
        assert launch["runtime_overlay"] == identity["artifacts"], launch
        assert health["m1_explorer"] is True, health
        assert health["kv_cost_loaded"] is True, health
        results["health"] = health

        # Find a long-H case (>= 2500 tokens) and a moderate-H case
        long_case = pick_case_with_min_h(2500)
        t1 = long_case["turns"][1]

        # 1. Forced CC
        prime_and_prepare(long_case)
        b_cc = dict(t1["request"], max_tokens=t1["D"], expected_decode_tokens=t1["D"], frozen_force_path="cc")
        r_cc = _agent_chat_resilient(BASE, b_cc, "t1-forced-cc", total_timeout=600)
        tim_cc = r_cc["timings"]
        assert tim_cc["route"] == "cpu" and tim_cc["kv_reduce"] is None and tim_cc["handoff_bytes"] == 0, tim_cc
        results["forced"]["cc"] = {"route": tim_cc["route"], "reason": tim_cc["route_reason"], "handoff_bytes": tim_cc["handoff_bytes"]}

        # 2. Forced HC-Full
        prime_and_prepare(long_case)
        b_hcf = dict(t1["request"], max_tokens=t1["D"], expected_decode_tokens=t1["D"], frozen_force_path="hc_full")
        r_hcf = _agent_chat_resilient(BASE, b_hcf, "t1-forced-hc-full", total_timeout=600)
        tim_hcf = r_hcf["timings"]
        assert tim_hcf["route"] == "htp" and tim_hcf["kv_reduce"] is None and tim_hcf["handoff_bytes"] > 0, tim_hcf
        results["forced"]["hc_full"] = {"route": tim_hcf["route"], "reason": tim_hcf["route_reason"], "handoff_bytes": tim_hcf["handoff_bytes"]}

        # 3. Forced HC-Compact
        prime_and_prepare(long_case)
        b_hcc = dict(
            t1["request"], max_tokens=t1["D"], expected_decode_tokens=t1["D"],
            frozen_force_path="hc_compact", frozen_kv_keep_tokens=1536,
        )
        r_hcc = _agent_chat_resilient(BASE, b_hcc, "t1-forced-hc-compact", total_timeout=600)
        tim_hcc = r_hcc["timings"]
        kvr = tim_hcc.get("kv_reduce") or {}
        assert tim_hcc["route"] == "htp" and kvr.get("compacted") is True and kvr.get("evicted_count", 0) > 0 and tim_hcc["handoff_bytes"] > 0, tim_hcc
        results["forced"]["hc_compact"] = {
            "route": tim_hcc["route"], "reason": tim_hcc["route_reason"],
            "handoff_bytes": tim_hcc["handoff_bytes"], "evicted_count": kvr.get("evicted_count"),
            "kept_count": kvr.get("kept_count"),
        }

        # 4. Joint 3-way routing under M1-Explorer (same build, no frozen_force_path)
        # 4a. Trigger CC via short user follow-up (P < routing_p_min=32)
        short_p_case = min(
            (c for c in FIXTURES["cases"] if len(c["turns"]) >= 2),
            key=lambda c: c["turns"][1]["P"],
        )
        _http_post(BASE + "/a2o/reset", {}, timeout=60)
        t0_s = short_p_case["turns"][0]
        p0_req = {"messages": t0_s["request"]["messages"], "max_tokens": 4, "expected_decode_tokens": 4}
        if "tools" in t0_s["request"]:
            p0_req["tools"] = t0_s["request"]["tools"]
        r0_s = _agent_chat_resilient(BASE, p0_req, "t1-joint-cc-prime", total_timeout=600)
        t_begin = time.monotonic()
        prep = _http_post(BASE + "/a2o/tool-start", {}, timeout=30)
        assert prep.get("started") is True, f"tool-start failed: {prep}"
        time.sleep(max(0.0, 1.0 - (time.monotonic() - t_begin)))
        msgs_cc = list(t0_s["request"]["messages"]) + [r0_s["choices"][0]["message"], {"role": "user", "content": "ok"}]
        cc_req = {"messages": msgs_cc, "max_tokens": 8, "expected_decode_tokens": 8}
        if "tools" in t0_s["request"]:
            cc_req["tools"] = t0_s["request"]["tools"]
        rj_cc = _agent_chat_resilient(BASE, cc_req, "t1-joint-cc", total_timeout=600)
        tj_cc = rj_cc["timings"]
        assert tj_cc["route"] == "cpu" and tj_cc["prediction"]["selected_path"] == "cc", tj_cc
        results["joint"]["cc"] = {"selected_path": tj_cc["prediction"]["selected_path"], "route": tj_cc["route"], "reason": tj_cc["route_reason"], "P_evaluated": tj_cc["P_evaluated"], "m1_online_updated": tj_cc["m1_online_updated"]}

        # 4b. Trigger HC-Full (fast case with P=123 >> crossover, H+P=694 < k_min -> hc_full)
        hc_full_case = next(c for c in FIXTURES["cases"] if c["id"] == "send_message_with_contact_content_cellular_off")
        t1_lp = hc_full_case["turns"][1]
        prime_and_prepare(hc_full_case)
        rj_hcf = _agent_chat_resilient(BASE, dict(t1_lp["request"], max_tokens=t1_lp["D"], expected_decode_tokens=t1_lp["D"]), "t1-joint-hc-full", total_timeout=600)
        tj_hcf = rj_hcf["timings"]
        assert tj_hcf["route"] == "htp" and tj_hcf["kv_reduce"] is None and tj_hcf["prediction"]["selected_path"] == "hc_full", tj_hcf
        results["joint"]["hc_full"] = {"selected_path": tj_hcf["prediction"]["selected_path"], "route": tj_hcf["route"], "reason": tj_hcf["route_reason"], "m1_online_updated": tj_hcf["m1_online_updated"]}

        # 4c. Trigger HC-Compact (expected_decode_tokens=256 on long_case -> compaction wins)
        prime_and_prepare(long_case)
        rj_hcc = _agent_chat_resilient(BASE, dict(t1["request"], max_tokens=t1["D"], expected_decode_tokens=256), "t1-joint-hc-compact", total_timeout=600)
        tj_hcc = rj_hcc["timings"]
        assert tj_hcc["route"] == "htp" and (tj_hcc.get("kv_reduce") or {}).get("compacted") is True and tj_hcc["prediction"]["selected_path"] == "hc_compact", tj_hcc
        results["joint"]["hc_compact"] = {
            "selected_path": tj_hcc["prediction"]["selected_path"], "route": tj_hcc["route"],
            "reason": tj_hcc["route_reason"], "h_active": tj_hcc["prediction"]["h_active"],
            "evicted_count": tj_hcc["kv_reduce"]["evicted_count"], "m1_online_updated": tj_hcc["m1_online_updated"],
        }

        # 5. Verify HTP not-ready fallback stays strictly CC (no CPU-side KV compaction)
        _http_post(BASE + "/a2o/reset", {}, timeout=60)
        t0_nr = long_case["turns"][0]
        _agent_chat_resilient(BASE, dict(t0_nr["request"], max_tokens=t0_nr["D"], expected_decode_tokens=t0_nr["D"]), "t1-not-ready-prime", total_timeout=600)
        t1_nr = long_case["turns"][1]
        r_nr = _agent_chat_resilient(BASE, dict(t1_nr["request"], max_tokens=t1_nr["D"], expected_decode_tokens=t1_nr["D"]), "t1-not-ready-cc", total_timeout=600)
        t_nr = r_nr["timings"]
        assert t_nr["route"] == "cpu" and t_nr["kv_reduce"] is None, t_nr
        results["not_ready_fallback"] = {"route": t_nr["route"], "reason": t_nr["route_reason"], "kv_reduce": t_nr["kv_reduce"]}

        # 6. Verify /a2o/calibrate include_hc_compact (paired CC / HC-Full / HC-Compact at H=2048, P=64)
        cal_ids = t1_nr["request"]["frozen_prompt_ids"]
        cal_res = _http_post(BASE + "/a2o/calibrate", {
            "prefix_ids": cal_ids[:2048], "input_ids": cal_ids[2048:2048 + 64],
            "decode_evaluations": 1, "repetitions": 1,
            "include_hc_compact": True, "keep_tokens": 1536,
        }, timeout=600)
        cal_row = cal_res["rows"][0]
        assert cal_row["hc_compact_evicted_count"] > 0 and cal_row["hc_compact_handoff_bytes"] > 0, cal_row
        results["calibrate_3way_smoke"] = {
            "cc_ms": cal_row["cc_ms"], "hc_ms": cal_row["hc_ms"], "hc_compact_ms": cal_row["hc_compact_ms"],
            "hc_compact_kept_count": cal_row["hc_compact_kept_count"],
            "hc_compact_evicted_count": cal_row["hc_compact_evicted_count"],
        }
    finally:
        subprocess.run(stop_cmd, cwd=ROOT, check=True)

    runtime_log = (svc_dir / "device/runtime.log").read_text()
    meta_lines = [line for line in runtime_log.splitlines() if line.startswith("A2O_META ")]
    assert meta_lines, "Missing A2O_META in runtime.log"
    meta = json.loads(meta_lines[0].split("A2O_META ", 1)[1])
    for key in (
        "git_commit", "build_id", "device", "model", "shared_weights",
        "compact_anchor", "kv_hwm_trim", "kv_layer_stride", "kv_min_evict_tokens", "gap_ms",
    ):
        assert meta[key] == health[key], f"Runtime and health metadata differ: {key}"
    results["runtime_meta"] = meta
    (output / "dod-report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
