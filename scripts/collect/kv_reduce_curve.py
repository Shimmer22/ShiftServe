"""Collect quantitative time, history-length, keep-budget, and quality curves on Qwen3-1.7B."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analyze"))

from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import post
from kv_reduce_curve import summarize

ENDPOINT = "http://10.29.131.50:18780"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_health():
    with urllib.request.urlopen(ENDPOINT + "/health", timeout=30) as resp:
        return json.loads(resp.read().decode())


def route_service(action: str, cfg: dict, condition: str, kv_reduce: int, svc_dir: Path, log_path: Path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run/route_phone.py"),
        action,
        "--condition",
        condition,
        "--output",
        str(svc_dir),
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
            "--kv-reduce", str(kv_reduce),
            "--kv-policy", "prefill_prior",
            "--kv-keep-tokens", "1024",
            "--kv-sink-tokens", str(cfg["kv_sink_tokens"]),
            "--kv-recent-tokens", "256",
            "--kv-pool-window", str(cfg["kv_pool_window"]),
            "--kv-tail-tokens", str(cfg["kv_tail_default"]),
            "--replay-cache",
        ]
    else:
        cmd += ["--logs-only"]
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_gap_replay(meas_dir: Path, key: str, body: dict, meta: dict):
    meas_dir.mkdir(parents=True, exist_ok=True)
    path = meas_dir / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved["request"] == body, f"Cached request mismatch for {key}"
        return saved["response"]

    response = retry_network(lambda: post(ENDPOINT, "/a2o/gap-replay", body, 1800))
    record = {"meta": meta, "request": body, "response": response}
    assert response["mode"] == "always_prepare_gap_replay_v2"
    assert [r["condition"] for r in response["results"]] == body["order"]
    for row in response["results"]:
        assert (row["H"], row["P"], row["D"]) == (
            len(body["prefix_ids"]),
            len(body["input_ids"]),
            len(body["generated_ids"]),
        )
        for k in ["prefill_ms", "evict_ms", "handoff_ms", "decode_ms", "post_return_ms"]:
            assert math.isfinite(row[k]) and row[k] >= 0
    save_json(path, record)
    return response


def make_body(prefix_ids, input_ids, generated_ids, order, gap_ms, kv_reduce, kv_keep, kv_tail, kv_sink=4, kv_pool=5, verify_fg=False, free_tokens=128):
    recent = min(256, max(32, kv_keep // 4))
    return {
        "prefix_ids": prefix_ids,
        "input_ids": input_ids,
        "generated_ids": generated_ids,
        "order": order,
        "allow_partial_order": True,
        "gap_ms": float(gap_ms),
        "return_policy": "greedy",
        "prediction": {
            "validated": True,
            "cpu_prefill_ms": 15000.0,
            "cpu_restore_ms": 500.0,
            "htp_prefill_handoff_ms": 500.0,
            "prepare_ms": 1000.0,
        },
        "kv_reduce": bool(kv_reduce),
        "kv_reduce_on_cpu": False,
        "kv_reduce_policy": "prefill_prior",
        "kv_keep_tokens": int(kv_keep),
        "kv_sink_tokens": int(kv_sink),
        "kv_recent_tokens": int(recent),
        "kv_pool_window": int(kv_pool),
        "kv_tail_tokens": int(kv_tail),
        "verify_free_generation": bool(verify_fg),
        "free_tokens": int(free_tokens),
    }


def run_cpu_probes(cfg: dict, out: Path, status):
    meas_dir = out / "measurements"
    filler = "Status entry: node=alpha, queue_depth=12, cpu_util=41%, mem_free=8420MB, link=healthy.\n"
    for kv_reduce in [0, 1]:
        svc_name = f"cpu_probe_kv{kv_reduce}"
        svc_dir = out / f"service-{svc_name}"
        svc_log = out / f"service-{svc_name}.log"
        probe_path = meas_dir / f"cpu-probe-kv{kv_reduce}.json"
        if probe_path.exists():
            continue
        status("starting_service", service=svc_name)
        route_service("start", cfg, "cpu", kv_reduce, svc_dir, svc_log)
        try:
            health = get_health()
            assert health["condition"] == "cpu" and bool(health["kv_reduce"]) == bool(kv_reduce)
            results = []
            # Calibrate tokens per filler line once
            n_unit = len(post(ENDPOINT, "/tokenize", {"content": filler})["tokens"])
            for h_target in cfg["cpu_probe_h_targets"]:
                lines = max(1, (h_target - 80) // n_unit)
                history_txt = "System diagnostics log:\n" + (filler * lines)
                msg0 = [{"role": "user", "content": history_txt + "Acknowledge log receipt in one word."}]
                msg1 = msg0 + [
                    {"role": "assistant", "content": "Received."},
                    {"role": "user", "content": "Summarize in 64 words the repeated node status pattern."},
                ]
                for rnd in range(2):
                    post(ENDPOINT, "/a2o/reset", {})
                    r0 = post(ENDPOINT, "/v1/chat/completions", {"messages": msg0, "temperature": 0.0, "max_tokens": 8, "cache_prompt": True})
                    r1 = post(ENDPOINT, "/v1/chat/completions", {"messages": msg1, "temperature": 0.0, "max_tokens": cfg["cpu_probe_d"], "cache_prompt": True})
                    t0 = r0["timings"]
                    t1 = r1["timings"]
                    results.append({
                        "kv_reduce": kv_reduce,
                        "h_target": h_target,
                        "round": rnd,
                        "turn0_prompt_n": t0["prompt_n"],
                        "turn0_prompt_ms": t0["prompt_ms"],
                        "turn0_kv_reduce_ms": t0["kv_reduce_ms"],
                        "turn1_H_anchor": t1["H_anchor"],
                        "turn1_prompt_n": t1["prompt_n"],
                        "turn1_prompt_ms": t1["prompt_ms"],
                        "turn1_predicted_n": t1["predicted_n"],
                        "turn1_predicted_ms": t1["predicted_ms"],
                        "turn1_kv_reduce_ms": t1["kv_reduce_ms"],
                        "turn1_kv_reduce": t1["kv_reduce"],
                    })
                    print(
                        f"[cpu_probe kv={kv_reduce} H~{h_target} r{rnd}] "
                        f"t0_pre={t0['prompt_ms']:.1f}ms t1_H={t1['H_anchor']} t1_pre={t1['prompt_ms']:.1f}ms "
                        f"t1_dec={t1['predicted_ms']:.1f}ms ({t1['predicted_n']}tok)",
                        flush=True,
                    )
            save_json(probe_path, {"meta": {"kind": "cpu_probe", "kv_reduce": kv_reduce}, "rows": results})
        finally:
            status("stopping_service", service=svc_name)
            route_service("stop", cfg, "cpu", 0, svc_dir, svc_log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/qwen3-1_7b-kv-curve.json")
    args = p.parse_args()

    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    out = (ROOT / cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = (out / "driver.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(phase, **kw):
        save_json(out / "status.json", {"phase": phase, "pid": __import__("os").getpid(), "updated_unix": time.time(), **kw})

    reqs = json.loads((ROOT / cfg["requests"]).resolve().read_text())
    # Real natural tau2 token stream (>= 8704 tokens) built cleanly from b3 & b2
    b3 = reqs[3]
    b2 = reqs[2]
    stream = (b3["prefix_ids"] + b3["input_ids"] + b2["input_ids"] + b2["prefix_ids"])[:10000]
    gen_stream = b3["generated_ids"]

    plan = {
        "name": cfg["name"],
        "config_sha256": sha256(cfg_path),
        "binary_sha256": sha256((ROOT / cfg["binary"]).resolve()),
        "driver_sha256": sha256(Path(__file__)),
    }
    plan_path = out / "plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Existing plan.json differs"
    else:
        save_json(plan_path, plan)

    if (out / "completion.json").exists():
        print("Curve scan already completed:", out / "completion.json")
        return

    try:
        # 1. Service A: Baseline (no cb_eval) -> H-sweep and P-sweep
        svc_a = out / "service-baseline_a"
        log_a = out / "service-baseline_a.log"
        status("starting_service", service="baseline_a")
        route_service("start", cfg, "route", 0, svc_a, log_a)
        try:
            for rnd in range(cfg["grid_rounds"]):
                for idx, h in enumerate(cfg["h_sweep"] if rnd % 2 == 0 else list(reversed(cfg["h_sweep"]))):
                    order = ["cpu", "route"] if (idx + rnd) % 2 == 0 else ["route", "cpu"]
                    pval = cfg["h_sweep_p"]
                    dval = cfg["h_sweep_d"]
                    body = make_body(stream[:h], stream[h:h+pval], gen_stream[:dval], order, cfg["ready_gap_ms"], False, 1024, cfg["kv_tail_default"])
                    meta = {"kind": "h_sweep", "service": "baseline_a", "round": rnd, "kv_keep": h + pval, "kv_tail": 0}
                    status("measuring", kind="h_sweep", service="baseline_a", round=rnd, H=h, P=pval)
                    resp = run_gap_replay(out / "measurements", f"base-h{h}-p{pval}-r{rnd:02d}", body, meta)
                    print(f"[base h_sweep H={h} P={pval} r{rnd}] " + " | ".join(f"{r['condition']}:pre={r['prefill_ms']:.0f},hnd={r['handoff_ms']:.0f},dec={r['decode_ms']:.0f}" for r in resp["results"]), flush=True)

                for idx, pval in enumerate(cfg["p_sweep"]):
                    if pval == cfg["h_sweep_p"]:
                        continue  # Already measured in h_sweep at H=4096
                    h = cfg["p_sweep_h"]
                    dval = cfg["p_sweep_d"]
                    order = ["cpu", "route"] if (idx + rnd) % 2 == 0 else ["route", "cpu"]
                    body = make_body(stream[:h], stream[h:h+pval], gen_stream[:dval], order, cfg["ready_gap_ms"], False, 1024, cfg["kv_tail_default"])
                    meta = {"kind": "p_sweep", "service": "baseline_a", "round": rnd, "kv_keep": h + pval, "kv_tail": 0}
                    status("measuring", kind="p_sweep", service="baseline_a", round=rnd, H=h, P=pval)
                    resp = run_gap_replay(out / "measurements", f"base-p{pval}-h{h}-r{rnd:02d}", body, meta)
                    print(f"[base p_sweep H={h} P={pval} r{rnd}] " + " | ".join(f"{r['condition']}:pre={r['prefill_ms']:.0f}" for r in resp["results"]), flush=True)
        finally:
            status("stopping_service", service="baseline_a")
            route_service("stop", cfg, "route", 0, svc_a, log_a)

        # 2. Service B: PrefillPrior (cb_eval attached on HTP) -> H, P, tail, K, and tau2 quality curves
        svc_b = out / "service-prior_b"
        log_b = out / "service-prior_b.log"
        status("starting_service", service="prior_b")
        route_service("start", cfg, "route", 1, svc_b, log_b)
        try:
            for rnd in range(cfg["grid_rounds"]):
                for h in (cfg["h_sweep"] if rnd % 2 == 0 else list(reversed(cfg["h_sweep"]))):
                    pval = cfg["h_sweep_p"]
                    dval = cfg["h_sweep_d"]
                    kval = cfg["h_sweep_k"]
                    body = make_body(stream[:h], stream[h:h+pval], gen_stream[:dval], ["route"], cfg["ready_gap_ms"], True, kval, cfg["kv_tail_default"])
                    meta = {"kind": "h_sweep", "service": "prior_b", "round": rnd, "kv_keep": kval, "kv_tail": cfg["kv_tail_default"]}
                    status("measuring", kind="h_sweep", service="prior_b", round=rnd, H=h, P=pval)
                    resp = run_gap_replay(out / "measurements", f"prior-h{h}-p{pval}-r{rnd:02d}", body, meta)
                    r0 = resp["results"][0]
                    print(f"[prior h_sweep H={h} P={pval} r{rnd}] pre={r0['prefill_ms']:.0f} ev={r0['evict_ms']:.1f} hnd={r0['handoff_ms']:.0f} dec={r0['decode_ms']:.0f} nkv={r0['decode_n_kv']}", flush=True)

                for pval in cfg["p_sweep"]:
                    if pval == cfg["h_sweep_p"]:
                        continue
                    h = cfg["p_sweep_h"]
                    dval = cfg["p_sweep_d"]
                    body = make_body(stream[:h], stream[h:h+pval], gen_stream[:dval], ["route"], cfg["ready_gap_ms"], True, 1024, cfg["kv_tail_default"])
                    meta = {"kind": "p_sweep", "service": "prior_b", "round": rnd, "kv_keep": 1024, "kv_tail": cfg["kv_tail_default"]}
                    status("measuring", kind="p_sweep", service="prior_b", round=rnd, H=h, P=pval)
                    resp = run_gap_replay(out / "measurements", f"prior-p{pval}-h{h}-r{rnd:02d}", body, meta)
                    r0 = resp["results"][0]
                    print(f"[prior p_sweep H={h} P={pval} r{rnd}] pre={r0['prefill_ms']:.0f}", flush=True)

                for tv in cfg["tail_sweep"]:
                    if tv == cfg["kv_tail_default"]:
                        continue
                    h = cfg["tail_sweep_h"]
                    pval = cfg["tail_sweep_p"]
                    body = make_body(stream[:h], stream[h:h+pval], gen_stream[:cfg["p_sweep_d"]], ["route"], cfg["ready_gap_ms"], True, 1024, tv)
                    meta = {"kind": "tail_sweep", "service": "prior_b", "round": rnd, "kv_keep": 1024, "kv_tail": tv}
                    status("measuring", kind="tail_sweep", service="prior_b", round=rnd, tail=tv)
                    resp = run_gap_replay(out / "measurements", f"prior-tail{tv}-r{rnd:02d}", body, meta)
                    r0 = resp["results"][0]
                    print(f"[prior tail={tv} r{rnd}] pre={r0['prefill_ms']:.0f} ev={r0['evict_ms']:.1f}", flush=True)

                for kval in (cfg["k_sweep"] if rnd % 2 == 0 else list(reversed(cfg["k_sweep"]))):
                    h = cfg["k_sweep_h"]
                    pval = cfg["k_sweep_p"]
                    dval = cfg["k_sweep_d"]
                    body = make_body(stream[:h], stream[h:h+pval], gen_stream[:dval], ["route"], cfg["ready_gap_ms"], True, kval, cfg["kv_tail_default"])
                    meta = {"kind": "k_sweep", "service": "prior_b", "round": rnd, "kv_keep": kval, "kv_tail": cfg["kv_tail_default"]}
                    status("measuring", kind="k_sweep", service="prior_b", round=rnd, H=h, K=kval)
                    resp = run_gap_replay(out / "measurements", f"prior-k{kval}-h{h}-r{rnd:02d}", body, meta)
                    r0 = resp["results"][0]
                    print(f"[prior k_sweep H={h} K={kval} r{rnd}] ev={r0['evict_ms']:.1f} hnd={r0['handoff_ms']:.0f}({r0['handoff_bytes']}B) dec={r0['decode_ms']:.0f} nkv={r0['decode_n_kv']}", flush=True)

            # Quality inflection sweep across tau2-bench b0..b5
            for bx in reqs:
                b = bx["boundary"]
                for kval in cfg["quality_k_sweep"]:
                    body = make_body(
                        bx["prefix_ids"],
                        bx["input_ids"],
                        bx["generated_ids"][:32],
                        ["route"],
                        cfg["ready_gap_ms"],
                        True,
                        kval,
                        cfg["kv_tail_default"],
                        verify_fg=True,
                        free_tokens=cfg["quality_free_tokens"],
                    )
                    meta = {"kind": "quality_k", "service": "prior_b", "round": 0, "boundary": b, "domain": bx["domain"], "kv_keep": kval, "kv_tail": cfg["kv_tail_default"]}
                    status("measuring", kind="quality_k", boundary=b, K=kval, H=bx["H"], P=bx["P"])
                    resp = run_gap_replay(out / "measurements", f"qual-b{b:02d}-k{kval}", body, meta)
                    r0 = resp["results"][0]
                    print(f"[quality b{b} (H+P={bx['H']+bx['P']}) K={kval}] nkv={r0['decode_n_kv']} dec={r0['decode_ms']:.0f}ms top1={r0['free_generation']['top1_first']}", flush=True)
        finally:
            status("stopping_service", service="prior_b")
            route_service("stop", cfg, "route", 0, svc_b, log_b)

        # 3. CPU single-residency cb_eval & KV-reduce probe
        run_cpu_probes(cfg, out, status)

        status("analyzing")
        summary = summarize(out, (ROOT / cfg["baseline_campaign"]).resolve())
        save_json(out / "completion.json", {"completed": True, **summary})
        status("complete")
        print("CURVE_SCAN_COMPLETE", out / "summary.json")
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
