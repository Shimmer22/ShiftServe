"""Collect paired tau2-bench HC + PrefillPrior KV Reduce campaign on Qwen3-1.7B-Q4_0."""
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
from tau2_hc_kv import summarize

ENDPOINT = "http://10.29.131.50:18780"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_health():
    with urllib.request.urlopen(ENDPOINT + "/health", timeout=30) as resp:
        return json.loads(resp.read().decode())


def route_service(action: str, cfg: dict, svc_cfg: dict | None, svc_dir: Path, log_path: Path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run/route_phone.py"),
        action,
        "--condition",
        "route",
        "--output",
        str(svc_dir),
        "--retry-network",
    ]
    if action == "start":
        assert svc_cfg is not None
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
            "--kv-reduce", str(svc_cfg["kv_reduce"]),
            "--kv-policy", svc_cfg["kv_policy"],
            "--kv-keep-tokens", str(cfg["kv_keep_tokens"]),
            "--kv-sink-tokens", str(cfg["kv_sink_tokens"]),
            "--kv-recent-tokens", str(svc_cfg["kv_recent_tokens"]),
            "--kv-pool-window", str(cfg["kv_pool_window"]),
            "--kv-tail-tokens", str(cfg["kv_tail_tokens"]),
            "--replay-cache",
        ]
    else:
        cmd += ["--logs-only"]
    with log_path.open("a") as logf:
        subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_measurement(meas_dir: Path, key: str, body: dict, meta: dict):
    meas_dir.mkdir(parents=True, exist_ok=True)
    path = meas_dir / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved["request"] == body, f"Request mismatch for cached {key}"
        return saved["response"]

    response = retry_network(lambda: post(ENDPOINT, "/a2o/gap-replay", body, 1800))
    raw = meas_dir / f"{key}.raw.json"
    if raw.exists():
        idx = len(list(meas_dir.glob(f"{key}*.raw.json")))
        raw = meas_dir / f"{key}.attempt-{idx:02d}.raw.json"
    record = {"meta": meta, "request": body, "response": response}
    save_json(raw, record)

    assert response["mode"] == "always_prepare_gap_replay_v2"
    assert [r["condition"] for r in response["results"]] == body["order"]
    for row in response["results"]:
        assert (row["H"], row["P"], row["D"]) == (
            len(body["prefix_ids"]),
            len(body["input_ids"]),
            len(body["generated_ids"]),
        )
        for k in [
            "actual_gap_ms",
            "prepare_ms",
            "wait_ms",
            "restore_ms",
            "prefill_ms",
            "evict_ms",
            "handoff_ms",
            "decode_ms",
            "post_return_ms",
            "retirement_tail_ms",
        ]:
            assert math.isfinite(row[k]) and row[k] >= 0, f"Invalid {k}: {row[k]}"
        if row["condition"] == "route":
            assert row["route"] == "htp", f"Expected forced HC route=htp, got {row['route']}"
            if body.get("kv_reduce"):
                assert row["decode_n_kv"] <= body["kv_keep_tokens"], (
                    f"KV reduce did not compact n_kv: {row['decode_n_kv']} > {body['kv_keep_tokens']}"
                )
        if body.get("verify_free_generation"):
            fg = row.get("free_generation")
            assert fg and len(fg.get("tokens", [])) > 0, "Missing free_generation"
    save_json(path, record)
    return response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/tau2-qwen3-1_7b-hc-kv.json")
    args = p.parse_args()

    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    out = (ROOT / cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    guard = (out / "driver.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(phase, **kw):
        save_json(out / "status.json", {"phase": phase, "pid": __import__("os").getpid(), "updated_unix": time.time(), **kw})

    binary_path = (ROOT / cfg["binary"]).resolve()
    reqs_path = (ROOT / cfg["requests"]).resolve()
    reqs = json.loads(reqs_path.read_text())

    plan = {
        "name": cfg["name"],
        "config_sha256": sha256(cfg_path),
        "binary_sha256": sha256(binary_path),
        "requests_sha256": sha256(reqs_path),
        "driver_sha256": sha256(Path(__file__)),
        "rounds": cfg["rounds"],
        "boundaries": len(reqs),
        "services": [s["name"] for s in cfg["services"]],
    }
    plan_path = out / "plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Existing plan.json differs; use a clean output_dir or matching config"
    else:
        save_json(plan_path, plan)

    if (out / "completion.json").exists():
        print("Campaign already completed:", out / "completion.json")
        return

    # Forced-HC routing prediction so Mechanism 1 -> Mechanism 2 path always exercises HC on Service A/B/C
    forced_hc_pred = {
        "validated": True,
        "cpu_prefill_ms": 15000.0,
        "cpu_restore_ms": 500.0,
        "htp_prefill_handoff_ms": 500.0,
        "prepare_ms": 1000.0,
    }

    variant_map = {
        "baseline_a": "hc_baseline",
        "prior_b": "hc_prefill_prior",
        "recent_c": "hc_recent",
    }

    try:
        for svc_cfg in cfg["services"]:
            svc_name = svc_cfg["name"]
            svc_dir = out / f"service-{svc_name}"
            svc_log = out / f"service-{svc_name}.log"
            status("starting_service", service=svc_name)
            route_service("start", cfg, svc_cfg, svc_dir, svc_log)
            try:
                health = get_health()
                save_json(out / f"health-{svc_name}.json", health)
                assert bool(health["kv_reduce"]) == bool(svc_cfg["kv_reduce"])
                if svc_cfg["kv_reduce"]:
                    assert health["kv_policy"] == svc_cfg["kv_policy"]

                for rnd in range(cfg["rounds"]):
                    # Reverse boundary scan order on odd rounds to balance thermal drift
                    b_indices = list(range(len(reqs))) if rnd % 2 == 0 else list(reversed(range(len(reqs))))
                    for b_idx in b_indices:
                        bx = reqs[b_idx]
                        modes = list(svc_cfg["modes"])
                        if len(modes) > 1 and (b_idx + rnd) % 2 == 1:
                            modes = list(reversed(modes))

                        verify_fg = rnd == 0
                        body = {
                            "prefix_ids": bx["prefix_ids"],
                            "input_ids": bx["input_ids"],
                            "generated_ids": bx["generated_ids"],
                            "order": modes,
                            "allow_partial_order": True,
                            "gap_ms": float(cfg["ready_gap_ms"]),
                            "return_policy": "greedy",
                            "prediction": forced_hc_pred,
                            "kv_reduce": bool(svc_cfg["kv_reduce"]),
                            "kv_reduce_on_cpu": False,
                            "kv_reduce_policy": svc_cfg["kv_policy"],
                            "kv_keep_tokens": int(cfg["kv_keep_tokens"]),
                            "kv_sink_tokens": int(cfg["kv_sink_tokens"]),
                            "kv_recent_tokens": int(svc_cfg["kv_recent_tokens"]),
                            "kv_pool_window": int(cfg["kv_pool_window"]),
                            "kv_tail_tokens": int(cfg["kv_tail_tokens"]),
                            "verify_free_generation": verify_fg,
                            "free_tokens": int(cfg["free_tokens"]),
                        }
                        meta = {
                            "service": svc_name,
                            "variant_name": variant_map[svc_name],
                            "gap_mode": "ready",
                            "boundary": bx["boundary"],
                            "domain": bx["domain"],
                            "task_id": bx["task_id"],
                            "round": rnd,
                        }
                        status("measuring", service=svc_name, gap_mode="ready", round=rnd, boundary=bx["boundary"], H=bx["H"], P=bx["P"], D=bx["D"])
                        resp = run_measurement(out / "measurements", f"{svc_name}-ready-r{rnd:02d}-b{bx['boundary']:02d}", body, meta)
                        summary_line = [
                            f"{r['condition']}({r['route']}):post={r['post_return_ms']:.0f}ms,nkv={r['decode_n_kv']},hnd={r['handoff_bytes']}B"
                            for r in resp["results"]
                        ]
                        print(f"[{svc_name} ready r{rnd} b{bx['boundary']}] " + " | ".join(summary_line), flush=True)

                        # Real gap_ms check (2.1..3.2 ms external wait -> real prepare wait_ms in HC)
                        if svc_cfg.get("include_real_gap") and rnd == 0:
                            rg_body = dict(body)
                            rg_body["order"] = ["route"]
                            rg_body["gap_ms"] = float(bx["gap_ms"])
                            rg_body["verify_free_generation"] = False
                            rg_meta = dict(meta)
                            rg_meta["gap_mode"] = "real_gap"
                            status("measuring", service=svc_name, gap_mode="real_gap", round=rnd, boundary=bx["boundary"], H=bx["H"], P=bx["P"], D=bx["D"])
                            rg_resp = run_measurement(out / "measurements", f"{svc_name}-realgap-r{rnd:02d}-b{bx['boundary']:02d}", rg_body, rg_meta)
                            r0 = rg_resp["results"][0]
                            print(
                                f"[{svc_name} real_gap r{rnd} b{bx['boundary']}] gap={r0['requested_gap_ms']:.2f}ms "
                                f"prep={r0['prepare_ms']:.1f}ms wait={r0['wait_ms']:.1f}ms post={r0['post_return_ms']:.0f}ms",
                                flush=True,
                            )
            finally:
                status("stopping_service", service=svc_name)
                route_service("stop", cfg, None, svc_dir, svc_log)

        status("analyzing")
        summary = summarize(out)
        save_json(out / "completion.json", {"completed": True, **summary})
        status("complete")
        print("CAMPAIGN_COMPLETE", json.dumps(summary.get("hc_kv_vs_hc_baseline", {})))
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
