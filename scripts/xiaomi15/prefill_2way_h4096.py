#!/usr/bin/env python3
"""Xiaomi 15 (SM8750, Hexagon v79) Qwen3-8B Q4_0 2-way (shared_cpu vs independent_htp)
Prefill + M(H) Handoff sweep at H=4096, P=16..48 (step 2), n=10 interleaved AB/BA repeats.
"""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SSH_CFG = ROOT / "configs/xiaomi15/termux-ssh.json"
DEFAULT_MODEL_CFG = ROOT / "configs/model-default.json"
DEFAULT_FIXTURE = ROOT / "results/mainline-eval/prefill-motivation-h4096-8b/fixture.json"
DEFAULT_OUT = ROOT / "results/xiaomi15/prefill-2way-h4096-8b"

H_TOKENS = 4096
P_POINTS = tuple(range(16, 49, 2))  # 16, 18, ..., 48 (17 points)
HANDOFF_P_POINTS = (0, 16, 32, 48)
PORT = 18780
CONTEXT_SIZE = 8192
CPU_THREADS = 4
HTP_HOST_THREADS = 4


def read_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


class X15Phone:
    def __init__(self, ssh_cfg_path: Path):
        self.cfg = read_json(ssh_cfg_path)
        self.host = self.cfg["host"]
        self.port = str(self.cfg["port"])
        self.user = self.cfg["user"]
        self.key = str(Path(self.cfg["identity_file"]).expanduser())
        kh = Path(self.cfg["known_hosts"])
        self.known_hosts = str(kh if kh.is_absolute() else (ROOT / kh))
        self.remote_root = self.cfg["remote_root"].rstrip("/")
        self.runtime_lib = self.cfg["runtime_library_dir"].rstrip("/")
        self.base_url = f"http://{self.host}:{PORT}"
        self.ssh_opts = [
            "-p", self.port,
            "-i", self.key,
            "-o", "ConnectTimeout=8",
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
        ]

    def ssh(self, cmd: str, *, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
        full = ["ssh", *self.ssh_opts, f"{self.user}@{self.host}", cmd]
        last_err = None
        for attempt in range(4):
            try:
                cp = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
                if cp.returncode == 255 and attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
                if check and cp.returncode != 0:
                    raise RuntimeError(f"SSH failed ({cp.returncode}): {cp.stderr.strip() or cp.stdout.strip()}")
                return cp
            except subprocess.TimeoutExpired as exc:
                last_err = exc
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"SSH failed after retries: {last_err}")

    def pull(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        scp_opts = [
            "-P", self.port,
            "-i", self.key,
            "-o", "ConnectTimeout=8",
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
        ]
        for attempt in range(4):
            cp = subprocess.run(
                ["scp", *scp_opts, f"{self.user}@{self.host}:{remote_path}", str(local_path)],
                capture_output=True, text=True, timeout=60,
            )
            if cp.returncode == 0:
                return
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"SCP pull failed: {cp.stderr.strip()}")

    def post_ab(self, request: dict, timeout: int = 900) -> dict:
        body = json.dumps(request).encode()
        last_err = None
        for attempt in range(4):
            req = Request(
                f"{self.base_url}/a2o/prefill-weight-ab",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=timeout) as resp:
                    return json.load(resp)
            except (URLError, OSError, TimeoutError) as exc:
                last_err = exc
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"POST /a2o/prefill-weight-ab failed: {last_err}")

    def get_health(self, timeout: int = 3) -> dict:
        with urlopen(f"{self.base_url}/health", timeout=timeout) as resp:
            return json.load(resp)

    def stage_run_dir(self, run_dir: str) -> None:
        smoke_dir = f"{self.remote_root}/smoke-8b-v79"
        cmd = (
            f"termux-wake-lock || true; "
            f"mkdir -p {shlex.quote(run_dir)} {shlex.quote(run_dir + '/replay-cache')}; "
            f"for f in service libggml-cpu.so libllama.so; do "
            f"  ln -sfn {shlex.quote(smoke_dir)}/\"$f\" {shlex.quote(run_dir)}/\"$f\"; "
            f"done"
        )
        self.ssh(cmd)

    def start_service(self, run_dir: str, model_path: str, arm: str, log_name: str) -> dict:
        if arm == "prepare_cpu":
            cond, shared = "cpu", 0
        elif arm == "shared_cpu":
            cond, shared = "route", 1
        elif arm == "independent_htp":
            cond, shared = "htp", 0
        else:
            raise ValueError(f"Unknown arm: {arm}")

        self.stop_service(run_dir)
        lib = f"{run_dir}:{self.runtime_lib}:/vendor/lib64:/vendor/lib64/egl"
        remote_log = f"{run_dir}/{log_name}"
        remote_pid = f"{run_dir}/pid"
        replay_dir = f"{run_dir}/replay-cache"
        inner = (
            f"export LD_LIBRARY_PATH={shlex.quote(lib)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(self.runtime_lib)} "
            f"GGML_HEXAGON_USE_HMX=1 GGML_HEXAGON_EXPERIMENTAL=1 "
            f"GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1 "
            f"A2O_CPU_THREADS={CPU_THREADS} A2O_CPU_REPACK=0 "
            f"A2O_HTP_KV_OFFLOAD=0 A2O_KV_HWM_TRIM=0 "
            f"A2O_SHARED_WEIGHTS={shared} A2O_DEVICE=xiaomi15 "
            f"A2O_MODEL_ID=qwen3-8b-q4_0 "
            f"A2O_REPLAY_CACHE_DIR={shlex.quote(replay_dir)}; "
            f"exec {shlex.quote(run_dir + '/service')} {shlex.quote(model_path)} "
            f"{cond} {CONTEXT_SIZE} {PORT} none"
        )
        launch = (
            f"rm -f {shlex.quote(remote_log)} {shlex.quote(remote_pid)}; "
            f"nohup /system/bin/sh -c {shlex.quote(inner)} > {shlex.quote(remote_log)} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(remote_pid)}"
        )
        self.ssh(launch)

        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            try:
                health = self.get_health(timeout=3)
                if (health.get("status") != "ok"
                        or health.get("shared_weights") is not bool(shared)
                        or health.get("context_size") != CONTEXT_SIZE
                        or health.get("cpu_threads") != CPU_THREADS
                        or health.get("htp_host_threads") != HTP_HOST_THREADS
                        or health.get("device") != "xiaomi15"):
                    raise RuntimeError(f"Health mismatch for {arm}: {health}")
                return health
            except (URLError, OSError, TimeoutError):
                cp = self.ssh(
                    f"p=$(cat {shlex.quote(remote_pid)} 2>/dev/null); "
                    f"[ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null && echo alive || echo dead",
                    check=False,
                )
                if "dead" in cp.stdout:
                    log_tail = self.ssh(f"tail -n 40 {shlex.quote(remote_log)} 2>/dev/null || true", check=False).stdout
                    raise RuntimeError(f"Service exited early for {arm}:\n{log_tail}")
                time.sleep(1.5)
        raise RuntimeError(f"Service timed out waiting for /health ({arm})")

    def stop_service(self, run_dir: str) -> None:
        remote_pid = f"{run_dir}/pid"
        cmd = (
            f"p=$(cat {shlex.quote(remote_pid)} 2>/dev/null || true); "
            f"if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null; then "
            f"  kill \"$p\" 2>/dev/null || true; "
            f"  for i in $(seq 1 20); do kill -0 \"$p\" 2>/dev/null || break; sleep 1; done; "
            f"  kill -9 \"$p\" 2>/dev/null || true; "
            f"fi; "
            f"pkill -9 -x service 2>/dev/null || true; "
            f"rm -f {shlex.quote(remote_pid)}"
        )
        self.ssh(cmd, check=False)


def verify_runtime_log(log_path: Path, arm: str) -> None:
    txt = log_path.read_text(errors="replace")
    if arm == "shared_cpu":
        if "HTP0-REPACK model buffer size" not in txt or "CPU-only scheduler, shared packed weights" not in txt:
            raise RuntimeError(f"shared_cpu weight layout mismatch in {log_path}")
    elif arm == "independent_htp":
        if "HTP0-REPACK model buffer size" not in txt or "CPU_Mapped model buffer size" in txt:
            raise RuntimeError(f"independent_htp weight layout mismatch in {log_path}")


def validate_sample(res: dict, req: dict, arm: str, *, measure_handoff: bool = False) -> None:
    expected_backend = "htp" if (arm == "independent_htp" or measure_handoff) else "cpu"
    expected_shared = (arm == "shared_cpu") or measure_handoff
    if (res.get("mode") != "cpu_prefill_ab_v1"
            or res.get("backend") != expected_backend
            or res.get("shared_weights") is not expected_shared
            or bool(res.get("measure_handoff", False)) is not measure_handoff
            or res.get("snapshot_checksum") != req["snapshot_checksum"]
            or res.get("H") != len(req["prefix_ids"])
            or res.get("P") != len(req.get("input_ids", []))
            or not res.get("restore_verified")
            or not math.isfinite(res.get("restore_ms", -1))):
        raise RuntimeError(f"Sample validation failed: {res}")
    if not measure_handoff:
        if res["n_threads"] != CPU_THREADS or res["n_threads_batch"] != CPU_THREADS:
            raise RuntimeError(f"Thread count mismatch: {res['n_threads']}/{res['n_threads_batch']}")
        if not math.isfinite(res.get("prefill_ms", -1)) or res["prefill_ms"] <= 0:
            raise RuntimeError(f"Invalid prefill_ms: {res}")
    else:
        if res.get("handoff_ms", -1) <= 0 or res.get("handoff_bytes", 0) <= 0:
            raise RuntimeError(f"Invalid handoff: {res}")


def iqr_upper_fence(values: list[float]) -> float:
    if len(values) < 4:
        return float("inf")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    q1 = statistics.median(s[:mid])
    q3 = statistics.median(s[mid + (n % 2):])
    return q3 + 1.5 * (q3 - q1)


def summarize_and_plot(out: Path, prefill_rows: list[dict], handoff_rows: list[dict]) -> dict:
    # Write raw CSVs
    with (out / "stage-times.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prefill_rows[0]))
        w.writeheader()
        w.writerows(prefill_rows)
    if handoff_rows:
        with (out / "handoff-times.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(handoff_rows[0]))
            w.writeheader()
            w.writerows(handoff_rows)

    # Top-token agreement per (rep, P)
    agreements = 0
    total_pairs = 0
    by_rp = {}
    for r in prefill_rows:
        by_rp.setdefault((r["rep"], r["P"]), {})[r["arm"]] = r["top_token"]
    for (_, _), d in by_rp.items():
        if "shared_cpu" in d and "independent_htp" in d:
            total_pairs += 1
            if d["shared_cpu"] == d["independent_htp"]:
                agreements += 1

    # Handoff M(4096) stats (p==0)
    hnd_p0 = [r["handoff_ms"] for r in handoff_rows if r["P"] == 0]
    hnd_fence = iqr_upper_fence(hnd_p0)
    hnd_clean = [x for x in hnd_p0 if x <= hnd_fence]
    m_h_med = statistics.median(hnd_clean) if hnd_clean else 0.0
    m_h_mean = statistics.mean(hnd_clean) if hnd_clean else 0.0

    # Per-P summary with IQR filtering
    summary_rows = []
    outliers_removed = 0
    for p in P_POINTS:
        row = {"H": H_TOKENS, "P": p}
        for arm in ("shared_cpu", "independent_htp"):
            vals = [r["prefill_ms"] for r in prefill_rows if r["arm"] == arm and r["P"] == p]
            fence = iqr_upper_fence(vals)
            clean = [v for v in vals if v <= fence]
            outliers_removed += len(vals) - len(clean)
            s_clean = sorted(clean)
            n = len(s_clean)
            q1 = statistics.median(s_clean[: n // 2]) if n >= 2 else s_clean[0]
            q3 = statistics.median(s_clean[n // 2 + (n % 2):]) if n >= 2 else s_clean[0]
            row[f"{arm}_n_raw"] = len(vals)
            row[f"{arm}_n_clean"] = len(clean)
            row[f"{arm}_raw_median_ms"] = round(statistics.median(vals), 3)
            row[f"{arm}_median_ms"] = round(statistics.median(clean), 3)
            row[f"{arm}_mean_ms"] = round(statistics.mean(clean), 3)
            row[f"{arm}_q1_ms"] = round(q1, 3)
            row[f"{arm}_q3_ms"] = round(q3, 3)
        row["htp_plus_m_median_ms"] = round(row["independent_htp_median_ms"] + m_h_med, 3)
        row["cc_vs_htp_ms"] = round(row["shared_cpu_median_ms"] - row["independent_htp_median_ms"], 3)
        row["cc_vs_htp_plus_m_ms"] = round(row["shared_cpu_median_ms"] - row["htp_plus_m_median_ms"], 3)
        summary_rows.append(row)

    with (out / "summary-table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    summary = {
        "device": "xiaomi15",
        "soc": "SM8750 (Snapdragon 8 Elite, Hexagon v79)",
        "model": "qwen3-8b-q4_0",
        "H": H_TOKENS,
        "P_points": list(P_POINTS),
        "cpu_threads": CPU_THREADS,
        "htp_host_threads": HTP_HOST_THREADS,
        "total_prefill": len(prefill_rows),
        "outliers_removed": outliers_removed,
        "top_token_agreement": f"{agreements}/{total_pairs}",
        "handoff_p0_n_raw": len(hnd_p0),
        "handoff_p0_n_clean": len(hnd_clean),
        "handoff_m_4096_median_ms": round(m_h_med, 3),
        "handoff_m_4096_mean_ms": round(m_h_mean, 3),
        "points": summary_rows,
    }
    save_json(out / "summary.json", summary)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ps = [r["P"] for r in summary_rows]
        cpu_med = [r["shared_cpu_median_ms"] / 1000.0 for r in summary_rows]
        cpu_q1 = [r["shared_cpu_q1_ms"] / 1000.0 for r in summary_rows]
        cpu_q3 = [r["shared_cpu_q3_ms"] / 1000.0 for r in summary_rows]
        htp_med = [r["independent_htp_median_ms"] / 1000.0 for r in summary_rows]
        htp_q1 = [r["independent_htp_q1_ms"] / 1000.0 for r in summary_rows]
        htp_q3 = [r["independent_htp_q3_ms"] / 1000.0 for r in summary_rows]
        htp_m = [r["htp_plus_m_median_ms"] / 1000.0 for r in summary_rows]

        fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=180)
        ax.fill_between(ps, cpu_q1, cpu_q3, color="#1f77b4", alpha=0.18)
        ax.plot(ps, cpu_med, "-o", color="#1f77b4", linewidth=2.0, markersize=4.5, label="Shared-Weight CPU (4T)")
        ax.fill_between(ps, htp_q1, htp_q3, color="#d62728", alpha=0.18)
        ax.plot(ps, htp_med, "-s", color="#d62728", linewidth=2.0, markersize=4.5, label="HTP v79 (HMX, 4T)")
        ax.plot(ps, htp_m, "--^", color="#ff7f0e", linewidth=1.8, markersize=4.0, label=f"HTP + M(4096) ({m_h_med:.0f} ms)")

        ax.set_xlabel("Input Tokens P (H = 4096)")
        ax.set_ylabel("Prefill Latency (s)")
        ax.set_title("Xiaomi 15 (8 Elite, Hexagon v79): Shared CPU vs HTP (Qwen3-8B Q4_0, n=10)")
        ax.set_xticks(ps)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        fig.tight_layout()
        fig.savefig(out / "prefill-2way-h4096-x15.png")
        fig.savefig(out / "prefill-2way-h4096-x15.pdf")
        plt.close(fig)
    except Exception as exc:
        print(f"Plot warning: {exc}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ssh-config", type=Path, default=DEFAULT_SSH_CFG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--cool-s", type=float, default=15.0)
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = read_json(args.model_config.resolve())
    prompt = read_json(args.fixture.resolve())["choices"][0]["prompt_token_ids"]
    if len(prompt) < H_TOKENS + max(P_POINTS):
        raise ValueError(f"Fixture too short: {len(prompt)}")

    phone = X15Phone(args.ssh_config.resolve())
    run_dir = f"{phone.remote_root}/x15-2way-h4096-8b"

    with (out / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            phone.stage_run_dir(run_dir)
            snap_meta_path = out / "snapshots.json"
            if snap_meta_path.exists():
                snap = read_json(snap_meta_path)
            else:
                save_json(out / "status.json", {"phase": "preparing_snapshot", "H": H_TOKENS})
                print("[Phase A] Preparing H=4096 KV snapshot on Xiaomi 15...", flush=True)
                try:
                    phone.start_service(run_dir, model["model_path"], "prepare_cpu", "prepare-runtime.log")
                    snap = phone.post_ab({
                        "prepare_snapshot": True,
                        "snapshot_name": "prefill-ab-h4096.json",
                        "prefix_ids": prompt[:H_TOKENS],
                    })
                    if snap.get("mode") != "cpu_prefill_ab_prepare_v1" or snap.get("H") != H_TOKENS:
                        raise RuntimeError(f"Snapshot prepare failed: {snap}")
                    save_json(snap_meta_path, snap)
                finally:
                    phone.stop_service(run_dir)
                    phone.pull(f"{run_dir}/prepare-runtime.log", out / "prepare-runtime.log")
                print(f"[Phase A] Snapshot ready: checksum={snap['snapshot_checksum'][:12]}...", flush=True)

            total_blocks = args.reps * 2
            prefill_rows = []
            handoff_rows = []
            completed_blocks = 0

            for rep in range(args.reps):
                arm_order = ("shared_cpu", "independent_htp") if (rep % 2 == 0) else ("independent_htp", "shared_cpu")
                for arm in arm_order:
                    block_dir = out / f"rep-{rep}-{arm}"
                    block_dir.mkdir(parents=True, exist_ok=True)
                    p_seq = list(P_POINTS)
                    random.Random(2026112450 + rep * 17 + (1 if arm == "shared_cpu" else 2)).shuffle(p_seq)
                    hnd_seq = list(HANDOFF_P_POINTS) if arm == "shared_cpu" else []
                    random.Random(2026112490 + rep).shuffle(hnd_seq)

                    all_cached = (
                        (block_dir / "warmup.json").exists()
                        and (block_dir / "runtime.log").exists()
                        and all((block_dir / f"h{H_TOKENS}-p{p:03d}.json").exists() for p in p_seq)
                        and all((block_dir / f"handoff-h{H_TOKENS}-p{p:03d}.json").exists() for p in hnd_seq)
                    )

                    if not all_cached:
                        if completed_blocks > 0 and args.cool_s > 0:
                            save_json(out / "status.json", {
                                "phase": "cooling", "rep": rep, "arm": arm,
                                "completed_blocks": completed_blocks, "total_blocks": total_blocks,
                            })
                            time.sleep(args.cool_s)
                        save_json(out / "status.json", {
                            "phase": "running", "rep": rep, "arm": arm,
                            "completed_blocks": completed_blocks, "total_blocks": total_blocks,
                        })
                        try:
                            health = phone.start_service(run_dir, model["model_path"], arm, f"r{rep}-{arm}.log")
                            save_json(block_dir / "health.json", health)

                            w_file = block_dir / "warmup.json"
                            if not w_file.exists():
                                w_req = {
                                    "snapshot_name": snap["snapshot_name"],
                                    "snapshot_checksum": snap["snapshot_checksum"],
                                    "prefix_ids": prompt[:H_TOKENS],
                                    "input_ids": prompt[H_TOKENS:H_TOKENS + 32],
                                }
                                w_res = phone.post_ab(w_req)
                                validate_sample(w_res, w_req, arm)
                                save_json(w_file, {"request": w_req, "response": w_res})

                            for p in p_seq:
                                s_file = block_dir / f"h{H_TOKENS}-p{p:03d}.json"
                                if s_file.exists():
                                    continue
                                req = {
                                    "snapshot_name": snap["snapshot_name"],
                                    "snapshot_checksum": snap["snapshot_checksum"],
                                    "prefix_ids": prompt[:H_TOKENS],
                                    "input_ids": prompt[H_TOKENS:H_TOKENS + p],
                                }
                                res = phone.post_ab(req)
                                validate_sample(res, req, arm)
                                save_json(s_file, {"request": req, "response": res})

                            for p in hnd_seq:
                                h_file = block_dir / f"handoff-h{H_TOKENS}-p{p:03d}.json"
                                if h_file.exists():
                                    continue
                                req = {
                                    "snapshot_name": snap["snapshot_name"],
                                    "snapshot_checksum": snap["snapshot_checksum"],
                                    "prefix_ids": prompt[:H_TOKENS],
                                    "input_ids": [] if p == 0 else prompt[H_TOKENS:H_TOKENS + p],
                                    "measure_handoff": True,
                                    "handoff_only_h": (p == 0),
                                    "verify_handoff_bytes": False,
                                }
                                res = phone.post_ab(req)
                                validate_sample(res, req, arm, measure_handoff=True)
                                save_json(h_file, {"request": req, "response": res})
                        finally:
                            phone.stop_service(run_dir)
                            phone.pull(f"{run_dir}/r{rep}-{arm}.log", block_dir / "runtime.log")
                        verify_runtime_log(block_dir / "runtime.log", arm)

                    # Collect rows from block_dir
                    p_samples = {}
                    for p in P_POINTS:
                        rec = read_json(block_dir / f"h{H_TOKENS}-p{p:03d}.json")["response"]
                        p_samples[p] = rec["prefill_ms"]
                        prefill_rows.append({
                            "rep": rep, "arm": arm, "threads": CPU_THREADS,
                            "H": H_TOKENS, "P": p,
                            "restore_ms": rec["restore_ms"],
                            "prefill_ms": rec["prefill_ms"],
                            "top_token": rec["top_token"],
                            "swap_kib": rec["resources_after"]["VmSwap_kib"],
                            "rss_kib": rec["resources_after"]["VmRSS_kib"],
                        })
                    for p in hnd_seq:
                        rec = read_json(block_dir / f"handoff-h{H_TOKENS}-p{p:03d}.json")["response"]
                        handoff_rows.append({
                            "rep": rep, "H": H_TOKENS, "P": p, "total_tokens": H_TOKENS + p,
                            "restore_ms": rec["restore_ms"],
                            "htp_prefill_ms": rec["prefill_ms"],
                            "handoff_ms": rec["handoff_ms"],
                            "handoff_bytes": rec["handoff_bytes"],
                            "top_token": rec["top_token"],
                            "swap_kib": rec["resources_after"]["VmSwap_kib"],
                        })
                    completed_blocks += 1
                    print(
                        f"[{completed_blocks}/{total_blocks}] rep={rep} arm={arm} | "
                        f"P=16:{p_samples[16]:.0f}ms P=24:{p_samples[24]:.0f}ms "
                        f"P=32:{p_samples[32]:.0f}ms P=40:{p_samples[40]:.0f}ms P=48:{p_samples[48]:.0f}ms",
                        flush=True,
                    )

            summary = summarize_and_plot(out, prefill_rows, handoff_rows)
            save_json(out / "status.json", {"phase": "complete", "completed_blocks": completed_blocks, "total_blocks": total_blocks})
            print(f"[DONE] top_token={summary['top_token_agreement']} M(4096)_med={summary['handoff_m_4096_median_ms']}ms", flush=True)
        except Exception as exc:
            save_json(out / "status.json", {"phase": "failed", "error": str(exc)})
            raise
        finally:
            phone.ssh("termux-wake-unlock || true", check=False)


if __name__ == "__main__":
    main()
