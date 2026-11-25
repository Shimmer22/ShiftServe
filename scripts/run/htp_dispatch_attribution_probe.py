#!/usr/bin/env python3
"""Controlled 6-arm factorial probe with GGML_HEXAGON_PROFILE=1 to attribute
HTP Prefill dispatch_us gaps between Full-HTP and Shared-Weights HC on Xiaomi 14 Pro."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from a2o.runtime.device_transport import DeviceTransport
from a2o.validation.oracle import active_model_processes
from scripts.run.frozen_ablation import BASE, Campaign, _agent_chat_resilient, read, request, save

CASE = "find_current_city_low_battery_mode"
DEFAULT_BUILD = ROOT / "results/mainline-eval/htp-hc-decode-probe-build"
DEFAULT_FIXTURES = ROOT / "results/20261124-toolsandbox-10cases/fixtures.json"
DEFAULT_OUTPUT = ROOT / "results/mainline-eval/20261125-htp-dispatch-attribution"

COLD_MODEL_HTP = {
    "kind": "cold_route_v1",
    "model_id": "qwen3-8b-q4_0",
    "shared_weights": True,
    "points": [
        {"P": 1, "cc_ms": 100000.0, "hc_ms": 1000.0},
        {"P": 9472, "cc_ms": 100000.0, "hc_ms": 1000.0},
    ],
}

ARMS = [
    {
        "name": "arm1_full_htp_before",
        "condition": "htp",
        "shared_weights": False,
        "cpu_threads": 6,
        "use_cold_htp": False,
        "turn_modes": ["full_htp", "full_htp", "full_htp"],
        "expected_routes": ["htp", "htp", "htp"],
        "expected_decode_backends": ["htp", "htp", "htp"],
    },
    {
        "name": "arm2_shared_cold_htp_prev_htp_dec",
        "condition": "route",
        "shared_weights": True,
        "cpu_threads": 6,
        "use_cold_htp": True,
        "turn_modes": ["cold_htp", "retain_htp_decode", "retain"],
        "expected_routes": ["htp", "htp", "htp"],
        "expected_decode_backends": ["cpu", "htp", "cpu"],
    },
    {
        "name": "arm3_shared_cold_htp_cpu_dec_6t",
        "condition": "route",
        "shared_weights": True,
        "cpu_threads": 6,
        "use_cold_htp": True,
        "turn_modes": ["cold_htp", "rebuild_async", "rebuild_async"],
        "expected_routes": ["htp", "htp", "htp"],
        "expected_decode_backends": ["cpu", "cpu", "cpu"],
    },
    {
        "name": "arm4_shared_cold_htp_cpu_dec_4t",
        "condition": "route",
        "shared_weights": True,
        "cpu_threads": 4,
        "use_cold_htp": True,
        "turn_modes": ["cold_htp", "rebuild_async", "rebuild_async"],
        "expected_routes": ["htp", "htp", "htp"],
        "expected_decode_backends": ["cpu", "cpu", "cpu"],
    },
    {
        "name": "arm5_shared_real_t6_6t",
        "condition": "route",
        "shared_weights": True,
        "cpu_threads": 6,
        "use_cold_htp": False,
        "turn_modes": ["real_cpu", "rebuild_async", "rebuild_async"],
        "expected_routes": ["cpu", "htp", "htp"],
        "expected_decode_backends": ["cpu", "cpu", "cpu"],
    },
    {
        "name": "arm6_full_htp_after",
        "condition": "htp",
        "shared_weights": False,
        "cpu_threads": 6,
        "use_cold_htp": False,
        "turn_modes": ["full_htp", "full_htp", "full_htp"],
        "expected_routes": ["htp", "htp", "htp"],
        "expected_decode_backends": ["htp", "htp", "htp"],
    },
]

BATCH_RE = re.compile(r"ggml-hex: HTP0 profile-batch n-ops (\d+) batch-dur-usec (\d+) htp-ops-usec (\d+)")
OP_RE = re.compile(r"ggml-hex: HTP0 profile-op (\S+): .* : usec (\d+) cycles (\d+)")


class ProfiledCampaign(Campaign):
    """Campaign subclass that launches route_phone.py with --htp-profile 1."""

    def service_arm(self, action: str, arm_cfg: dict, directory: Path) -> None:
        condition = arm_cfg["condition"]
        shared = arm_cfg["shared_weights"]
        cpu_threads = arm_cfg["cpu_threads"]
        command = [
            sys.executable,
            str(ROOT / "scripts/run/route_phone.py"),
            action,
            "--condition",
            condition,
            "--output",
            str(directory),
            "--logs-only",
            "--retry-network",
        ]
        if action == "start":
            m = self.model
            command += [
                "--runtime-overlay",
                str(self.args.build),
                "--model",
                m["model_path"],
                "--model-id",
                m["model_id"],
                "--model-sha256",
                m["model_sha256"],
                "--context-size",
                str(self.context),
                "--cpu-repack",
                "0",
                "--kv-hwm-trim",
                str(self.args.kv_hwm_trim),
                "--device",
                self.args.device,
                "--gap-ms",
                str(self.args.gap_ms),
                "--htp-profile",
                "1",
                "--verify-kv-state",
                str(self.args.verify_kv_state),
                "--cpu-threads",
                str(cpu_threads),
                "--max-quiet-c",
                str(self.args.max_quiet_c),
                "--cooldown-timeout-s",
                str(self.args.cooldown_timeout_s),
                "--min-mem-avail-kib",
                "4000000",
                "--memory-ledger",
            ]
            if shared:
                command += ["--shared-weights"]
        subprocess.run(command, cwd=ROOT, check=True)
        if action == "start":
            health = read(directory / "health.json")
            if (
                health["condition"] != condition
                or health["shared_weights"] != shared
                or health["kv_reduce"] is not False
                or health["context_size"] != self.context
                or health["frozen_decode_version"] != 1
                or health.get("build_id") != self.build_id
                or health.get("cpu_threads") != cpu_threads
                or read(directory / "manifest.json")["runtime_overlay"] != self.build
                or read(directory / "manifest.json").get("htp_profile") != 1
            ):
                raise RuntimeError(f"mismatched phone condition {arm_cfg['name']}")


def check_phone_clean(transport: DeviceTransport, output: Path) -> None:
    processes = transport.call("shell", "/system/bin/ps -A -o PID,ARGS").stdout.decode()
    (output / "phone-processes-before.txt").write_text(processes)
    if active_model_processes(processes) or any(
        "/runs/" in line and "/service " in line for line in processes.splitlines()
    ):
        raise RuntimeError("phone still has a model process running")


def run_arm(
    campaign: ProfiledCampaign,
    arm_cfg: dict,
    case: dict,
    output: Path,
) -> list[dict]:
    arm_name = arm_cfg["name"]
    arm_dir = output / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    svc_dir = arm_dir / f"service-{time.time_ns()}"

    turns_out = []
    campaign.service_arm("start", arm_cfg, svc_dir)
    try:
        save(arm_dir / "reset.json", request("/a2o/reset", {}, timeout=30, retries=1))
        if arm_cfg["use_cold_htp"]:
            cost_resp = request("/a2o/costs", {"cold_model": COLD_MODEL_HTP}, timeout=30, retries=1)
            save(arm_dir / "cold-costs.json", cost_resp)
            if not cost_resp.get("cold_cost_loaded"):
                raise RuntimeError(f"{arm_name}: cold_cost_loaded is False: {cost_resp}")

        for i, turn in enumerate(case["turns"][:3]):
            mode = arm_cfg["turn_modes"][i]
            body = dict(turn["request"], max_tokens=turn["D"], frozen_graph_profile=True)
            prep = None
            if i > 0 and arm_cfg["condition"] == "route":
                prep = request("/a2o/tool-start", {}, timeout=20, retries=1)
                if not prep.get("started"):
                    raise RuntimeError(f"{arm_name} turn {i}: HTP preparation failed to start: {prep}")
                body.update(frozen_force_path="hc_full", frozen_hc_lifecycle_probe=mode)

            response = _agent_chat_resilient(BASE, body, f"{arm_name}:turn-{i}", total_timeout=180)
            save(arm_dir / f"turn-{i:02}.json", {"prepare": prep, "response": response})

            t = response["timings"]
            if (t["H_anchor"], t["H_effective"], t["P_evaluated"], t["predicted_n"]) != (
                turn["H"],
                turn["H"],
                turn["P"],
                turn["D"],
            ):
                raise RuntimeError(f"{arm_name} turn {i}: frozen shape mismatch")
            if response["__verbose"]["tokens"] != turn["request"]["frozen_decode_ids"]:
                raise RuntimeError(f"{arm_name} turn {i}: frozen decode mismatch")
            if t["route"] != arm_cfg["expected_routes"][i]:
                raise RuntimeError(
                    f"{arm_name} turn {i}: expected route={arm_cfg['expected_routes'][i]}, got {t['route']} ({t.get('route_reason')})"
                )
            if t["decode_backend"] != arm_cfg["expected_decode_backends"][i]:
                raise RuntimeError(
                    f"{arm_name} turn {i}: expected decode_backend={arm_cfg['expected_decode_backends'][i]}, got {t['decode_backend']}"
                )
            if t["kv_reduce"] is not None or t["graph_profile"]["prefill"]["callback_graphs"] != 0:
                raise RuntimeError(f"{arm_name} turn {i}: unexpected compaction or callback")

            gp = t["graph_profile"]["prefill"]
            print(
                f"[{arm_name}] turn-{i:02} route={t['route']} dec={t['decode_backend']} "
                f"prompt={t['prompt_ms']:.1f}ms disp={gp['dispatch_us']/1000:.1f}ms",
                flush=True,
            )
            turns_out.append({"turn": i, "timings": t})
    finally:
        if svc_dir.exists():
            campaign.service_arm("stop", arm_cfg, svc_dir)

    return turns_out


def _extract_clock_summary(clocks_obj: dict) -> dict:
    if not clocks_obj:
        return {}
    tz_map = {v["type"]: v["value"] / 1000.0 for v in (clocks_obj.get("thermal_sysfs_raw") or {}).values()}
    pol = clocks_obj.get("cpu_policy_khz") or {}
    return {
        "quiet_therm_c": tz_map.get("quiet_therm"),
        "battery_c": tz_map.get("battery"),
        "nsphvx_0_c": tz_map.get("nsphvx-0"),
        "nsphmx_0_c": tz_map.get("nsphmx-0"),
        "cpu_therm_c": tz_map.get("cpu_therm"),
        "ddr_c": tz_map.get("ddr"),
        "policy0_khz": pol.get("policy0"),
        "policy2_khz": pol.get("policy2"),
        "policy5_khz": pol.get("policy5"),
        "policy7_khz": pol.get("policy7"),
    }


def _extract_memory_summary(timings: dict) -> dict:
    samples = (timings.get("memory_ledger") or {}).get("samples") or []
    by_stage = {s.get("stage"): s for s in samples if "stage" in s}
    bp = by_stage.get("before_prefill")
    ap = by_stage.get("after_prefill")
    if not bp or not ap:
        return {}
    return {
        "before_prefill_vmswap_kib": bp["process"].get("VmSwap_kib"),
        "before_prefill_pss_dma_kib": bp.get("pss_plus_identified_dma_kib"),
        "before_prefill_mem_avail_kib": bp["meminfo"].get("MemAvailable_kib"),
        "delta_minflt": ap["process"]["minor_faults"] - bp["process"]["minor_faults"],
        "delta_majflt": ap["process"]["major_faults"] - bp["process"]["major_faults"],
    }


def parse_runtime_log_prefill_windows(log_path: Path) -> list[dict]:
    """Segment runtime.log into [before_prefill, after_prefill] windows and parse DSP profile lines."""
    windows = []
    in_window = False
    cur = None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("A2O_MEMORY "):
            if '"stage":"before_prefill"' in line:
                in_window = True
                cur = {
                    "n_batches": 0,
                    "batch_ops_total": 0,
                    "batch_dur_usec": 0,
                    "htp_ops_usec_from_batch": 0,
                    "n_htp_ops": 0,
                    "op_usec_sum": 0,
                    "op_cycles_sum": 0,
                    "by_op": defaultdict(lambda: {"count": 0, "usec": 0, "cycles": 0}),
                }
            elif '"stage":"after_prefill"' in line and in_window:
                windows.append(cur)
                in_window = False
                cur = None
            continue
        if not in_window or cur is None:
            continue
        mb = BATCH_RE.search(line)
        if mb:
            cur["n_batches"] += 1
            cur["batch_ops_total"] += int(mb.group(1))
            cur["batch_dur_usec"] += int(mb.group(2))
            cur["htp_ops_usec_from_batch"] += int(mb.group(3))
            continue
        mo = OP_RE.search(line)
        if mo:
            op_name = mo.group(1)
            usec = int(mo.group(2))
            cycles = int(mo.group(3))
            cur["n_htp_ops"] += 1
            cur["op_usec_sum"] += usec
            cur["op_cycles_sum"] += cycles
            cur["by_op"][op_name]["count"] += 1
            cur["by_op"][op_name]["usec"] += usec
            cur["by_op"][op_name]["cycles"] += cycles
    return windows


def summarize_arm(arm_cfg: dict, output: Path) -> dict:
    arm_name = arm_cfg["name"]
    arm_dir = output / arm_name
    svc_dirs = sorted(arm_dir.glob("service-*"))
    if not svc_dirs:
        raise RuntimeError(f"missing service dir in {arm_dir}")
    log_path = svc_dirs[-1] / "device/runtime.log"
    if not log_path.exists():
        raise RuntimeError(f"missing runtime.log: {log_path}")

    windows = parse_runtime_log_prefill_windows(log_path)
    if len(windows) != 3:
        raise RuntimeError(f"{arm_name}: expected 3 prefill windows in {log_path}, found {len(windows)}")

    turns_summary = []
    for i in range(3):
        tjson = json.loads((arm_dir / f"turn-{i:02}.json").read_text())
        tm = tjson["response"]["timings"]
        gp = tm["graph_profile"]["prefill"]
        w = windows[i]

        disp_us = gp["dispatch_us"]
        batch_dur_us = w["batch_dur_usec"]
        htp_ops_us = w["htp_ops_usec_from_batch"]
        cycles = w["op_cycles_sum"]

        eff_mhz = (cycles / htp_ops_us) if htp_ops_us > 0 else 0.0
        rpc_queue_ms = (batch_dur_us - htp_ops_us) / 1000.0
        cpu_split_ms = (disp_us - batch_dur_us) / 1000.0

        by_op_out = {}
        for op_name, st in sorted(w["by_op"].items(), key=lambda kv: -kv[1]["usec"]):
            by_op_out[op_name] = {
                "count": st["count"],
                "usec": st["usec"],
                "ms": st["usec"] / 1000.0,
                "cycles_M": st["cycles"] / 1e6,
                "eff_mhz": (st["cycles"] / st["usec"]) if st["usec"] > 0 else 0.0,
            }

        clocks = tm["graph_profile"].get("clocks") or {}
        hc_lp = tm.get("hc_lifecycle_probe") or {}
        turns_summary.append({
            "turn": i,
            "route": tm["route"],
            "decode_backend": tm["decode_backend"],
            "H": tm["H_anchor"],
            "P": tm["P_evaluated"],
            "D": tm["predicted_n"],
            "prompt_ms": tm["prompt_ms"],
            "dispatch_ms": disp_us / 1000.0,
            "build_ms": gp["build_us"] / 1000.0,
            "sync_ms": gp["sync_us"] / 1000.0,
            "graphs": gp["graphs"],
            "prefill_chunks": hc_lp.get("prefill_chunks"),
            "n_batches": w["n_batches"],
            "n_htp_ops": w["n_htp_ops"],
            "htp_ops_ms": htp_ops_us / 1000.0,
            "htp_cycles_M": cycles / 1e6,
            "htp_eff_freq_mhz": eff_mhz,
            "rpc_queue_overhead_ms": rpc_queue_ms,
            "cpu_split_and_sched_ms": cpu_split_ms,
            "predicted_ms": tm["predicted_ms"],
            "by_op": by_op_out,
            "clocks_before": _extract_clock_summary(clocks.get("before_prefill") or {}),
            "clocks_after": _extract_clock_summary(clocks.get("after_prefill") or {}),
            "memory": _extract_memory_summary(tm),
        })

    return {
        "arm": arm_name,
        "condition": arm_cfg["condition"],
        "shared_weights": arm_cfg["shared_weights"],
        "cpu_threads": arm_cfg["cpu_threads"],
        "turns": turns_summary,
    }


def print_summary_tables(arms_summary: list[dict]) -> None:
    for turn_idx in (0, 1, 2):
        print(f"\n### Turn {turn_idx} Attribution (P={arms_summary[0]['turns'][turn_idx]['P']}, H={arms_summary[0]['turns'][turn_idx]['H']})")
        print(
            "| Arm | Route | Dispatch (ms) | HTP Ops (ms) | HTP Cycles (M) | HTP Eff Freq (MHz) | RPC/Queue (ms) | CPU Split+Sched (ms) | Batches | Ops | quiet_c / nsp_c |"
        )
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for arm in arms_summary:
            t = arm["turns"][turn_idx]
            cb = t["clocks_before"]
            qc = f"{cb.get('quiet_therm_c', 0):.1f}" if cb.get("quiet_therm_c") is not None else "-"
            nc = f"{cb.get('nsphvx_0_c', 0):.1f}" if cb.get("nsphvx_0_c") is not None else "-"
            print(
                f"| {arm['arm']} | {t['route']} | {t['dispatch_ms']:.1f} | {t['htp_ops_ms']:.1f} | "
                f"{t['htp_cycles_M']:.1f} | {t['htp_eff_freq_mhz']:.1f} | {t['rpc_queue_overhead_ms']:.1f} | "
                f"{t['cpu_split_and_sched_ms']:.1f} | {t['n_batches']} | {t['n_htp_ops']} | {qc} / {nc} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-quiet-c", type=float, default=36.0)
    parser.add_argument("--cooldown-timeout-s", type=int, default=90)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        corpus = read(args.fixtures)
        case = next(c for c in corpus["cases"] if c["id"] == CASE)
        transport = DeviceTransport(ROOT, "ssh", config=ROOT / "configs/termux-ssh.json")
        settings = SimpleNamespace(
            build=args.build.resolve(),
            output=out,
            fixtures=args.fixtures.resolve(),
            case=[CASE],
            model=ROOT / "configs/model-default.json",
            route_costs=None,
            kv_costs=None,
            conditions=["htp", "route"],
            cpu_registry=ROOT / "results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so",
            context_size=9472,
            gap_ms=0,
            verify_kv_state=0,
            kv_hwm_trim=0,
            device="xiaomi14pro",
            memory_ledger=True,
            compact_anchor=0,
            kv_layer_stride=4,
            kv_min_evict_tokens=256,
            force_path="none",
            single_turn=False,
            m1_explorer=False,
            cpu_threads=6,
            max_quiet_c=args.max_quiet_c,
            cooldown_timeout_s=args.cooldown_timeout_s,
        )
        campaign = ProfiledCampaign(settings, corpus)
        save(out / "plan.json", {"case": CASE, "build": str(args.build.resolve()), "arms": ARMS, "htp_profile": 1})

        with (ROOT / "results/.termux-model.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_phone_clean(transport, out)
            for idx, arm_cfg in enumerate(ARMS):
                save(out / "status.json", {"status": "running", "arm": arm_cfg["name"], "index": idx})
                run_arm(campaign, arm_cfg, case, out)
                if idx == 0:
                    arm1_summary = summarize_arm(arm_cfg, out)
                    if any(arm1_summary["turns"][ti]["n_htp_ops"] == 0 for ti in (0, 1, 2)):
                        raise RuntimeError(
                            f"Arm 1 produced 0 HTP profile-op lines in runtime.log; aborting before running remaining arms: {arm1_summary}"
                        )
                    print(
                        f"Arm 1 DSP profile check OK: turn-01 n_htp_ops={arm1_summary['turns'][1]['n_htp_ops']}, "
                        f"htp_eff_freq_mhz={arm1_summary['turns'][1]['htp_eff_freq_mhz']:.1f}",
                        flush=True,
                    )

    arms_summary = [summarize_arm(arm_cfg, out) for arm_cfg in ARMS]
    save(out / "summary.json", {"case": CASE, "arms": arms_summary})
    save(out / "status.json", {"status": "passed"})
    print_summary_tables(arms_summary)


if __name__ == "__main__":
    main()
