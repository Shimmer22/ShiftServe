#!/usr/bin/env python3
"""Analyze pure-action (H, P, D) distributions for DroidTask & AndroidControl-Low
and export phone-ready frozen replay fixtures (Stage 1)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

EMPTY_THINK = [151667, 271, 151668, 271]
USER_HEADER = [151644, 872, 198]
PHONE_N_CTX = 9472
PHONE_RECENT = 32
PHONE_MIN_EVICT = 256

SELECTED_DROIDTASK = [
    "firefox_b6056f6eeb89",
    "calendar_e8b601c47a04",
    "calendar_77b4402acf90",
]
SELECTED_AC_LOW = [
    "ac_ep_8982",
    "ac_ep_6119",
    "ac_ep_13841",
    "ac_ep_2405",
    "ac_ep_4451",
]


def quantile(xs: list[int | float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    idx = (len(s) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def simulate_phone_prior(case: dict, ratio: float | None = None, fixed_keep: int | None = None) -> tuple[int, bool]:
    """Simulate phone admission (n_ctx=9472, recent=32, min_evict=256) and count prefill_prior turns."""
    pin = case["pin"]
    he = 0
    prior_hits = 0
    fits = True
    for idx, t in enumerate(case["turns"]):
        active = he + t["P"]
        if active + t["D"] > PHONE_N_CTX:
            fits = False
            break
        sink = max(4, pin)
        rec = min(max(0, active - sink), PHONE_RECENT)
        if fixed_keep is not None:
            if idx == 0:
                he = active
                continue
            keep = fixed_keep
            prior_budget = keep - (sink + rec)
            evictable = max(0, active - max(keep, sink + rec))
            will_reduce = evictable > 0 and active > keep
        else:
            assert ratio is not None
            mid = max(0, active - sink - rec)
            keep = sink + rec + round(ratio * mid)
            prior_budget = max(0, keep - (sink + rec))
            evictable = max(0, active - max(keep, sink + rec))
            will_reduce = evictable > 0 and active > (sink + PHONE_RECENT + 64)
        if will_reduce:
            if prior_budget > 0 and evictable >= PHONE_MIN_EVICT:
                prior_hits += 1
            he = max(keep, sink + rec)
        else:
            he = active
    return prior_hits, fits


def write_distribution_artifacts(hpd_rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_csv = out_dir / "turn_hpd_distribution.csv"
    ep_csv = out_dir / "episode_hpd_summary.csv"

    with turn_csv.open("w", newline="") as tf, ep_csv.open("w", newline="") as ef:
        tw = csv.writer(tf)
        ew = csv.writer(ef)
        tw.writerow([
            "benchmark", "case_id", "in_top50", "in_clean4_pool", "turn_index",
            "pin_tokens", "H", "P", "D", "prompt_tokens", "logical_tokens",
        ])
        ew.writerow([
            "benchmark", "case_id", "in_top50", "in_clean4_pool", "turns",
            "pin_tokens", "max_logical_tokens", "P_min", "P_p50", "P_max",
            "P_le_32_turns", "P_le_64_turns", "D_max",
            "prior_hits_r50_9472", "prior_hits_r75_9472", "fits_9472_r50", "fits_9472_r75",
        ])
        for c in hpd_rows:
            ps = [t["P"] for t in c["turns"]]
            ds = [t["D"] for t in c["turns"]]
            max_log = max(t["logical"] for t in c["turns"])
            p50_hits, fit50 = simulate_phone_prior(c, ratio=0.5)
            p75_hits, fit75 = simulate_phone_prior(c, ratio=0.75)
            ew.writerow([
                c["benchmark"], c["id"], int(bool(c.get("in_top50"))), int(bool(c.get("in_clean4_pool", True))),
                len(ps), c["pin"], max_log, min(ps), round(quantile(ps, 0.5), 1), max(ps),
                sum(1 for x in ps if x <= 32), sum(1 for x in ps if x <= 64), max(ds),
                p50_hits, p75_hits, int(fit50), int(fit75),
            ])
            for t in c["turns"]:
                tw.writerow([
                    c["benchmark"], c["id"], int(bool(c.get("in_top50"))), int(bool(c.get("in_clean4_pool", True))),
                    t["turn"], c["pin"], t["H"], t["P"], t["D"], t["prompt_tokens"], t["logical"],
                ])

    def summarize(name: str, subset: list[dict]) -> dict:
        n_turns = [len(c["turns"]) for c in subset]
        max_logs = [max(t["logical"] for t in c["turns"]) for c in subset]
        all_p = [t["P"] for c in subset for t in c["turns"]]
        warm_p = [t["P"] for c in subset for t in c["turns"][1:]]
        all_d = [t["D"] for c in subset for t in c["turns"]]
        s50 = [simulate_phone_prior(c, ratio=0.5) for c in subset]
        s75 = [simulate_phone_prior(c, ratio=0.75) for c in subset]
        return {
            "cohort": name,
            "tasks": len(subset),
            "total_turns": len(all_p),
            "warm_turns": len(warm_p),
            "turns_per_task": {"min": min(n_turns), "p50": quantile(n_turns, 0.5), "max": max(n_turns)},
            "max_logical_tokens": {
                "min": min(max_logs), "p25": quantile(max_logs, 0.25),
                "p50": quantile(max_logs, 0.5), "p75": quantile(max_logs, 0.75), "max": max(max_logs),
            },
            "P_all_turns": {
                "min": min(all_p), "p25": quantile(all_p, 0.25), "p50": quantile(all_p, 0.5),
                "p75": quantile(all_p, 0.75), "max": max(all_p),
                "le_32_count": sum(1 for x in all_p if x <= 32),
                "le_32_pct": sum(1 for x in all_p if x <= 32) / len(all_p) * 100.0,
                "le_64_count": sum(1 for x in all_p if x <= 64),
                "le_64_pct": sum(1 for x in all_p if x <= 64) / len(all_p) * 100.0,
            },
            "P_warm_turns": {
                "min": min(warm_p), "p25": quantile(warm_p, 0.25), "p50": quantile(warm_p, 0.5),
                "p75": quantile(warm_p, 0.75), "max": max(warm_p),
                "le_32_count": sum(1 for x in warm_p if x <= 32),
                "le_32_pct": sum(1 for x in warm_p if x <= 32) / len(warm_p) * 100.0,
                "le_64_count": sum(1 for x in warm_p if x <= 64),
                "le_64_pct": sum(1 for x in warm_p if x <= 64) / len(warm_p) * 100.0,
            },
            "D_pure_action": {"min": min(all_d), "p50": quantile(all_d, 0.5), "p75": quantile(all_d, 0.75), "max": max(all_d)},
            "phone_9472_prefill_prior_est": {
                "ratio_0.50": {
                    "avg_triggers_per_task": sum(x[0] for x in s50) / len(s50),
                    "tasks_ge_2_triggers": sum(1 for x in s50 if x[0] >= 2),
                    "tasks_fit_9472": sum(1 for x in s50 if x[1]),
                },
                "ratio_0.75": {
                    "avg_triggers_per_task": sum(x[0] for x in s75) / len(s75),
                    "tasks_ge_2_triggers": sum(1 for x in s75 if x[0] >= 2),
                    "tasks_fit_9472": sum(1 for x in s75 if x[1]),
                },
            },
        }

    cohorts = [
        summarize("droidtask_all_158", [r for r in hpd_rows if r["benchmark"] == "droidtask"]),
        summarize("droidtask_top50", [r for r in hpd_rows if r["benchmark"] == "droidtask" and r.get("in_top50")]),
        summarize("ac_low_shard0_valid_62", [r for r in hpd_rows if r["benchmark"] == "androidcontrol_low"]),
        summarize("ac_low_clean4_pool_52", [r for r in hpd_rows if r["benchmark"] == "androidcontrol_low" and r.get("in_clean4_pool")]),
        summarize("ac_low_top50", [r for r in hpd_rows if r["benchmark"] == "androidcontrol_low" and r.get("in_top50")]),
    ]
    summary = {"cohorts": cohorts}
    (out_dir / "distribution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_frozen_fixtures_on_agent(agent_base: str, out_dir: Path) -> dict:
    import host_droidtask_worker as dt_w
    import host_androidcontrol_worker as ac_w

    def post_json(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            agent_base.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def freeze_case(benchmark: str, cid: str, source: str, sys_msg: str, users: list[str], assts: list[str]) -> dict:
        msgs = [{"role": "system", "content": sys_msg}]
        prev: list[int] = []
        turns = []
        for idx, (uc, ac) in enumerate(zip(users, assts)):
            assert "<think>" not in ac and "</think>" not in ac, f"{cid} turn {idx} contains <think>"
            msgs.append({"role": "user", "content": uc})
            req_msgs = [dict(m) for m in msgs]
            rendered = post_json("/apply-template", {"messages": req_msgs})["prompt"]
            prompt_ids = post_json("/tokenize", {"content": rendered})["tokens"]
            decode_ids = post_json("/tokenize", {"content": ac})["tokens"]
            assert prompt_ids[-4:] == EMPTY_THINK, f"{cid} turn {idx} missing EMPTY_THINK"
            assert prompt_ids[:len(prev)] == prev and len(prompt_ids) - 4 > len(prev), f"{cid} turn {idx} broken anchor"
            assert decode_ids and min(decode_ids) >= 0 and max(decode_ids) < 151936
            assert 151667 not in decode_ids and 151668 not in decode_ids, f"{cid} turn {idx} think token in decode"
            turns.append({
                "index": idx,
                "H": len(prev),
                "P": len(prompt_ids) - len(prev),
                "D": len(decode_ids),
                "observed_gap_ms": None,
                "request": {
                    "messages": req_msgs,
                    "frozen_prompt_ids": prompt_ids,
                    "frozen_decode_ids": decode_ids,
                },
            })
            prev = prompt_ids[:-4]
            msgs.append({"role": "assistant", "content": ac})
        first = turns[0]["request"]["frozen_prompt_ids"]
        pin = next((i for i in range(len(first) - 2) if first[i:i + 3] == USER_HEADER), 0)
        keep = max(pin + 32 + 24, round(0.75 * (turns[1]["H"] + turns[1]["P"])))
        max_len = max(len(t["request"]["frozen_prompt_ids"]) + t["D"] for t in turns)
        assert max_len <= 8000, f"{cid} max_len={max_len} > 8000"
        return {
            "benchmark": benchmark,
            "id": cid,
            "source": source,
            "source_model": "qwen3-8b-q4_0-pure-action",
            "pinned_estimate": pin,
            "keep_tokens": keep,
            "turns": turns,
        }

    cases = []
    dt_tasks = {t["id"]: t for t in dt_w._load_tasks(ROOT / "scenarios/droidtask/profile_no_answer.json", 0)}
    for tid in SELECTED_DROIDTASK:
        t = dt_tasks[tid]
        sys_msg = f"{dt_w._SYSTEM_PROMPT}\n\nApp: {t['app']}\nTask: {t['task']}"
        users = [f"Step {i} UI state:\n{s['state']}" for i, s in enumerate(t["steps"])]
        assts = [dt_w._gold_action_json(s["label"]) for s in t["steps"]]
        cases.append(freeze_case("droidtask", tid, "scenarios/droidtask/profile_no_answer.json", sys_msg, users, assts))

    ac_cfg = json.loads((ROOT / "configs/host-kv-accuracy-androidcontrol-low-8b.json").read_text())
    ac_eps = {e["id"]: e for e in ac_w.load_episodes(Path(ac_cfg["dataset_path"]), Path(ac_cfg["splits_path"]), 4)}
    for eid in SELECTED_AC_LOW:
        ep = ac_eps[eid]
        sys_msg = f"{ac_w._SYSTEM_PROMPT}\n\nGoal: {ep['goal']}"
        users = [
            f"Step {i} instruction: {s['instruction']}\nStep {i} UI tree:\n{s['rendered']}" if s.get("instruction")
            else f"Step {i} UI tree:\n{s['rendered']}"
            for i, s in enumerate(ep["steps"])
        ]
        assts = [ac_w.gold_action_json(s) for s in ep["steps"]]
        cases.append(freeze_case("androidcontrol_low", eid, ac_cfg["dataset_path"], sys_msg, users, assts))

    # Replay audit on live agent in both full_kv and fixed-keep prefill_prior modes
    audit = {}
    ctx = math.ceil(max(len(t["request"]["frozen_prompt_ids"]) + t["D"] for c in cases for t in c["turns"]) / 256) * 256
    for c in cases:
        for t in c["turns"]:
            rendered = post_json("/apply-template", {"messages": t["request"]["messages"]})["prompt"]
            ids = post_json("/tokenize", {"content": rendered})["tokens"]
            if ids != t["request"]["frozen_prompt_ids"]:
                raise RuntimeError(f"renderer mismatch on {c['id']} turn {t['index']}")
        post_json("/a2o/reset", {})
        evicted = 0
        prior_triggers = 0
        for i, t in enumerate(c["turns"]):
            body = dict(t["request"], max_tokens=t["D"], expected_decode_tokens=t["D"])
            body["frozen_kv_keep_tokens"] = c["keep_tokens"] if i else ctx
            resp = post_json("/v1/chat/completions", body)
            tm = resp["timings"]
            if (resp["__verbose"]["tokens"] != t["request"]["frozen_decode_ids"]
                    or resp["__verbose"]["stop_type"] != "forced"
                    or (tm["H_anchor"], tm["P_evaluated"], tm["predicted_n"]) != (t["H"], t["P"], t["D"])):
                raise RuntimeError(f"frozen shape mismatch on {c['id']} turn {i}")
            if tm["H_effective"] != t["H"] - evicted:
                raise RuntimeError(f"effective history mismatch on {c['id']} turn {i}")
            kv = tm.get("kv_reduce") or {}
            if kv.get("policy") == "prefill_prior" and kv.get("prior_ranked_tokens", 0) > 0:
                prior_triggers += 1
            if kv.get("compact_anchor") and i + 1 < len(c["turns"]):
                evicted = c["turns"][i + 1]["H"] - kv["anchor_live_tokens"]
        audit[c["id"]] = {"renderer_ok": True, "frozen_replay_ok": True, "prior_triggers_fixed_keep": prior_triggers}

    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_payload = {"cases": cases, "calibration_cases": [], "rejected": [], "incompatible": []}
    fpath = out_dir / "fixtures.json"
    fpath.write_text(json.dumps(fixtures_payload, ensure_ascii=False) + "\n")
    (out_dir / "renderer-audit.json").write_text(json.dumps({c["id"]: None for c in cases}, indent=2) + "\n")
    (out_dir / "replay-smoke-audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    inv_cases = []
    for c in cases:
        max_len = max(len(t["request"]["frozen_prompt_ids"]) + t["D"] for t in c["turns"])
        inv_cases.append({
            **{k: v for k, v in c.items() if k != "turns"},
            "max_logical_tokens": max_len,
            "prior_triggers_fixed_keep": audit[c["id"]]["prior_triggers_fixed_keep"],
            "rounds": [{"H": t["H"], "P": t["P"], "D": t["D"]} for t in c["turns"]],
        })
    inv = {
        "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
        "context_aligned_256": ctx,
        "cases": inv_cases,
    }
    (out_dir / "inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2) + "\n")
    return inv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hpd-json", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/mainline-eval/mobile-gui-hpd")
    p.add_argument("--freeze-agent-base", help="If set, freeze 8 traces and verify replay against this live agent")
    args = p.parse_args()

    rows = json.loads(args.hpd_json.read_text())
    summary = write_distribution_artifacts(rows, args.output)
    print(json.dumps(summary, indent=2))
    if args.freeze_agent_base:
        inv = build_frozen_fixtures_on_agent(args.freeze_agent_base, args.output)
        print("FROZEN INVENTORY:")
        print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
