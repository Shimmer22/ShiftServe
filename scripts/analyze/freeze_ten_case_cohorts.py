#!/usr/bin/env python3
"""导出并校验 10+10+10 三套冻结测试夹具：ToolSandbox 10 条 / DroidTask 10 条 / AC-Low 10 条。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path.cwd() if (Path.cwd() / "configs/host-kv-accuracy-androidcontrol-low-8b.json").exists() else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))
sys.path.insert(0, str(ROOT / "scripts/analyze"))

EMPTY_THINK = [151667, 271, 151668, 271]
USER_HEADER = [151644, 872, 198]

DROIDTASK_10 = [
    "firefox_b6056f6eeb89",
    "calendar_e8b601c47a04",
    "calendar_77b4402acf90",
    "gallery_68284a77e92c",
    "filemanager_e269c90eebf3",
    "calendar_08cd9f2b2ff5",
    "contacts_3f70d4da52a3",
    "dialer_527fae6860e5",
    "notes_9ec4d898bf75",
    "filemanager_58d3cd387bfc",
]

AC_LOW_10 = [
    "ac_ep_8982",
    "ac_ep_6119",
    "ac_ep_13841",
    "ac_ep_2405",
    "ac_ep_4451",
    "ac_ep_9464",
    "ac_ep_15690",
    "ac_ep_17621",
    "ac_ep_13018",
    "ac_ep_4692",
]

TOOLSANDBOX_10 = [
    "turn_on_cellular_low_battery_mode_implicit",
    "turn_on_location_low_battery_mode_implicit",
    "turn_on_wifi_low_battery_mode_implicit",
    "find_temperature_low_battery_mode",
    "find_current_city_low_battery_mode",
    "find_stock_symbol_with_company_name_low_battery_mode",
    "find_temperature_f_with_location_wifi_off",
    "modify_contact_with_message_recency_alt",
    "send_message_with_contact_content_cellular_off",
    "find_days_till_holiday_wifi_off",
]


def post_json(agent_base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        agent_base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def validate_case(case: dict, max_ctx: int = 9472) -> tuple[int, int]:
    prev: list[int] = []
    max_len = 0
    for idx, turn in enumerate(case["turns"]):
        req = turn["request"]
        pids, dids = req["frozen_prompt_ids"], req["frozen_decode_ids"]
        assert pids[-4:] == EMPTY_THINK, f"{case['id']} turn {idx} missing EMPTY_THINK"
        assert pids[:len(prev)] == prev and len(pids) - len(prev) == turn["P"], f"{case['id']} turn {idx} bad H/P"
        assert len(prev) == turn["H"] and len(dids) == turn["D"], f"{case['id']} turn {idx} bad size"
        max_len = max(max_len, len(pids) + len(dids))
        prev = pids[:-4]
    assert max_len <= max_ctx, f"{case['id']} max_len={max_len} > {max_ctx}"
    return len(case["turns"]) - 1, max_len


def freeze_mobile_case(agent: str, benchmark: str, cid: str, source: str, sys_msg: str, users: list[str], assts: list[str]) -> dict:
    msgs = [{"role": "system", "content": sys_msg}]
    prev: list[int] = []
    turns = []
    for idx, (uc, ac) in enumerate(zip(users, assts)):
        assert "<think>" not in ac and "</think>" not in ac, f"{cid} turn {idx} contains <think>"
        msgs.append({"role": "user", "content": uc})
        req_msgs = [dict(m) for m in msgs]
        rendered = post_json(agent, "/apply-template", {"messages": req_msgs})["prompt"]
        prompt_ids = post_json(agent, "/tokenize", {"content": rendered})["tokens"]
        decode_ids = post_json(agent, "/tokenize", {"content": ac})["tokens"]
        assert prompt_ids[-4:] == EMPTY_THINK
        assert prompt_ids[:len(prev)] == prev and len(prompt_ids) - 4 > len(prev)
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
    return {
        "benchmark": benchmark,
        "id": cid,
        "source": source,
        "source_model": "qwen3-8b-q4_0-pure-action",
        "pinned_estimate": pin,
        "keep_tokens": keep,
        "turns": turns,
    }


def export_mobile_from_wins(agent: str, out_json: Path) -> None:
    import host_androidcontrol_worker as ac_w
    import host_droidtask_worker as dt_w

    dt_tasks = {t["id"]: t for t in dt_w._load_tasks(ROOT / "scenarios/droidtask/profile_no_answer.json", 0)}
    ac_cfg = json.loads((ROOT / "configs/host-kv-accuracy-androidcontrol-low-8b.json").read_text())
    ac_eps = {e["id"]: e for e in ac_w.load_episodes(Path(ac_cfg["dataset_path"]), Path(ac_cfg["splits_path"]), 4)}

    cases = []
    for tid in DROIDTASK_10:
        t = dt_tasks[tid]
        sys_msg = f"{dt_w._SYSTEM_PROMPT}\n\nApp: {t['app']}\nTask: {t['task']}"
        users = [f"Step {i} UI state:\n{s['state']}" for i, s in enumerate(t["steps"])]
        assts = [dt_w._gold_action_json(s["label"]) for s in t["steps"]]
        cases.append(freeze_mobile_case(agent, "droidtask", tid, "scenarios/droidtask/profile_no_answer.json", sys_msg, users, assts))

    for eid in AC_LOW_10:
        ep = ac_eps[eid]
        sys_msg = f"{ac_w._SYSTEM_PROMPT}\n\nGoal: {ep['goal']}"
        users = [
            f"Step {i} instruction: {s['instruction']}\nStep {i} UI tree:\n{s['rendered']}" if s.get("instruction")
            else f"Step {i} UI tree:\n{s['rendered']}"
            for i, s in enumerate(ep["steps"])
        ]
        assts = [ac_w.gold_action_json(s) for s in ep["steps"]]
        cases.append(freeze_mobile_case(agent, "androidcontrol_low", eid, ac_cfg["dataset_path"], sys_msg, users, assts))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"cases": cases}, ensure_ascii=False) + "\n")
    print(f"Exported {len(cases)} mobile cases to {out_json}")


def write_cohort_dir(out_dir: Path, cases: list[dict]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_total, max_len = 0, 0
    details = []
    for c in cases:
        w, ml = validate_case(c)
        warm_total += w
        max_len = max(max_len, ml)
        details.append({"id": c["id"], "benchmark": c["benchmark"], "warm_turns": w, "max_logical_tokens": ml})
    fpath = out_dir / "fixtures.json"
    fpath.write_text(json.dumps({"cases": cases, "calibration_cases": [], "rejected": [], "incompatible": []}, ensure_ascii=False) + "\n")
    summary = {
        "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
        "cases": len(cases),
        "warm_turns": warm_total,
        "max_logical_tokens": max_len,
        "details": details,
    }
    (out_dir / "cohort-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_win = sub.add_parser("wins-mobile", help="在 wins 上运行：从原始数据集提取并冻结 DT10+ACL10")
    p_win.add_argument("--agent", required=True)
    p_win.add_argument("--out-json", type=Path, required=True)

    p_loc = sub.add_parser("assemble-local", help="在本地运行：冻结 ToolSandbox10 并拆分写入三套目录")
    p_loc.add_argument("--agent", required=True)
    p_loc.add_argument("--wins-mobile-json", type=Path, required=True)
    p_loc.add_argument("--ts-source", type=Path, default=ROOT / "results/_archive/20261107-toolsandbox-full/cpu")
    p_loc.add_argument("--out-base", type=Path, default=ROOT / "results")
    args = p.parse_args()

    if args.cmd == "wins-mobile":
        export_mobile_from_wins(args.agent, args.out_json)
        return

    from toolsandbox_hpd_and_freeze import retokenize

    mobile = json.loads(args.wins_mobile_json.read_text())["cases"]
    dt_cases = [c for c in mobile if c["benchmark"] == "droidtask"]
    ac_cases = [c for c in mobile if c["benchmark"] == "androidcontrol_low"]
    assert [c["id"] for c in dt_cases] == DROIDTASK_10
    assert [c["id"] for c in ac_cases] == AC_LOW_10

    existing_ts = {
        c["id"]: c
        for c in json.loads((ROOT / "results/20261124-toolsandbox-phone-fixtures/fixtures.json").read_text())["cases"]
    }
    ts_cases = []
    for tid in TOOLSANDBOX_10:
        if tid in existing_ts:
            ts_cases.append(existing_ts[tid])
        else:
            case, _, _ = retokenize(args.ts_source / tid, args.agent)
            ts_cases.append(case)

    s_ts = write_cohort_dir(args.out_base / "20261124-toolsandbox-10cases", ts_cases)
    s_dt = write_cohort_dir(args.out_base / "20261124-droidtask-10cases", dt_cases)
    s_ac = write_cohort_dir(args.out_base / "20261124-aclow-10cases", ac_cases)
    print(json.dumps({"toolsandbox": s_ts, "droidtask": s_dt, "aclow": s_ac}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
