#!/usr/bin/env python3
"""按封版 Qwen3 渲染器重放 ToolSandbox 已录制消息，导出 HPD 与手机夹具。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

from a2o.workloads.toolsandbox import post
from frozen_ablation_corpus import normalize, read
from mobile_gui_hpd_and_freeze import quantile

PHONE_CONTEXT = 9472
TARGET_MODEL = "qwen3-8b-q4_0"
SOURCE_MODEL = "qwen35-9b-q4_0"
SHORT = 32


def reference_text(response):
    message = response["choices"][0]["message"]
    text = message.get("content") or ""
    for call in message.get("tool_calls") or []:
        function = call["function"]
        payload = json.dumps({"name": function["name"], "arguments": json.loads(function["arguments"])},
                             ensure_ascii=False)
        text += ("\n" if text else "") + f"<tool_call>\n{payload}\n</tool_call>"
    if not text:
        raise ValueError("empty visible assistant output")
    return text


def retokenize(directory, agent):
    result = read(directory / "result.json")
    responses = sorted(directory.glob("*.response.json"))
    if not result["trace_complete"] or len(responses) < 2:
        raise ValueError("no completed multi-turn trace")
    gaps = {}
    for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines()):
        if event["kind"] == "tool_return":
            idx = event["next_step"]
            gaps[idx] = max(gaps.get(idx, 0), event["issue_to_next_request_ms"])

    def tokenize(content):
        return post(agent, "/tokenize", {"content": content}, timeout=30)["tokens"]

    records = []
    for i, response_file in enumerate(responses):
        stem = f"{i:03d}"
        request_file = directory / f"{stem}.request.json"
        source = read(request_file)
        prompt = post(agent, "/apply-template", {k: source[k] for k in ("messages", "tools")}, timeout=30)
        records.append((i, request_file, tokenize(prompt["prompt"]),
                        tokenize(reference_text(read(response_file))), gaps.get(i)))
    case = normalize("toolsandbox", directory.name, directory.relative_to(ROOT), SOURCE_MODEL, records)
    score = result["evaluation"]["similarity"]
    full_score = bool(result["full_score"])
    return case, score, full_score


def distribution(turns):
    result = {"return_turns": len(turns)}
    for key in ("P", "H"):
        values = [t[key] for t in turns]
        result[key] = {name: quantile(values, fraction) for name, fraction in
                       (("min", 0), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90), ("max", 1))}
        if key == "P":
            for cutoff in (32, 64):
                count = sum(p <= cutoff for p in values)
                result[key][f"le_{cutoff}_count"] = count
                result[key][f"le_{cutoff}_pct"] = 100 * count / len(values)
    return result


def choose_cases(entries):
    def features(entry):
        case, score, full_score = entry
        ps = [t["P"] for t in case["turns"][1:]]
        return case, len(ps), sum(p <= SHORT for p in ps), sum(p <= 64 for p in ps), score, full_score

    pool = [features(entry) for entry in entries if
            max(t["H"] + t["P"] + t["D"] for t in entry[0]["turns"]) <= PHONE_CONTEXT]
    best = None
    for size in (3, 4):
        for group in combinations(pool, size):
            if len({item[0]["id"].removesuffix("_alt") for item in group}) != size:
                continue
            turns = sum(item[1] for item in group)
            if not 20 <= turns <= 30:
                continue
            short = sum(item[2] for item in group)
            medium = sum(item[3] for item in group)
            rank = (short / turns, medium / turns, sum(item[5] for item in group) / size,
                    sum(item[4] for item in group) / size, -abs(turns - 25))
            if best is None or rank > best[0]:
                best = (rank, group)
    if best is None:
        raise ValueError("no 3–4 compatible traces totaling 20–30 return turns")
    return [item[0] for item in best[1]]


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(args.agent + "/health", timeout=30) as response:
        health = json.load(response)
    if health["model"] != TARGET_MODEL or health["git_commit"] != "05b88107d89c7253d33f91603c878ffe348c0593":
        raise ValueError(f"Unexpected renderer identity: {health}")
    directories = sorted((ROOT / args.source).glob("*/result.json"))
    entries, rejected = [], []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [(f.parent, executor.submit(retokenize, f.parent, args.agent)) for f in directories]
        for directory, future in futures:
            try:
                entries.append(future.result())
            except (AssertionError, ValueError, KeyError) as exc:
                rejected.append({"id": directory.name, "reason": str(exc)})
    cases = choose_cases(entries)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fixtures.json"
    with path.open("x") as stream:
        json.dump({"cases": cases, "calibration_cases": [], "rejected": [], "incompatible": []},
                  stream, ensure_ascii=False)
        stream.write("\n")
    selected = {c["id"] for c in cases}
    write_json(out / "inventory.json", {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "renderer": health, "source_campaign": str(args.source),
        "source_model": SOURCE_MODEL, "reference_encoding": TARGET_MODEL,
        "cases": [{**{k: v for k, v in case.items() if k != "turns"},
                   "source_similarity": score, "source_full_score": full,
                   "rounds": [{k: t[k] for k in ("index", "H", "P", "D")} for t in case["turns"]]}
                  for case, score, full in entries if case["id"] in selected],
    })
    write_json(out / "distribution_summary.json", {
        "scenarios_recorded": len(directories), "complete_multiturn_trajectories": len(entries),
        "rejected": rejected,
        "all_complete_multiturn": distribution([t for c, _, _ in entries for t in c["turns"][1:]]),
        "frozen_subset": distribution([t for c in cases for t in c["turns"][1:]]),
    })
    with (out / "tool_return_hpd.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "turn_index", "H", "P", "D", "selected", "source_similarity", "source_full_score"])
        for case, score, full in entries:
            for t in case["turns"][1:]:
                writer.writerow([case["id"], t["index"], t["H"], t["P"], t["D"], case["id"] in selected, score, full])


if __name__ == "__main__":
    main()
