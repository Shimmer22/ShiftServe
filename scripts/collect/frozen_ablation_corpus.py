#!/usr/bin/env python3
"""从已冻结的运行记录构造可逐 token 复验的多轮消融夹具。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

EMPTY_THINK = [151667, 271, 151668, 271]
USER_HEADER = [151644, 872, 198]
CTX = 14336


def read(path):
    return json.loads(path.read_text())


def normalize(benchmark, name, source, source_model, records):
    previous = []
    turns = []
    for stem, request_file, prompt, decode, gap in records:
        request = read(request_file)
        assert prompt[-4:] == EMPTY_THINK, f"{stem}: missing exact Qwen3 empty thinking suffix"
        assert prompt[:len(previous)] == previous and len(prompt)-4 > len(previous), f"{stem}: broken inherited anchor"
        assert decode and min(decode) >= 0 and max(decode) < 151936, f"{stem}: invalid decode IDs"
        assert len(prompt) + len(decode) <= CTX, f"{stem}: context overflow"
        messages = request["messages"]
        if turns:
            assert messages[-1]["role"] == "tool", f"{stem}: boundary does not follow a tool result"
        turns.append(dict(index=stem, H=len(previous), P=len(prompt)-len(previous), D=len(decode),
                          observed_gap_ms=gap, request={**{k: request[k] for k in ('messages','tools') if k in request},
                              "frozen_prompt_ids":prompt, "frozen_decode_ids":decode}))
        previous = prompt[:-4]
    if len(turns) < 2:
        raise ValueError("no tool return")
    first = turns[0]["request"]["frozen_prompt_ids"]
    pin = next((i for i in range(len(first)-2) if first[i:i+3] == USER_HEADER), 0)
    keep = max(pin + 32 + 24, round(0.75*(turns[1]["H"]+turns[1]["P"])))
    return dict(benchmark=benchmark, id=name, source=str(source), source_model=source_model,
                pinned_estimate=pin, keep_tokens=keep, turns=turns)


def collect(source):
    cases, rejected = [], []
    tau = source/'20261118-tau2-host-qwen3-4b-q4-vllm'
    index = read(tau/'analysis/tool-chains.json')
    for row in index:
        trace = tau/'traces'/Path(row['trace_dir']).parent.name/Path(row['trace_dir']).name
        name = row['task_key'] + '-' + str(row['indices'][0])
        try:
            gaps = {x['index']:x['gap_ms'] for x in row['boundaries']}
            records = []
            for i in row['indices']:
                base = f'{i:03d}-agent'
                data = read(trace/f'{base}.response.json')
                records.append((i, trace/f'{base}.request.json', data['prompt_token_ids'],
                                data['choices'][0]['token_ids'], gaps.get(i)))
            cases.append(normalize('tau2-'+row['domain'],name,trace,'qwen3-4b-q4_0',records))
        except (AssertionError, KeyError, ValueError, OSError) as exc:
            rejected.append(dict(id=name,error=str(exc)))

    tool = source/'20261114-qwen3-8b-kv-toolsandbox/baseline'
    for task in sorted(x for x in tool.iterdir() if x.is_dir()):
        try:
            records = []
            for item in sorted(task.glob('*.response.json')):
                stem = item.name.split('.')[0]
                record = read(item)
                records.append((stem,task/f'{stem}.request.json',read(task/f'{stem}.prompt-ids.json'),
                                record['__verbose']['tokens'],None))
            cases.append(normalize('toolsandbox',task.name,task,'qwen3-8b-q4_0',records))
        except (AssertionError, KeyError, ValueError, OSError) as exc:
            rejected.append(dict(id=task.name,error=str(exc)))

    bfcl = source/'20261108-bfcl-text-pilot'
    incompatible = [dict(benchmark='bfcl',source=str(bfcl),input_records=len(list(bfcl.glob('*/*.input.json'))),
                         response_records=len(list(bfcl.glob('*/*.response.json'))),
                         reason='旧 Qwen3.5-9B 词表，保留原始冻结 token，不能混入 Qwen3-4B/8B 主表')]
    return cases,rejected,incompatible


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases,rejected,incompatible=collect(args.source_results.resolve())
    reserved={'find_days_till_holiday','airline-3fdba35f04dc-1','telecom-25e67b464860-1'}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'fixtures.json'
    path.write_text(json.dumps(dict(cases=cases,calibration_cases=sorted(reserved),
                                    rejected=rejected,incompatible=incompatible),ensure_ascii=False)+'\n')
    index=[]
    for case in cases:
        index.append({**{k:v for k,v in case.items() if k!='turns'},
                      'calibration':case['id'] in reserved,
                      'rounds':[dict(H=t['H'],P=t['P'],D=t['D'],observed_gap_ms=t['observed_gap_ms']) for t in case['turns']]})
    report=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                chains=dict(Counter(c['benchmark'] for c in cases)), cases=index,
                rejected=rejected,incompatible=incompatible)
    (args.output/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
