#!/usr/bin/env python3
"""保留全部相邻反事实，汇总 P 预测与延迟/准备占用取舍。"""
import argparse
import json
from pathlib import Path
import statistics
import random
from collections import Counter


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    assert json.loads((a.run/'completion.json').read_text())['passed']
    rows=[json.loads(x) for x in (a.run/'checked-rows.jsonl').read_text().splitlines()]
    assert all(r['passed'] and r['response_check']['passed'] for r in rows)
    assert all(r['numerical_passed'] for r in rows if r['phase']=='calibration')
    measurement=[r for r in rows if r['phase']=='measurement']
    pairs=[]
    history=[]
    config=json.loads((a.run/'manifest.json').read_text())['config']
    assert len(measurement)==2*len(config['p_value_sequence'])
    assert sorted({r['repetition'] for r in measurement})==list(range(len(config['p_value_sequence'])))
    for block in sorted({r['repetition'] for r in measurement}):
        group=[r for r in measurement if r['repetition']==block]
        assert len(group)==2
        always=next(r for r in group if r['strategy']=='always')
        ev=next(r for r in group if r['strategy']=='cost_aware')
        assert always['P']==ev['P']==config['p_value_sequence'][block]
        decision=ev['speculative_value']
        probs = {int(k): v for k,v in decision.get('distribution', {}).items()}
        previous=history[-config['p_value_window']:]
        expected={p:n/len(previous) for p,n in Counter(previous).items()}
        assert probs==expected, 'P predictor leaked future or used wrong window'
        assert ev['execution_strategy']==('cost_aware' if decision['prepare'] else 'after_return')
        history.append(ev['P'])
        brier = sum((probs.get(p,0.) - int(p==ev['P']))**2 for p in (24,32)) if probs else None
        pairs.append(dict(block=block,P=ev['P'],prepare=decision['prepare'],prediction=decision,brier=brier,
            latency_delta_ms=ev['tool_to_response_ms']-always['tool_to_response_ms'],
            response_delta_ms=ev['response_after_return_ms']-always['response_after_return_ms'],
            actual_gap_delta_ms=ev['actual_gap_ms']-always['actual_gap_ms'],
            prepare_saved_ms=always['prepare_ms']-ev['prepare_ms'],
            always_route=always['route'],ev_route=ev['route']))
    rng=random.Random(20261106)
    deltas=[p['latency_delta_ms'] for p in pairs]
    bootstrap=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(5000))
    regimes=[]
    for name,group in [('P24',pairs[:12]),('P32',pairs[12:24]),('alternating',pairs[24:])]:
        regimes.append(dict(name=name,n=len(group),prepare_rate=statistics.mean(p['prepare'] for p in group),
            latency_delta_mean_ms=statistics.mean(p['latency_delta_ms'] for p in group),
            response_delta_mean_ms=statistics.mean(p['response_delta_ms'] for p in group),
            actual_gap_delta_mean_ms=statistics.mean(p['actual_gap_delta_ms'] for p in group),
            prepare_saved_mean_ms=statistics.mean(p['prepare_saved_ms'] for p in group)))
    report=dict(run=str(a.run.resolve()),accepted=True,paired_decisions=len(pairs),
        always_latency_mean_ms=statistics.mean(r['tool_to_response_ms'] for r in measurement if r['strategy']=='always'),
        ev_latency_mean_ms=statistics.mean(r['tool_to_response_ms'] for r in measurement if r['strategy']=='cost_aware'),pairs=pairs,regimes=regimes,prepare_rate=statistics.mean(p['prepare'] for p in pairs),
        response_delta_mean_ms=statistics.mean(p['response_delta_ms'] for p in pairs),
        actual_gap_delta_mean_ms=statistics.mean(p['actual_gap_delta_ms'] for p in pairs),
        latency_delta_mean_ms=statistics.mean(deltas),latency_delta_median_ms=statistics.median(deltas),
        paired_bootstrap_mean_95_ms=[bootstrap[125],bootstrap[4874]],
        prepare_saved_mean_ms=statistics.mean(p['prepare_saved_ms'] for p in pairs),
        limitation='单会话合成P序列；配对bootstrap不代表跨会话置信度；无能耗测量，weight为显式敏感性参数。')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'device-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
