#!/usr/bin/env python3
"""独立校准上的 P 混合分布敏感性实验；不当作在线实测收益。"""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from a2o.scheduling.speculative_value import expected_value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    output=[]
    for run in a.runs:
        rows=[json.loads(x) for x in (run/'checked-rows.jsonl').read_text().splitlines()]
        cal=[r for r in rows if r['phase']=='calibration']
        table={}
        for p in sorted({r['P'] for r in cal}):
            group=[r for r in cal if r['P']==p]
            ok=len(group)==6 and all(r['passed'] and r['numerical_passed'] and r['response_check']['passed'] for r in group)
            table[p]=tuple(statistics.median(r['prefill_ms']+r['handoff_checked_ms']+r['decode_ms'] for r in group if r['route']==route) for route in ('CC','HC'))+(ok,)
        prep=max(r['prepare_ms'] for r in cal if r['route']=='HC')
        sweep=[]
        for q in [0,.1,.25,.5,.75,.9,1.]:
            for gap in [0,50,200]:
                for weight in [0,.25,.5,1.]:
                    decision=expected_value({24:1-q,32:q},table,prep,gap,weight=weight)
                    sweep.append(dict(probability_P32=q,gap_ms=gap,**decision))
        output.append(dict(run=str(run.resolve()),H=cal[0]['H'],costs=table,prepare_upper_ms=prep,sweep=sweep))
    (a.output/'sensitivity.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([dict(H=x['H'],costs=x['costs'],prepare_upper_ms=x['prepare_upper_ms']) for x in output],indent=2))

if __name__=='__main__': main()
