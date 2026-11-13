#!/usr/bin/env python3
"""汇总结构扩展、prefill选择和greedy前缀接受率；不把token相似度当接受率。"""
import json
from pathlib import Path
from statistics import median
from decode_attribution import read_case
ROOT=Path(__file__).resolve().parents[2]


def main():
    summary={}
    for config in ['qwen3-expanded-sparse','qwen3-prefill-prior-speed']:
        plan=json.loads((ROOT/'configs'/f'{config}.json').read_text());out=ROOT/'results'/plan['name']
        assert json.loads((out/'completion.json').read_text())['completed']
        cases={}
        for case in plan['cases']:
            d,s,_=read_case(out/case['name']);assert d['warmup_elapsed_ms']>=40000 and d['checkpoint_replay']['final_state_bitwise_equal']
            r=[r for r in d['rows'] if not r['warmup']]
            s.update(ffn_packing_ms=d['ffn_compaction']['packing_ms'],query_packing_ms=d['query_compaction']['packing_ms'],
                prefill_ms=d['full_prefill_ms'],major_fault_deltas=[r['resources_after']['major_faults']-r['resources_before']['major_faults'] for r in r],swap_kib=[r['resources_after']['VmSwap_kib'] for r in r])
            cases[case['name']]=s
        summary[config]=cases
    verify=ROOT/'results/20261113-qwen3-prefill-prior-verify'
    assert json.loads((verify/'completion.json').read_text())['completed']
    acceptance={}
    for p in sorted(verify.glob('verify-*/response.json')):
        d=json.loads(p.read_text());assert d['completed'] and d['checkpoint_replay']['final_state_bitwise_equal']
        blocks=d['acceptance_blocks'];accepted=sum(b['accepted_prefix'] for b in blocks);proposed=sum(b['proposed'] for b in blocks)
        acceptance[p.parent.name]=dict(accepted_prefix_tokens=accepted,proposed_tokens=proposed,prefix_acceptance_fraction=accepted/proposed,mean_accepted_prefix=accepted/len(blocks),blocks=blocks)
    control=json.loads((ROOT/'results/20261113-qwen3-acceptance-control-verify/cpu-control/response.json').read_text())
    assert control['completed'] and all(b['accepted_prefix']==b['proposed'] for b in control['acceptance_blocks'])
    summary['acceptance_control']=control['acceptance_blocks']
    summary['greedy_acceptance']=acceptance
    out=ROOT/'results/20261113-qwen3-expanded-sparse';(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
