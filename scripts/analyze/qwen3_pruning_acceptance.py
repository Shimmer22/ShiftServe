#!/usr/bin/env python3
"""仅汇总保留50%/75%结构的greedy连续前缀接受率，不推导延迟收益。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    plan=json.loads((ROOT/'configs/qwen3-pruning-acceptance-verify.json').read_text())
    out=ROOT/'results'/plan['name'];assert json.loads((out/'completion.json').read_text())['completed']
    summary={}
    for case in plan['cases']:
        d=json.loads((out/case['name']/'response.json').read_text())
        assert d['completed'] and d['checkpoint_replay']['final_state_bitwise_equal']
        blocks=d['acceptance_blocks'];lengths=[b['accepted_prefix'] for b in blocks];proposed=sum(b['proposed'] for b in blocks)
        summary[case['name']]=dict(accepted=sum(lengths),proposed=proposed,prefix_fraction=sum(lengths)/proposed,mean_prefix=sum(lengths)/len(blocks),first_token_match=sum(v>0 for v in lengths)/len(blocks),full_block_match=sum(b['accepted_prefix']==b['proposed'] for b in blocks)/len(blocks),lengths=lengths,anchors=[b['anchor'] for b in blocks])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
