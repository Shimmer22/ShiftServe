#!/usr/bin/env python3
"""汇总主机布局实验，分开报告逻辑 H、实际缓存命中、prefill 和调用变化。"""
import argparse,ast,collections,json,statistics
from pathlib import Path


def semantic(calls):
    if calls is None:return None
    out=[]
    for text in calls:
        node=ast.parse(text,mode='eval').body
        out.append((node.func.id,[ast.literal_eval(v) for v in node.args],{v.arg:ast.literal_eval(v.value) for v in node.keywords}))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'results/20261108-bfcl-text-pilot');a=p.parse_args();rows=json.loads((a.directory/'measurements.json').read_text());pre=json.loads((a.directory/'preflight.json').read_text());modes=list(dict.fromkeys(r['mode'] for r in pre['rows']));groups={m:[r for r in rows if r['mode']==m] for m in modes}
    by={(r['task'],r['step'],r['mode']):r for r in rows};summary={}
    for mode,rs in groups.items():
        if not rs:continue
        pref=[r['metric_delta'].get('vllm:request_prefill_time_seconds_sum') for r in rs]
        assert all(x is not None and x>=0 for x in pref)
        for r in rs:
            assert r['metric_delta']['vllm:request_prefill_time_seconds_count']==1
            assert r['metric_delta']['vllm:prefix_cache_queries_total']==r['total_tokens']
        comparisons=[(by.get((r['task'],r['step'],'native')),r) for r in rs]
        pairs=[(b,r) for b,r in comparisons if b is not None]
        summary[mode]=dict(n=len(rs),logical_P_median=statistics.median(r['P'] for r in rs),logical_P_sum=sum(r['P'] for r in rs),total_input_tokens=sum(r['total_tokens'] for r in rs),cached_tokens=sum(r['metric_delta']['vllm:prefix_cache_hits_total'] for r in rs),physical_cache_hit_cases=sum(r['metric_delta']['vllm:prefix_cache_hits_total']>0 for r in rs),prefill_s_sum=sum(pref),prefill_s_median=statistics.median(pref),wall_s_sum=sum(r['wall_s'] for r in rs),output_tokens=sum(r['D'] for r in rs),canonical_strings_match_original=sum(r['calls_match'] for r in rs),calls_match_original=sum(r['parser_error'] is None and semantic(r['actual_calls'])==semantic(r['expected_calls']) for r in rs),calls_match_current_native=sum(semantic(r['actual_calls'])==semantic(b['actual_calls']) and r['parser_error'] is None for b,r in pairs),paired_native_n=len(pairs),length_stops=sum(r['finish_reason']=='length' for r in rs),parser_errors=sum(r['parser_error'] is not None for r in rs),changes=[dict(task=r['task'],step=r['step'],original=r['expected_calls'],actual=r['actual_calls']) for r in rs if r['parser_error'] is not None or semantic(r['actual_calls'])!=semantic(r['expected_calls'])])
    for mode,entry in summary.items():
        passed=[]
        for r in groups[mode]:
            path=a.source/r['task']/'result.json'
            if json.loads(path.read_text())['quality'].get('valid',False):passed.append(r)
        entry['originally_passed_targets']=len(passed)
        entry['originally_passed_targets_calls_match']=sum(r['parser_error'] is None and semantic(r['actual_calls'])==semantic(r['expected_calls']) for r in passed)
    save=dict(scope='18 frozen release boundaries; four layouts; one pass with rotated order; host CUDA BF16; not live task quality or phone latency',complete=len(rows)==pre['cases']*len(modes),summary=summary)
    (a.directory/'summary.json').write_text(json.dumps(save,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(save,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
