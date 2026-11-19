#!/usr/bin/env python3
"""统计全部 τ² 轨迹并按领域/P 分位选择手机 pilot，不按任务 reward 筛选。"""
import argparse
from collections import Counter
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from a2o.profiling.bfcl_attention import save


def common_prefix(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))


def distribution(values):
    values=sorted(values)
    if not values:
        return dict(n=0)
    def q(f):
        return values[round((len(values)-1)*f)]
    return dict(n=len(values),min=values[0],p25=q(.25),median=q(.5),p75=q(.75),p90=q(.9),max=values[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tasks-per-domain',type=int,default=4)
    parser.add_argument('--context-size',type=int,default=16384)
    parser.add_argument('--strip-anchor-tail-ids', type=int, nargs='+', default=[],
                        help='Only strip this exact token suffix from the saved anchor before matching')
    parser.add_argument('--min-tool-chain-turns', type=int, default=0,
                        help='Also export contiguous tool-only replay chains with at least this many agent turns')
    parser.add_argument('--gap-kind',choices=['tool','balanced'],default='balanced')
    parser.add_argument('--quantiles',default=None,help='显式P分位，例如0.25,0.75；数量等于每域任务数')
    args=parser.parse_args()
    quantiles=[float(q) for q in args.quantiles.split(',')] if args.quantiles else [i/max(1,args.tasks_per_domain-1) for i in range(args.tasks_per_domain)]
    assert len(quantiles)==args.tasks_per_domain and all(0<=q<=1 for q in quantiles)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.input/'manifest.json').read_text())
    rows=[];excluded=[];task_results=[]
    for task in manifest['tasks']:
        folder=args.input/task['key']
        result=json.loads((folder/'result.json').read_text())
        task_results.append(result)
        attempt=folder/result['attempt']
        events=[json.loads(line) for line in (attempt/'events.jsonl').read_text().splitlines()]
        previous=None
        for path in sorted(attempt.glob('*-agent.response.json')):
            index=int(path.name.split('-')[0]);response=json.loads(path.read_text())
            ids=response['prompt_token_ids'];generated=response['choices'][0]['token_ids']
            current_event=next(e for e in events if e['kind']=='llm_response' and e['index']==index)
            if previous is not None:
                old,old_event=previous
                h=common_prefix(old['prompt_token_ids']+old['choices'][0]['token_ids'],ids)
                anchor_ids=old['prompt_token_ids']
                tail=args.strip_anchor_tail_ids
                if tail and len(anchor_ids)>len(tail) and anchor_ids[-len(tail):]==tail:
                    anchor_ids=anchor_ids[:-len(tail)]
                anchor_h=common_prefix(anchor_ids,ids)
                between=[e for e in events if old_event['response_ns']<=e['monotonic_ns']<=current_event['request_ns']]
                tool_events=[e for e in between if e['kind']=='tool_result' and e['call']['requestor']=='assistant']
                user_events=[e for e in between if e['kind']=='llm_response' and e['role']=='user']
                gap_kind='user' if user_events else 'tool' if tool_events else 'other'
                gap=(current_event['request_ns']-old_event['response_ns'])/1e6
                execution=(max(e['end_ns'] for e in tool_events)-min(e['start_ns'] for e in tool_events))/1e6 if tool_events else None
                reusable=anchor_h==len(anchor_ids)
                # 正式服务只保存完整上一输入锚点；不可裁剪 recurrent 状态来伪造 H。
                physical_h=len(anchor_ids) if reusable else 0
                row=dict(domain=task['domain'],task_id=task['id'],task_key=task['key'],index=index,
                    previous_index=old_event['index'],
                    H=physical_h,P=len(ids)-physical_h,logical_H=h,logical_P=len(ids)-h,
                    D=len(generated),N=len(ids),gap_kind=gap_kind,gap_ms=gap,
                    tool_execution_ms=execution,anchor_H=anchor_h,anchor_P=len(ids)-anchor_h,
                    anchor_fully_reusable=reusable,
                    finish_reason=response['choices'][0]['finish_reason'],
                    source_response=str(path),task_complete=result['complete'],reward=result.get('reward'))
                rows.append(row)
                reasons=[]
                if not reusable:reasons.append('saved anchor is not a full prefix; live service must recompute on CPU')
                if not row['H'] or not row['P']:reasons.append('empty replay shape')
                if row['finish_reason']=='length':reasons.append('incomplete output')
                if len(ids)+len(generated)>=args.context_size:reasons.append('context capacity')
                if gap_kind=='other':reasons.append('unsupported gap kind')
                if reasons:
                    excluded.append(dict(**row,reasons=reasons))
            previous=(response,current_event)
    invalid={(r['task_key'],r['index']) for r in excluded}
    candidates=[r for r in rows if (r['task_key'],r['index']) not in invalid]
    selected=[]
    # 每域四个 P 分位、每任务最多一个边界；完全不读取 reward 作为选择条件。
    for domain in manifest['config']['domains']:
        pool=sorted([r for r in candidates if r['domain']==domain and (args.gap_kind=='balanced' or r['gap_kind']=='tool')],key=lambda r:(r['P'],r['task_key'],r['index']))
        used=set()
        if len({r['task_key'] for r in pool})<args.tasks_per_domain:
            raise RuntimeError(f'{domain}: insufficient distinct tasks for tool-boundary pilot')
        for i in range(args.tasks_per_domain):
            kind='user' if args.gap_kind=='balanced' and i%2 else 'tool'
            subpool=[r for r in pool if r['gap_kind']==kind]
            if not subpool:raise RuntimeError(f'{domain}: missing {kind} boundaries')
            target=subpool[round((len(subpool)-1)*quantiles[i])]['P']
            choices=[r for r in subpool if r['task_key'] not in used]
            if not choices:raise RuntimeError(f'{domain}: insufficient distinct tasks for {kind} boundaries')
            chosen=min(choices,key=lambda r:(abs(r['P']-target),r['task_key'],r['index']))
            selected.append(dict(chosen,selection_P_target=target))
            used.add(chosen['task_key'])
    requests=[]
    for i,row in enumerate(selected):
        response=json.loads(Path(row['source_response']).read_text())
        ids=response['prompt_token_ids']
        generated=response['choices'][0]['token_ids']
        requests.append(dict(boundary=i,**row,prefix_ids=ids[:row['H']],input_ids=ids[row['H']:],generated_ids=generated,
            source_sha256=hashlib.sha256(Path(row['source_response']).read_bytes()).hexdigest()))
    summary=dict(attempted=len(task_results),complete=sum(r['complete'] for r in task_results),
        scored_successes=sum(r.get('reward')==1 for r in task_results),
        boundary_count=len(rows),gap_counts=dict(Counter(r['gap_kind'] for r in rows)),
        distributions={k:distribution([r[k] for r in rows]) for k in ['H','P','D','gap_ms','anchor_H','anchor_P','logical_H','logical_P']},
        reusable_anchors=sum(r['anchor_fully_reusable'] for r in rows),
        by_domain={d:dict(tasks=sum(r['domain']==d for r in task_results),
            complete=sum(r['domain']==d and r['complete'] for r in task_results),
            scored_successes=sum(r['domain']==d and r.get('reward')==1 for r in task_results),
            **{k:distribution([r[k] for r in rows if r['domain']==d]) for k in ['H','P','D']}) for d in manifest['config']['domains']},
        excluded_count=len(excluded),selected_tasks=len(requests),context_size=args.context_size,
        strip_anchor_tail_ids=args.strip_anchor_tail_ids,
        selection=f'每域{args.tasks_per_domain}个不同任务、P分位{quantiles}；gap_kind={args.gap_kind}；只取完整可恢复锚点，不按质量筛选；冻结性能pilot，不是全任务手机评分')
    save(args.output/'boundaries.json',rows)
    save(args.output/'excluded.json',excluded)
    save(args.output/'requests.json',requests)
    if args.min_tool_chain_turns:
        chains=[]
        for _, task_rows in groupby(rows, key=lambda r: r['task_key']):
            for eligible, chain in groupby(task_rows, key=lambda r: r['gap_kind']=='tool'
                                           and (r['task_key'],r['index']) not in invalid):
                boundaries=list(chain)
                if eligible and len(boundaries)+1>=args.min_tool_chain_turns:
                    first=boundaries[0]
                    chains.append(dict(domain=first['domain'],task_key=first['task_key'],
                        trace_dir=str(Path(first['source_response']).parent.resolve()),
                        indices=[first['previous_index'],*(r['index'] for r in boundaries)],
                        boundaries=boundaries))
        save(args.output/'tool-chains.json',chains)
        summary['tool_chains']=len(chains)
    save(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
