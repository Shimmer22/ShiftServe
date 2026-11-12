#!/usr/bin/env python3
"""区分输入重写、状态检查点缺口和手机锚点保守性；不启动模型。"""
import argparse,collections,json,statistics
from pathlib import Path


def common(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bfcl',required=True,type=Path);p.add_argument('--layout',required=True,type=Path);p.add_argument('--toolsandbox-cpu',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    events=[];groups=collections.defaultdict(list)
    for f in sorted(a.bfcl.glob('multi_turn_*/events.jsonl')):
        entry=json.loads((f.parent/'entry.json').read_text());es=[json.loads(x) for x in f.read_text().splitlines()];post={t.get('next_step') for t in json.loads((f.parent/'tools.json').read_text())}
        for i,r in enumerate(es):
            if not i:continue
            previous=es[i-1];old=(f.parent/(previous['step']+'.prompt.txt')).read_text();new=(f.parent/(r['step']+'.prompt.txt')).read_text()
            oi=json.loads((f.parent/(previous['step']+'.input.json')).read_text())['input_ids']+previous['generated_ids'];ni=json.loads((f.parent/(r['step']+'.input.json')).read_text())['input_ids'];h=common(oi,ni);assert h==r['H_common_prefix']
            group='post_tool' if r['step'] in post else 'new_tool_docs' if r['step'].endswith('s00') and str(r['turn']) in entry.get('held_docs',{}) else 'other_user_turn'
            normalized_old=old.replace('<think>\n\n</think>\n\n','');normalized_new=new.replace('<think>\n\n</think>\n\n','')
            row=dict(task=f.parent.name,step=r['step'],group=group,H=h,P=len(ni)-h,old_prompt_is_text_prefix=new.startswith(old),old_prompt_prefix_after_removing_empty_think=normalized_new.startswith(normalized_old))
            groups[group].append(row);events.append(row)
    classified={g:dict(n=len(rs),old_prompt_is_text_prefix=sum(r['old_prompt_is_text_prefix'] for r in rs),old_prompt_prefix_after_removing_empty_think=sum(r['old_prompt_prefix_after_removing_empty_think'] for r in rs)) for g,rs in groups.items()}
    measurements=json.loads((a.layout/'measurements.json').read_text());layout={};checkpoint=[]
    for r in measurements:
        c=r['metric_delta']['vllm:prefix_cache_hits_total'];assert 0<=c<=r['H']
        q=r['previous_tokens']//528*528;pred=q if r['H']>=q else 0
        checkpoint.append(dict(task=r['task'],mode=r['mode'],H=r['H'],previous_tokens=r['previous_tokens'],last_aligned_checkpoint=q,predicted_cached_tokens=pred,actual_cached_tokens=c,match=pred==c))
    for mode in dict.fromkeys(r['mode'] for r in measurements):
        rs=[r for r in measurements if r['mode']==mode];N=sum(r['total_tokens'] for r in rs);H=sum(r['H'] for r in rs);C=sum(r['metric_delta']['vllm:prefix_cache_hits_total'] for r in rs)
        layout[mode]=dict(n=len(rs),input_tokens=N,logical_shared_tokens=H,actual_cached_tokens=C,logical_suffix_tokens=N-H,logical_shared_but_not_restored_tokens=H-C,not_served_from_prefix_cache_tokens=N-C)
    live=[]
    for f in sorted(a.toolsandbox_cpu.glob('*/metrics.json')):
        for r in json.loads(f.read_text())['inference'][1:]:
            b=r.get('backend_timings',{})
            if 'P_evaluated' in b:live.append(dict(task=f.parent.name,step=r['step'],logical_H=r['H_common_prefix'],logical_P=r['P_append'],actual_anchor=b['H_anchor'],evaluated_P=b['P_evaluated'],extra_evaluated=b['P_evaluated']-r['P_append']))
    report=dict(scope='offline source/trace audit; no new latency or quality experiment',bfcl_adjacent_requests=len(events),input_rewrite=classified,layout_loss_decomposition=layout,checkpoint_rule=dict(block_tokens=528,source='captured server.log: attention block size set for Mamba page size',observations=len(checkpoint),matches=sum(x['match'] for x in checkpoint),caveat='empirical rule for this engine configuration and priming protocol; not a general guarantee'),phone_anchor=dict(n=len(live),extra_work_cases=sum(r['extra_evaluated']>0 for r in live),logical_P_median=statistics.median(r['logical_P'] for r in live),evaluated_P_median=statistics.median(r['evaluated_P'] for r in live),extra_evaluated_median=statistics.median(r['extra_evaluated'] for r in live),logical_P_sum=sum(r['logical_P'] for r in live),evaluated_P_sum=sum(r['evaluated_P'] for r in live),reset_despite_logical_H=sum(r['actual_anchor']==0 and r['logical_H']>0 for r in live)))
    for name,value in [('cache-loss-audit',report),('prefix-rewrites',events),('checkpoint-rule',checkpoint),('phone-anchor',live)]:
        (a.output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
