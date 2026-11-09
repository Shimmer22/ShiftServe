#!/usr/bin/env python3
"""从保存的 BFCL prompt/token/工具原文分析结构、重复率与跨情节前缀。"""
import argparse,ast,collections,csv,hashlib,json,statistics
from pathlib import Path


def parse(text):
    try:return json.loads(text),'json'
    except (ValueError,TypeError):
        try:return ast.literal_eval(text),'python_literal'
        except (ValueError,SyntaxError):return text,'text'


def shape(v):
    if isinstance(v,dict):return {k:shape(x) for k,x in sorted(v.items())}
    if isinstance(v,list):return {'array_element_shapes':sorted({dump(shape(x)) for x in v})}
    return type(v).__name__


def dump(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def lcp(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def stats(values):
    v=sorted(values)
    return {'n':len(v),'median':statistics.median(v),'p90':v[int((len(v)-1)*.9)],'min':v[0],'max':v[-1]} if v else {'n':0}


def repetition(values):
    c=collections.Counter(values);n=len(values)
    return dict(n=n,unique=len(c),redundant_fraction=(n-len(c))/n if n else 0,
                repeated_observation_fraction=sum(v for v in c.values() if v>1)/n if n else 0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--toolsandbox-cpu',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tools=[];boundaries=[];files={};prompts=[]
    for result in sorted(a.input.glob('multi_turn_*/result.json')):
        folder=result.parent
        for name in ['result.json','tools.json','entry.json']:
            path=folder/name;files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        r=json.loads(result.read_text());task=r['id'];scenario=task.rsplit('_',1)[1];passed=bool(r['quality'].get('valid'));groups=collections.defaultdict(list)
        for t in json.loads((folder/'tools.json').read_text()):
            value,fmt=parse(t['content']);record=dict(task=task,scenario=scenario,passed=passed,tool=t['call']['name'],step=t['id'],next_step=t.get('next_step'),P=t.get('P_append'),P_body=t['result_tokens'],format=fmt,value_type=type(value).__name__,raw=t['content'],canonical=dump(value),shape=dump(shape(value)),top_keys=tuple(sorted(value)) if isinstance(value,dict) else (),error_flag=t['error'],error_field=isinstance(value,dict) and 'error' in value)
            tools.append(record)
            if t.get('next_step') is not None:groups[t['next_step']].append((t,record))
        for step,g in groups.items():
            path=folder/(step+'.input.json');files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();inp=json.loads(path.read_text());h=g[0][0]['H_common_prefix'];P=g[0][0]['P_append'];ids=inp['input_ids'];assert len(ids)-h==P
            spans={s['id']:s for s in inp['tool_spans']};body=0
            for t,_ in g:
                s=spans[t['id']];body+=max(0,min(s['end'],len(ids))-max(s['start'],h))
            boundaries.append(dict(task=task,scenario=scenario,passed=passed,step=step,H=h,P=P,D=g[0][0].get('D_next'),tool_count=len(g),tool_names=tuple(t['call']['name'] for t,_ in g),body_tokens_in_append=body,nonbody_tokens=P-body,raw=dump([t['content'] for t,_ in g]),canonical=dump([parse(t['content'])[0] for t,_ in g]),shape=dump([shape(parse(t['content'])[0]) for t,_ in g]),leading_nonbody_tokens=max(0,min(spans[t['id']]['start'] for t,_ in g)-h),suffix=ids[h:],history_preserved=all(h==t['H_issue'] for t,_ in g),G_ms=g[0][0]['G_ms']))
        first=folder/'t00-s00.input.json'
        if first.exists():prompts.append(dict(task=task,scenario=scenario,ids=json.loads(first.read_text())['input_ids']))
    # Exclude all variants of the same base scenario; shared seeds make category
    # replicas dependent. Candidate lookup only uses known tool names/order.
    for b in boundaries:
        candidates=[z for z in boundaries if z['scenario']!=b['scenario'] and z['tool_names']==b['tool_names']]
        b['other_scenario_candidates']=len(candidates)
        b['common_candidate_prefix_tokens']=min((lcp(candidates[0]['suffix'],z['suffix']) for z in candidates),default=0) if candidates else 0
        b['common_candidate_prefix_actual_match']=lcp(b['suffix'],candidates[0]['suffix'][:b['common_candidate_prefix_tokens']]) if candidates else 0
        b['exact_body_other_scenario']=any(z['raw']==b['raw'] for z in candidates)
        b['same_shape_other_scenario']=any(z['shape']==b['shape'] for z in candidates)
        b['oracle_prefix_tokens']=max((lcp(b['suffix'],z['suffix']) for z in candidates),default=0)
        counts=collections.Counter(tuple(z['suffix']) for z in candidates)
        for k in [1,3,5]:b[f'top{k}_prefix_tokens']=max((lcp(b['suffix'],v) for v,_ in counts.most_common(k)),default=0)
    bins=[(0,32),(33,48),(49,64),(65,96),(97,100000)]
    def summary(bb,tt):
        total=sum(b['P'] for b in bb)
        return dict(tools=len(tt),boundaries=len(bb),P=stats([b['P'] for b in bb]),tool_formats=dict(collections.Counter(t['format'] for t in tt)),tool_types=dict(collections.Counter(t['value_type'] for t in tt)),embedded_error_fields=sum(t['error_field'] for t in tt),adapter_error_flags=sum(t['error_flag'] is not None for t in tt),raw_body_repetition=repetition([t['raw'] for t in tt]),canonical_body_repetition=repetition([t['canonical'] for t in tt]),body_shape_repetition=repetition([t['shape'] for t in tt]),boundary_body_repetition=repetition([b['raw'] for b in bb]),suffix_token_repetition=repetition([tuple(b['suffix']) for b in bb]),multi_tool_boundaries=sum(b['tool_count']>1 for b in bb),tool_count_histogram=dict(collections.Counter(b['tool_count'] for b in bb)),nonbody_tokens=stats([b['nonbody_tokens'] for b in bb]),leading_nonbody_tokens=stats([b['leading_nonbody_tokens'] for b in bb]),leading_nonbody_token_fraction=sum(b['leading_nonbody_tokens'] for b in bb)/total if total else 0,nonbody_token_fraction=sum(b['nonbody_tokens'] for b in bb)/total if total else 0,history_preserved=sum(b['history_preserved'] for b in bb),G_ms=stats([b['G_ms'] for b in bb]),other_scenario_same_tool_coverage=sum(bool(b['other_scenario_candidates']) for b in bb)/len(bb) if bb else 0,exact_body_other_scenario_fraction=sum(b['exact_body_other_scenario'] for b in bb)/len(bb) if bb else 0,same_shape_other_scenario_fraction=sum(b['same_shape_other_scenario'] for b in bb)/len(bb) if bb else 0,empirical_prefix={k:dict(tokens=stats([b[k] for b in bb]),token_fraction=sum(b[k] for b in bb)/total if total else 0) for k in ['oracle_prefix_tokens','top1_prefix_tokens','top3_prefix_tokens','top5_prefix_tokens','common_candidate_prefix_actual_match']})
    report=dict(source=str(a.input),scope='当前 BFCL 主机轨迹离线分析，不读取工具未来值制定策略，不修改手机实验',counts=dict(tasks=len(prompts),passed_tasks=len({b['task'] for b in boundaries if b['passed']}),base_scenarios=len(set(p['scenario'] for p in prompts))),definitions=dict(redundant_fraction='(N-unique)/N',repeated_observation_fraction='frequency>=2 的记录数/N',cross_scenario='排除相同数字场景 ID 的全部类别变体',candidate_pool='所有513边界，包含成功/失败任务；passed 为目标任务质量子集，候选库仍为所有其他场景',prefix='同工具序列、其他场景中历史观测 token 的最长共同前缀。oracle 看全候选；topK 用留一场景外频数排名。均非部署命中率，也非跨历史可直接拼接 KV。'),all=summary(boundaries,tools),passed=summary([b for b in boundaries if b['passed']],[t for t in tools if t['passed']]),failed=summary([b for b in boundaries if not b['passed']],[t for t in tools if not t['passed']]),bins=[])
    for lo,hi in bins:
        bb=[b for b in boundaries if lo<=b['P']<=hi];tt=[t for t in tools if t['P'] is not None and lo<=t['P']<=hi]
        report['bins'].append(dict(bin=f'{lo}-{hi if hi<100000 else "max"}',**summary(bb,tt),top_tools=collections.Counter(t['tool'] for t in tt).most_common(8),examples=[dict(task=b['task'],step=b['step'],P=b['P'],tools=b['tool_names'],content=json.loads(b['raw'])) for b in bb[:3]]))
    passed_pool=[b for b in boundaries if b['passed']]
    pp=dict(n=len(passed_pool),candidate_pool='only officially passed tasks; exclude same scenario variants',covered=0,exact=0,same_shape=0)
    for b in passed_pool:
        candidates=[z for z in passed_pool if z['scenario']!=b['scenario'] and z['tool_names']==b['tool_names']]
        pp['covered']+=bool(candidates);pp['exact']+=any(z['raw']==b['raw'] for z in candidates);pp['same_shape']+=any(z['shape']==b['shape'] for z in candidates)
    report['passed_only_sensitivity']=pp
    (a.output/'passed-only-sensitivity.json').write_text(json.dumps(pp,indent=2)+'\n')
    report['top_repeated_bodies']=[dict(content=raw,count=count,tools=sorted({t['tool'] for t in tools if t['raw']==raw}),scenarios=len({t['scenario'] for t in tools if t['raw']==raw})) for raw,count in collections.Counter(t['raw'] for t in tools).most_common(15)]
    report['per_tool']=[]
    for name,count in collections.Counter(t['tool'] for t in tools).most_common():
        tt=[t for t in tools if t['tool']==name]
        report['per_tool'].append(dict(tool=name,n=count,scenarios=len({t['scenario'] for t in tt}),P_body=stats([t['P_body'] for t in tt]),raw=repetition([t['raw'] for t in tt]),shapes=repetition([t['shape'] for t in tt]),formats=dict(collections.Counter(t['format'] for t in tt)),examples=list(dict.fromkeys(t['raw'] for t in tt))[:3]))
    report['initial_prompt_other_scenario_best_lcp']=stats([max((lcp(p['ids'],q['ids']) for q in prompts if p['scenario']!=q['scenario']),default=0) for p in prompts])
    (a.output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('boundaries',boundaries),('tools',tools)]:
        keys=[k for k in rows[0] if k!='suffix']
        with (a.output/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:dump(v) if isinstance(v,(tuple,list)) else v for k,v in r.items() if k in keys} for r in rows)
    (a.output/'input-manifest.json').write_text(json.dumps(dict(inputs=files,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    if a.toolsandbox_cpu:
        analyze_toolsandbox(a.toolsandbox_cpu,a.output)
    print(json.dumps({k:report[k] for k in ['counts','all','passed']},ensure_ascii=False,indent=2))

def analyze_toolsandbox(base,output):
    rows=[];cases=[]
    for f in sorted(base.glob('*/metrics.json')):
        d=json.loads(f.read_text());r=json.loads((f.parent/'result.json').read_text());cases.append(r)
        for t in d['tools']:
            v,fmt=parse(t['result']);rows.append(dict(task=f.parent.name,tool=t['tool'],P=t['P_append'],P_body=t['P_body'],format=fmt,type=type(v).__name__,raw=t['result'],shape=dump(shape(v)),trace_complete=r['trace_complete'],full_score=r['full_score']))
    out=dict(source=str(base),condition='first round CPU only; do not pool rounds/backends',tasks=len(cases),trace_complete=sum(x['trace_complete'] for x in cases),full_score=sum(x['full_score'] for x in cases),tools=len(rows),formats=dict(collections.Counter(x['format'] for x in rows)),types=dict(collections.Counter(x['type'] for x in rows)),P=stats([x['P'] for x in rows]),body_repetition=repetition([x['raw'] for x in rows]),shape_repetition=repetition([x['shape'] for x in rows]),examples=rows[:8])
    (output/'toolsandbox-format.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
