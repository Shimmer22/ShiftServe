#!/usr/bin/env python3
"""检查已有 BFCL/ToolSandbox 是否存在 PromptWeaver 式长动态静态片段。"""
import argparse,collections,json,statistics
from pathlib import Path


def stat(rows,key):
    v=sorted(r[key] for r in rows)
    return dict(n=len(v),sum=sum(v),median=statistics.median(v),min=min(v),max=max(v)) if v else dict(n=0)


def lcp(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bfcl',type=Path,required=True);p.add_argument('--toolsandbox',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    groups=collections.defaultdict(list);fuel=[];fuel_ids=[];held_examples=[]
    for f in sorted(a.bfcl.glob('multi_turn_*/events.jsonl')):
        e=json.loads((f.parent/'entry.json').read_text());tools=json.loads((f.parent/'tools.json').read_text());post={t.get('next_step') for t in tools};held=e.get('held_docs',{})
        events=[json.loads(line) for line in f.read_text().splitlines()]
        for index,r in enumerate(events):
            key=r['step'];g='post_tool' if key in post else 'initial' if key=='t00-s00' else 'held_docs_added' if key.endswith('s00') and str(r['turn']) in held else 'other'
            row=dict(task=f.parent.name,step=key,H=r['H_common_prefix'],P=r['P_append'],prompt=r['prompt_tokens'],D=len(r['generated_ids']))
            if index:
                previous=events[index-1]
                old=json.loads((f.parent/(previous['step']+'.input.json')).read_text())['input_ids']
                old_full=old+previous['generated_ids']
                current=json.loads((f.parent/(key+'.input.json')).read_text())['input_ids']
                assert lcp(old_full,current)==row['H']
                row.update(old_prompt=len(old),old_full=len(old_full),cut_within_old_prompt=row['H']<len(old),old_tokens_past_cut=len(old_full)-row['H'])
            groups[g].append(row)
        for t in tools:
            if not t.get('next_step'):continue
            try:v=json.loads(t['content'])
            except ValueError:continue
            if isinstance(v,dict) and list(v)==['fuelLevel']:
                d=json.loads((f.parent/(t['next_step']+'.input.json')).read_text());ids=d['input_ids'][t['H_common_prefix']:]
                fuel_ids.append(ids);fuel.append(dict(task=f.parent.name,tool=t['call']['name'],P=t['P_append'],P_body=t['result_tokens'],content=t['content']))
        for turn,docs in held.items():
            prompt=f.parent/f't{int(turn):02d}-s00.prompt.txt'
            if prompt.exists():held_examples.append(dict(task=f.parent.name,turn=turn,new_tools=[d['name'] for d in docs],name_occurrences={d['name']:prompt.read_text().count(d['name']) for d in docs}))
    toolsets=collections.Counter();changes=[];pairs=0
    for task in sorted(a.toolsandbox.iterdir()):
        if not task.is_dir():continue
        previous=None
        for f in sorted(task.glob('[0-9][0-9][0-9].request.json')):
            tools=json.loads(f.read_text())['tools'];signature=json.dumps(tools,sort_keys=True);names=tuple(sorted(t.get('function',t).get('name','') for t in tools))
            if previous is None:toolsets[names]+=1
            else:
                pairs+=1
                if signature!=previous:changes.append(dict(task=task.name,step=f.stem,tools=names))
            previous=signature
    report=dict(sources=dict(bfcl=str(a.bfcl),toolsandbox=str(a.toolsandbox),paper='https://arxiv.org/html/2605.10380v1'),bfcl_regions={k:dict(P=stat(v,'P'),H=stat(v,'H'),cut_within_old_prompt=sum(r.get('cut_within_old_prompt',False) for r in v)) for k,v in groups.items()},fuel=dict(n=len(fuel),P=stat(fuel,'P'),P_body=stat(fuel,'P_body'),common_prefix_tokens=min(lcp(fuel_ids[0],ids) for ids in fuel_ids),generic_wrapper_tokens=6,condition='only observed successful fuelLevel-shaped outputs, not guaranteed before actual return',examples=fuel),held_docs_examples=held_examples,toolsandbox=dict(tasks=sum(toolsets.values()),unique_tool_sets=len(toolsets),adjacent_requests=pairs,tool_description_changes=len(changes),changed_examples=changes,common_tool_sets=[dict(tools=k,tasks=v) for k,v in toolsets.most_common(10)]))
    (a.output/'promptweaver-fit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (a.output/'prompt-regions.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['held_docs_examples','fuel']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
