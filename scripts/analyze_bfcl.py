#!/usr/bin/env python3
"""仅汇总已有 BFCL 轨迹；不访问模型或执行动作。"""
import argparse
import collections
import csv
import json
from pathlib import Path


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'n':0}
    def q(p):
        i=(len(values)-1)*p; a=int(i); b=min(a+1,len(values)-1)
        return values[a]+(values[b]-values[a])*(i-a)
    return {'n':len(values), 'min':values[0], 'p50':q(.5), 'p90':q(.9),
            'p95':q(.95), 'p99':q(.99), 'max':values[-1]}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('result',type=Path); args=parser.parse_args()
    rows=[]; tasks=[]
    for file in sorted(args.result.glob('*/*/events.jsonl')):
        resultfile=file.parent/'result.json'
        result=json.loads(resultfile.read_text()) if resultfile.exists() else {}
        events=[json.loads(s) for s in file.read_text().splitlines()]
        links={}; decodes={}
        for event in events:
            if event['kind']=='tool_next_prompt': links[event['tool_event_id']]=event
            if event['kind']=='tool_next_decode': decodes[event['tool_event_id']]=event
        tasks.append({'id':file.parent.name,'phase':file.parent.parent.name,
                      'complete':result.get('trace_complete',False),
                      'quality_pass':result.get('quality',{}).get('valid',False)})
        for event in events:
            if event['kind']!='tool': continue
            link=links.get(event['tool_event_id'],{}); decode=decodes.get(event['tool_event_id'],{})
            rows.append({'task_id':event['task_id'],'phase':event['phase'],
                         'category':event['task_id'].rsplit('_',1)[0],
                         'tool':event['call']['name'],'tool_type':event['tool_type'],
                         'gap_ms':event['gap_ms'],'execution_ms':event['execution_ms'],
                         'dispatch_ms':(event['execution_start_ns']-event['issued_ns'])/1e6,
                         'delivery_ms':(event['result_available_ns']-event['execution_complete_ns'])/1e6,
                         'result_tokens':event['result_tokens'],'result_bytes':event['result_bytes'],
                         'argument_bytes':event['argument_bytes'],
                         'H':link.get('H_common_prefix'),'P':link.get('P_append'),
                         'D':decode.get('D_generated'),'execution_success':event['execution_success'],
                         'trace_complete':result.get('trace_complete',False),
                         'quality_pass':result.get('quality',{}).get('valid',False)})
    if rows:
        with (args.result/'tool-transitions.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups=collections.defaultdict(list)
    for row in rows:
        groups[(row['phase'],row['category'],row['tool_type'],row['quality_pass'])].append(row)
    metrics=['gap_ms','execution_ms','dispatch_ms','delivery_ms','H','P','D','result_tokens']
    summaries=[{'phase':key[0],'category':key[1],'tool_type':key[2], 'quality_pass':key[3],
                **{m:stats([r[m] for r in group]) for m in metrics}}
               for key,group in sorted(groups.items())]
    output={'scope':'remote BFCL in-memory sandbox; no phone latency/energy or migration claims',
            'tasks':tasks,'tool_events':len(rows),'groups':summaries}
    (args.result/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'tasks':len(tasks),'tool_events':len(rows),'groups':len(groups)}))


if __name__=='__main__': main()
