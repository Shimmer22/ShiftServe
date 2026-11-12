#!/usr/bin/env python3
"""按实际返回边界统计原生模板 BFCL；任务通过/失败分别汇总。"""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

def stats(values):
    x=sorted(values)
    if not x: return {'n':0}
    return {'n':len(x),'min':x[0],'p25':x[int((len(x)-1)*.25)],'median':statistics.median(x),
            'p90':x[int((len(x)-1)*.9)],'max':x[-1],'fraction_le32':sum(v<=32 for v in x)/len(x)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);a=parser.parse_args()
    results=[]; rows=[]; transitions={}
    for f in sorted(a.directory.glob('multi_turn_*/result.json')):
        r=json.loads(f.read_text());results.append(r)
        entry=json.loads((f.parent/'entry.json').read_text())
        category=r['id'].rsplit('_',1)[0]
        for t in json.loads((f.parent/'tools.json').read_text()):
            row=dict(task=r['id'],category=category,classes=','.join(entry['involved_classes']),
                     passed=bool(r['quality'].get('valid')),task_error=r['error'] is not None,
                     tool=t['call']['name'],tool_error=t['error'],step=t['id'],next_step=t.get('next_step'),
                     H=t.get('H_common_prefix'),P=t.get('P_append'),D=t.get('D_next'),
                     P_body=t['result_tokens'],G_ms=t.get('G_ms'))
            rows.append(row)
            if row['P'] is not None: transitions[(r['id'],t['next_step'])]=row
    with (a.directory/'tool-transitions.csv').open('w') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bins=[(0,16),(17,24),(25,32),(33,48),(49,64),(65,96),(97,128),(129,256),(257,32768)]
    groups=[]
    for c in sorted({r['category'] for r in rows}):
        tasks=[r for r in results if r['id'].rsplit('_',1)[0]==c]
        for quality in ['all','passed','failed']:
            g=[r for r in transitions.values() if r['category']==c and (quality=='all' or r['passed']==(quality=='passed'))]
            groups.append(dict(category=c,quality=quality,P=stats([r['P'] for r in g]),
                H=stats([r['H'] for r in g]),D=stats([r['D'] for r in g if r['D'] is not None]),
                P_body=stats([r['P_body'] for r in rows if r['category']==c and (quality=='all' or r['passed']==(quality=='passed'))]),
                bins={f'{lo}-{hi}':sum(lo<=r['P']<=hi for r in g) for lo,hi in bins}))
    summary=dict(tasks=len(results),passed=sum(bool(r['quality'].get('valid')) for r in results),
        task_errors=sum(bool(r['error']) for r in results),tools=len(rows),distinct_return_transitions=len(transitions),
        all_P=stats([r['P'] for r in transitions.values()]),
        passed_P=stats([r['P'] for r in transitions.values() if r['passed']]),
        categories={c:dict(tasks=sum(r['id'].rsplit('_',1)[0]==c for r in results),passed=sum(r['id'].rsplit('_',1)[0]==c and bool(r['quality'].get('valid')) for r in results)) for c in sorted({r['category'] for r in rows})},
        groups=groups,failures=[dict(id=r['id'],error=r['error'],quality=r['quality']) for r in results if not r['quality'].get('valid')])
    candidates=[]
    for r in results:
        if not r['quality'].get('valid'): continue
        g=[x for x in transitions.values() if x['task']==r['id']]
        candidates.append(dict(id=r['id'],classes=next((x['classes'] for x in rows if x['task']==r['id']),''),
            transitions=len(g),P_values=sorted({x['P'] for x in g}),
            H_min=min((x['H'] for x in g),default=None),H_max=max((x['H'] for x in g),default=None),
            le32=sum(x['P']<=32 for x in g),
            within_4k=sum(x['H']+x['P']+1024<=4096 for x in g),
            within_8k=sum(x['H']+x['P']+1024<=8192 for x in g)))
    (a.directory/'passed-candidates.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n')
    (a.directory/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','failures')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
