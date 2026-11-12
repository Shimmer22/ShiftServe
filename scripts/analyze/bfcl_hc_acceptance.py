"""补测验收：精确请求、轮换顺序、强制HC、计时守恒与缓存来源。"""
import json,collections,datetime,csv
from pathlib import Path
root=Path('results/20261110-bfcl-small-p-hc-supplement');old=Path('results/20261109-bfcl-cached-repeats')
requests=json.loads((root/'requests.json').read_text());counts=collections.Counter();paired=[];stages=collections.defaultdict(float);cache=collections.Counter()
order=sorted(range(len(requests)),key=lambda i:(requests[i]['task'],requests[i]['step']))
files=sorted((root/'measurements').glob('r*-b*.json'));assert len(files)==45
for f in files:
 d=json.loads(f.read_text());q=d['request'];rep=int(f.name[1:3]);i=int(f.stem.split('-b')[1]);base=requests[i]
 assert all(q[k]==v for k,v in base.items());j=order.index(i);conds=['cpu','htp','route'];shift=(j+rep)%3
 assert q['condition_order']==conds[shift:]+conds[:shift]
 orig=json.loads((old/'measurements'/('r00-'+base['original_boundary']+'.json')).read_text())['request']
 assert all(q[k]==orig[k] for k in ('H','P','D','prompt_ids','generated_ids','task','step'))
 a=d['response'];assert a['cache']['prefix_tokens']==q['H'];cache[a['cache']['source']]+=1
 assert [x['condition'] for x in a['results']]==q['condition_order'];rr={x['condition']:x for x in a['results']}
 for c,x in rr.items():
  assert all(x[k]==q[k] for k in ('H','P','D'));assert all(x[k]>=0 for k in ('prefill_ms','handoff_ms','decode_ms','total_ms'))
  assert abs(x['total_ms']-sum(x[k] for k in ('prefill_ms','handoff_ms','decode_ms')))<10
  stages[c]+=x['total_ms']/1000
 assert rr['route']['route']=='htp' and rr['route']['handoff_ms']>0
 cc=rr['cpu']['prefill_ms']/1000;hc=(rr['route']['prefill_ms']+rr['route']['handoff_ms'])/1000
 paired.append(dict(round=rep,boundary=base['original_boundary'],P=q['P'],H=q['H'],D=q['D'],cc_s=cc,hc_s=hc,hc_saving=1-hc/cc,cc_total_s=rr['cpu']['total_ms']/1000,hc_total_s=rr['route']['total_ms']/1000));counts[q['P']]+=1
with (root/'paired-paths.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
by={}
for p in sorted(counts):
 group=[x for x in paired if x['P']==p];cc=sum(x['cc_s'] for x in group);hc=sum(x['hc_s'] for x in group)
 by[p]=dict(n=len(group),hc_wins=sum(x['hc_s']<x['cc_s'] for x in group),cc_s=cc,hc_s=hc,saving=1-hc/cc)
result=dict(accepted=True,completed_at=datetime.datetime.fromtimestamp(json.loads((root/'status.json').read_text())['updated_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),boundaries=15,rounds=3,condition_evaluations=135,exact_tokens_and_HPD=True,order_rotation=True,all_forced_hc=True,cache_sources=cache,hc_prefill_handoff_wins=sum(x['hc_s']<x['cc_s'] for x in paired),paired_n=45,cc_prefill_s=sum(x['cc_s'] for x in paired),hc_prefill_handoff_s=sum(x['hc_s'] for x in paired),total_s=stages,by_P=by,scope='teacher forced; preparation and fixture restore excluded; H varies; not live quality or controlled P crossover')
(root/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
