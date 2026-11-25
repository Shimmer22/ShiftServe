#!/usr/bin/env python3
"""按独立 CPU 轨迹的真实锚点 H/P 校准；不外推，不用正式 route 的实测未来成本。"""
import argparse,json,sys
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.workloads.toolsandbox import post,save
from a2o.runtime.phone_preflight import require_http_phone_preflight
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--endpoint',default='http://10.29.131.50:18780');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
require_http_phone_preflight(ROOT, a.endpoint,a.output)
summary=json.loads((a.cpu_run/'summary.json').read_text());rows=[];seen=set();attempted=0
for case in summary['results']:
 folder=a.cpu_run/case['name'];metrics=json.loads((folder/'metrics.json').read_text())
 for step in metrics['inference']:
  if step['step']==0:continue
  i=step['step'];ids=json.loads((folder/f'{i:03d}.prompt-ids.json').read_text());h=step['backend_timings']['H_anchor'];pp=step['backend_timings']['P_evaluated']
  if h==0:continue
  key=(h,pp)
  if key in seen:continue
  seen.add(key);body=dict(prefix_ids=ids[:h],input_ids=ids[h:],decode_evaluations=4)
  save(a.output/f'H{h}-P{pp}.request.json',body)
  result=post(a.endpoint,'/a2o/calibrate',body,1800);save(a.output/f'H{h}-P{pp}.response.json',result);attempted+=1
  passed=all(r['passed'] for r in result['rows'])
  if passed:
   rows.append(dict(h_min=h,h_max=h,p_min=pp,p_max=pp,cc_ms=median(r['cc_ms'] for r in result['rows']),hc_ms=median(r['hc_ms'] for r in result['rows']),samples=3,decode_evaluations=4,source=f'H{h}-P{pp}.response.json'))
  save(a.output/'costs.json',dict(rows=rows,covered=len(rows),attempted=len(seen),scope='exact H/P; D=4 compute horizon; full responses evaluated separately'))
  print(json.dumps(dict(H=h,P=pp,passed=passed,covered=len(rows))),flush=True)
post(a.endpoint,'/a2o/costs',dict(rows=rows));save(a.output/'completion.json',dict(completed=True,covered=len(rows),attempted=len(seen)))
