#!/usr/bin/env python3
"""冻结独立H/P网格校准，拆分CPU/NPU prefill与双向迁移；不使用正式route未来成本。"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.workloads.toolsandbox import post,save
from a2o.runtime.phone_preflight import require_http_phone_preflight
from a2o.runtime.network_retry import retry_network
from a2o.scheduling.linear_route import fit_model,threshold


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='http://10.29.131.50:18780')
    p.add_argument('--source',type=Path,default=ROOT/'results/20261115-qwen3-4b-joint/calib-source.json')
    p.add_argument('--hs',default='128,768,1536')
    p.add_argument('--ps',default='1,8,32,128,384')
    p.add_argument('--repetitions',type=int,default=3)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    require_http_phone_preflight(ROOT, args.endpoint, args.output)
    source=json.loads(args.source.read_text());tokens=source['prefix_ids']+source['input_ids']
    hs=[int(x) for x in args.hs.split(',') if x.strip()]
    ps=[int(x) for x in args.ps.split(',') if x.strip()]
    grid=[(h,p) for h in hs for p in ps];random.Random(20261107).shuffle(grid)
    plan=dict(H=hs,P=ps,repetitions=args.repetitions,decode_evaluations=1,order=grid,
              source=str(args.source),seed=20261107,
              timing='prefill and migration only; decode excluded from routing',
              quality='greedy/logit differences reported separately; no performance-row quality filtering')
    save(args.output/'plan.json',plan)
    responses=[]
    for h,p in grid:
        path=args.output/f'H{h}-P{p}.response.json'
        if path.exists():response=json.loads(path.read_text())
        else:
            seq=(tokens*((h+p)//len(tokens)+1))[:h+p]
            body=dict(prefix_ids=seq[:h],input_ids=seq[h:],decode_evaluations=1,repetitions=args.repetitions)
            save(args.output/f'H{h}-P{p}.request.json',body)
            save(args.output/'progress.json',dict(phase='calibrating',H=h,P=p,completed=len(responses),total=len(grid),updated_unix=time.time()))
            # 请求结果丢失时可重复：此入口重建独立上下文，不影响正式会话。
            response=retry_network(lambda:post(args.endpoint,'/a2o/calibrate',body,1800))
            save(path,response)
        responses.append(response)
        print(json.dumps(dict(H=h,P=p,completed=len(responses),total=len(grid))),flush=True)
    model=fit_model(responses)
    artifact=dict(model=model,thresholds=[threshold(model,h) for h in (128,512,1024,1536,2048,3072)],
                  quality=[dict(H=r['H'],P=r['P'],all_passed=all(x['passed'] for x in r['rows'])) for r in responses])
    save(args.output/'cost-model.json',artifact)
    save(args.output/'completion.json',dict(completed=True,points=len(responses),model_id=model['model_id']))
    save(args.output/'progress.json',dict(phase='complete',completed=len(responses),total=len(grid),updated_unix=time.time()))
    print(json.dumps(artifact,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
