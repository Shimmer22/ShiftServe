#!/usr/bin/env python3
"""主图任务的只读进度与运行时间预测；预测仅用于回访，不进入性能图或Oracle。"""
import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import statistics as st
import sys
import time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.validation.observation import measurement_grid
TZ=ZoneInfo('Asia/Shanghai')


def read_rows(path):
    p=path/'checked-rows.jsonl'
    if not p.exists():return []
    lines=p.read_text().splitlines();rows=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    return rows


def snapshot(campaign,seed):
    state=json.loads((campaign/'campaign.json').read_text())
    paths=[Path(p) for p in state.get('runs',[])]
    for h in [2048,4096]:
        log=campaign/f'h{h}-campaign-console.log'
        if log.exists() and log.read_text().splitlines():
            p=Path(log.read_text().splitlines()[0])
            if p.is_dir() and p not in paths:paths.append(p)
    samples=read_rows(seed);done=[];events=[]
    for p in paths:
        done.extend(read_rows(p))
        ep=p/'native-events.jsonl'
        if ep.exists():events.extend(json.loads(x) for x in ep.read_text().splitlines())
    samples+=done
    successful=[r for r in samples if r.get('passed')]
    def parts(h,p,route):
        pool=[r for r in successful if r['route']==route]
        nearest_h=min({r['H'] for r in pool},key=lambda x:abs(math.log(h/x)))
        pool=[r for r in pool if r['H']==nearest_h]
        # HTP 的 P<32 与 >=32 分开估计，避免用核阶跃两侧的单位成本互相外推。
        same_kernel=[r for r in pool if (r['P']<32)==(p<32)]
        if same_kernel:pool=same_kernel
        anchor=min({r['P'] for r in pool},key=lambda x:abs(math.log(p/x)))
        group=[r for r in pool if r['P']==anchor]
        med=lambda key:st.median(r[key] for r in group)/1000
        hscale=1+.3*(h/nearest_h-1)
        work=med('prefill_ms')*(p/anchor)*hscale+med('decode_ms')*(1+.1*(h/nearest_h-1))+med('handoff_checked_ms')*(h+p)/(nearest_h+anchor)
        prep=med('prepare_ms')*h/nearest_h
        return work,prep
    plans=[];initialization=0
    for h in [2048,4096]:
        c=json.loads((ROOT/f'configs/observation-main-h{h}.json').read_text())
        keys=[('calibration',p,0,'after_return',r,rep) for p in c['new_tokens'] for rep in range(3) for r in ['CC','HC']]
        keys += [('measurement',p,g,s,r,rep) for rep in range(c['repetitions']) for p,g,s,r in measurement_grid(c)]
        observed={(r['phase'],r['P'],r['gap_requested_ms'],r['strategy'],r['requested_route'],r['repetition']) for r in done if r['H']==h}
        for key in keys:
            if key in observed:continue
            phase,p,g,s,r,rep=key
            if r=='router':
                cc,_=parts(h,p,'CC');hc,prep=parts(h,p,'HC');work=(cc+hc+prep)/2
            else:
                work,prep=parts(h,p,r);work+=prep
            plans.append(work+g/1000+.12)
        if not any(r['H']==h for r in done):
            seed_events=[json.loads(x) for x in (seed/'native-events.jsonl').read_text().splitlines()]
            ready=next(e for e in seed_events if e['event']=='session_ready')
            initialization+=(ready['model_context_init_ms']+ready['history_build_ms']*(h/2048)**1.3)/1000
    remaining=math.fsum(plans)
    remaining+=initialization
    now=datetime.now(TZ)
    complete=state.get('status') in ['figures_rendered','failed','blocked_device']
    return dict(updated=now.isoformat(),status=state.get('status'),device='Xiaomi 14 Pro / SM8650',
        completed_trials=len(done),planned_trials=762,calibration_completed=sum(r['phase']=='calibration' for r in done),
        measurements_completed=sum(r['phase']=='measurement' for r in done),remaining_trials=len(plans),
        estimate_remaining_minutes=round(remaining/60,1),
        estimated_finish_window=[(now+timedelta(seconds=remaining*f)).isoformat() for f in [.7,1.6]],
        runs=list(map(str,paths)),
        limitations='仅运行排期预测；未测形状按同kernel最近P线性外推，H按保守系数外推，未来Router耗时按两路径均值估计；不进入主图/Oracle。',
        terminal=complete)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--seed',type=Path,required=True);ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    history=a.campaign/'progress-history.jsonl'
    anchor=datetime.fromisoformat(json.loads(history.read_text().splitlines()[0])['updated']) if history.exists() else datetime.now(TZ)
    next_report=anchor+timedelta(minutes=10);first=True
    while True:
        d=snapshot(a.campaign,a.seed)
        now=datetime.now(TZ);report=first or now>=next_report or d['terminal'] or not a.watch
        while next_report<=now:next_report+=timedelta(minutes=10)
        d['next_check_at']=None if d['terminal'] else next_report.isoformat()
        (a.campaign/'progress.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        with (a.campaign/'progress-history.jsonl').open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
        if report:print(json.dumps(d,ensure_ascii=False),flush=True)
        first=False
        if not a.watch or d['terminal']:break
        time.sleep(60)


if __name__=='__main__':main()
