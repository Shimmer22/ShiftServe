#!/usr/bin/env python3
"""串行线程扫描，独立选定一个全局线程数，再补P<=64曲线。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.cpu_threads import load_thread_run,select_threads


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume-first',type=Path);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    state=dict(stage='thread_sweep',sweep_runs=[],curve_runs=[],device=a.device)
    def save():
        state['updated']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();(out/'campaign.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    def run(config,label):
        log=out/(label+'.log')
        with log.open('w') as f:code=subprocess.run([sys.executable,str(ROOT/'scripts/profile/oracle.py'),'--device',a.device,'--config',str(config)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError('实验失败，见'+str(log))
        p=Path(log.read_text().splitlines()[0]);load_thread_run(p);return str(p)
    try:
        if a.resume_first:
            p=a.resume_first.resolve();save()
            while not ((p/'completion.json').exists() and (p/'device/trials.jsonl').exists()):
                if (p/'failure.json').exists():raise RuntimeError('首个线程扫描失败')
                time.sleep(2)
            load_thread_run(p);state['sweep_runs'].append(str(p))
        else:state['sweep_runs'].append(run(ROOT/'configs/cpu-thread-sweep-h2048.json','sweep-h2048'))
        save();state['sweep_runs'].append(run(ROOT/'configs/cpu-thread-sweep-h4096.json','sweep-h4096'));save()
        rows=[]
        for p in state['sweep_runs']:rows+=load_thread_run(p)[1]
        selection=select_threads(rows,list(range(1,9)));(out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n');state.update(stage='best_thread_curve',best_threads=selection['best_threads']);save()
        for h in [2048,4096]:
            c=json.loads((ROOT/f'configs/cpu-thread-sweep-h{h}.json').read_text());c.update(new_tokens=[16,24,31,32,33,40,48,56,64],cpu_thread_scan=sorted({4,selection['best_threads']}))
            cfg=out/f'curve-h{h}-config.json';cfg.write_text(json.dumps(c,indent=2)+'\n')
            state['curve_runs'].append(run(cfg,f'curve-h{h}'));save()
        state['stage']='plotting';save()
        with (out/'plot.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts/plot/cpu_threads.py'),'--campaign',str(out)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['stage']='figures_rendered';save()
    except BaseException as e:
        state.update(stage='failed',error=repr(e));save();raise


if __name__=='__main__':main()
