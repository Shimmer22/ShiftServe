#!/usr/bin/env python3
"""等待私有探针诊断结束，再串行运行原版库十轮边界扫描。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.pd_sweep import save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True);a=p.parse_args()
out=ROOT/'results/20261111-htp-platform-exploration';lock=(out/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
baseline=ROOT/'results/20261111-htp-platform-p60-72-cpu6-t10'
def status(phase,**kw):save(out/'status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**kw))
def analyze():
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze/htp_platform_exploration.py'),'--diagnostic',str(a.diagnostic),'--baseline',str(baseline),'--output',str(out)],cwd=ROOT,check=True)
try:
    while True:
        if not (a.diagnostic/'status.json').exists():
            status('diagnostic_starting');time.sleep(2);continue
        s=json.loads((a.diagnostic/'status.json').read_text())
        if s['phase']=='failed':raise RuntimeError('诊断失败，保留现场：'+str(s))
        if s['phase']=='complete':break
        status('diagnostic_running',diagnostic=s);time.sleep(15)
    analyze()
    command=[sys.executable,str(ROOT/'scripts/collect/pd_stage_sweep.py'),'--config',str(ROOT/'configs/htp-platform-p60-72-cpu6-t10.json'),'--binary',str(ROOT/'results/20261111-pd-prefill-dense-h4096-cpu6-t10/build/agent')]
    with (out/'baseline.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            s=json.loads((baseline/'status.json').read_text()) if (baseline/'status.json').exists() else {}
            status('baseline_running',baseline=s);time.sleep(15)
        assert proc.returncode==0,'原版库扫描失败，查看baseline.log'
    analyze();status('complete',paired_requests=130)
except Exception:
    status('failed',error=traceback.format_exc());raise
