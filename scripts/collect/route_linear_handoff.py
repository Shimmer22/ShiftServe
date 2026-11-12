#!/usr/bin/env python3
"""等待既有CPU结束，持锁串行校准线性route，再接续未运行条件和重复轮次。"""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.network_retry import retry_network
from a2o.workloads.toolsandbox import save
from toolsandbox_full_background import remote_collector
BASE=ROOT/'results/20261107-toolsandbox-full'
OUT=BASE/'linear-routing'
BINARY=ROOT/'results/20261107-toolsandbox-route-e2e/build-service-212553/agent'
REGISTRY=ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'

def run(command,name):
    with (OUT/(name+'.log')).open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def status(phase,**fields):
    save(OUT/'progress.json',dict(phase=phase,updated_unix=time.time(),**fields))

def phone(action,condition,service):
    cmd=[sys.executable,'scripts/run/route_phone.py',action,'--retry-network','--condition',condition,'--output',str(service)]
    if action=='start':cmd+=['--binary',str(BINARY),'--registry',str(REGISTRY)]
    run(cmd,action+'-'+condition+'-'+service.name)

def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        while True:
            current=retry_network(lambda:remote_collector('20261107-full-cpu'))
            status('waiting_for_existing_cpu',collector=current)
            if current['phase']=='complete':break
            time.sleep(30)
        status('download_cpu')
        retry_network(lambda:subprocess.run(['rsync','-a','--timeout=30','-e','ssh -o ConnectTimeout=10','wins:/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261107-full-cpu/',str(BASE/'cpu')+'/'],check=True))
        if not (OUT/'cpu-stopped.json').exists():
            phone('stop','cpu',ROOT/'results/20261107-toolsandbox-route-e2e/service-full-cpu')
            save(OUT/'cpu-stopped.json',dict(stopped=True))
            save(BASE/'cpu-stage.json',dict(started=True,collecting=True,stopped=True,policy='fixed_cpu_original_binary'))
        calibration=OUT/'calibration'
        service=OUT/'service-linear-calibration'
        if not (calibration/'completion.json').exists():
            status('starting_calibration')
            phone('start','route',service)
            status('calibrating')
            run([sys.executable,'-u','scripts/collect/route_linear_calibration.py','--output',str(calibration)],'calibration')
        if not (OUT/'calibration-stopped.json').exists():
            phone('stop','route',service);save(OUT/'calibration-stopped.json',dict(stopped=True))
        status('starting_linear_campaign')
        run([sys.executable,'-u','scripts/collect/toolsandbox_full_background.py','--rounds','3',
             '--binary',str(BINARY),'--cost-model',str(calibration/'cost-model.json'),'--config-tag=-linear'], 'campaign')
        status('complete')
    except BaseException as exc:
        status('failed',error=repr(exc));raise

if __name__=='__main__':main()
