#!/usr/bin/env python3
"""持宿主机锁串行执行两组OpenCL扫描；设备锁由oracle入口持有。"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'results/.termux-model.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'running','runs':[]}
    def save():
        state['updated']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        (out/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for h in (2048,4096):
            state['active_H']=h;save()
            with (out/f'h{h}-console.log').open('w') as log:
                proc=subprocess.run([sys.executable,str(ROOT/'scripts/profile/oracle.py'),'--config',str(ROOT/f'configs/opencl-boundary-h{h}-termux.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            lines=(out/f'h{h}-console.log').read_text().splitlines()
            run=Path(next(x for x in lines if x.startswith(str(ROOT/'results/'))))
            state['runs'].append(str(run));save()
            if proc.returncode:raise RuntimeError(f'H{h} runner failed: {proc.returncode}')
            if not json.loads((run/'completion.json').read_text())['all_HC_candidates_numerically_qualified']:raise RuntimeError('OpenCL numeric admission failed')
        state['status']='measured';state['active_H']=None;save()
        command=[str(Path.home()/'miniforge3/envs/visiontrim-qwen25vl/bin/python'),str(ROOT/'scripts/plot/opencl_boundary.py'),
            '--original',str(ROOT/'results/20261105-222416-140819-http-oracle-grid'),str(ROOT/'results/20261105-225513-856128-http-oracle-grid'),
            '--opencl',*state['runs'],'--output',str(out/'figures')]
        with (out/'plot-console.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['status']='plotted';save()
    except BaseException as exc:
        state['status']='failed';state['error']=repr(exc);save();raise
    finally:
        t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
        # runner有完成记录才说明它已结束并释放设备锁。
        if state['runs'] and all((Path(r)/'completion.json').exists() for r in state['runs']):
            p=subprocess.run(t.prefix+['termux-wake-unlock'],capture_output=True,timeout=30)
            (out/'wake-unlock.log').write_bytes(p.stdout+p.stderr);state['wake_unlock_returncode']=p.returncode;save()

if __name__=='__main__':main()
