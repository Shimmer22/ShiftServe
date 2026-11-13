#!/usr/bin/env python3
"""Termux SSH持锁串行比较vendor搜索路径；不加载模型或更改设备设置。"""
import fcntl,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
out=ROOT/'results/20261113-htp-session-probe';out.mkdir(exist_ok=True)
lock=(out/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
remote=t.remote_root+'/'+out.name;rows=[]
try:
    t.prepare_ssh(out);t.call('shell','mkdir','-p',remote)
    t.call('push',str(out/'build/probe'),remote+'/probe')
    t.call('push',str(ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'),remote+'/libggml.so')
    t.call('shell','chmod','755',remote+'/probe')
    for vendor in [False,True]:
        for sessions in [1,2]:
            label=('vendor' if vendor else 'clean')+f'-sessions{sessions}'
            libs=remote+':'+t.config['runtime_library_dir']+(':/vendor/lib64:/vendor/lib64/egl' if vendor else '')
            env=dict(LD_LIBRARY_PATH=libs,ADSP_LIBRARY_PATH=t.config['runtime_library_dir'],GGML_HEXAGON_NDEV=str(sessions),GGML_HEXAGON_OPQUEUE='16',GGML_HEXAGON_OPBATCH='1024',GGML_HEXAGON_PROFILE='0',GGML_HEXAGON_EXPERIMENTAL='1',GGML_HEXAGON_USE_HMX='1')
            command=shlex.join(['env',*[k+'='+v for k,v in env.items()],remote+'/probe'])
            result=subprocess.run(t.shell_command(command),capture_output=True,timeout=90)
            (out/(label+'.stdout.txt')).write_bytes(result.stdout);(out/(label+'.stderr.txt')).write_bytes(result.stderr)
            row=dict(label=label,returncode=result.returncode,env=env,model_loaded=False)
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
finally:
    if t.lock_owner:
        t.release_lock();subprocess.run(t.prefix+['termux-wake-unlock'],capture_output=True,timeout=30,check=True)
        (out/'cleanup.json').write_text(json.dumps(dict(lock_released=True,wake_lock_released=True))+'\n')
