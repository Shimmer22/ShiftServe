#!/usr/bin/env python3
"""Termux持锁串行运行原版计时与私有DSP路径验证，保留负面数值结果。"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
out=ROOT/'results/20261111-pad31-matmul';out.mkdir(exist_ok=True)
t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json');q=shlex.quote;remote=t.remote_root+'/'+out.name
binary=out/'build/probe';probe_info=json.loads((ROOT/'results/20261111-htp-platform-exploration/probe-build-v75/completion.json').read_text())
def status(phase,**kw):(out/'status.json').write_text(json.dumps(dict(phase=phase,updated_unix=time.time(),**kw),ensure_ascii=False,indent=2)+'\n')
try:
    status('acquiring_lock');t.prepare_ssh(out);t.call('shell','mkdir','-p',remote)
    t.call('push',str(binary),remote+'/probe');t.call('push',str(ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'),remote+'/libggml.so');t.call('shell','chmod','755',remote+'/probe')
    t.call('shell','mkdir','-p',remote+'/probe-lib');t.call('push',probe_info['binary'],remote+'/probe-lib/libggml-htp-v75.so')
    lib=remote+':'+t.remote_root+'/a2o_oracle_20261106-143411-669461:'+t.config['runtime_library_dir']+':/vendor/lib64'
    model='/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf'
    manifest=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),model=model,repetitions=10,cpu_reference_threads=6,
        input='fixed uniform[-1,1] activation, seed20261119; real layer0 Q4_0 weights',
        scope='single independent matrix operator; no H, model state, sampling, or end-to-end prefill claim',probe_skel=probe_info)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for label,profile in [('unprofiled',0),('profiled',2)]:
        status('running',condition=label)
        adsp=remote+'/probe-lib' if profile else t.config['runtime_library_dir']
        env='export LD_LIBRARY_PATH='+q(lib)+' ADSP_LIBRARY_PATH='+q(adsp)+' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1 GGML_HEXAGON_USE_HMX=1 GGML_HEXAGON_PROFILE='+str(profile)+'; '
        command=env+shlex.join([remote+'/probe',model,remote+'/'+label+'.json'])+'; result=$?; echo "$result" > '+q(remote+'/'+label+'.exit')
        launch='nohup /system/bin/sh -c '+q(command)+' > '+q(remote+'/'+label+'.log')+' 2>&1 < /dev/null & echo $! > '+q(remote+'/pid')
        t.call('shell',launch)
        for _ in range(300):
            result=t.call('shell','if [ -f '+q(remote+'/'+label+'.exit')+' ]; then cat '+q(remote+'/'+label+'.exit')+'; else echo running; fi').stdout.decode().strip()
            if result!='running':break
            time.sleep(2)
        else:raise RuntimeError('probe timeout; lock and logs retained')
        t.call('pull',remote+'/'+label+'.log',str(out/(label+'.log')))
        if result!='0':raise RuntimeError(label+' failed with exit '+result)
        t.call('pull',remote+'/'+label+'.json',str(out/(label+'.json')))
        data=json.loads((out/(label+'.json')).read_text());assert len(data['cases'])==3 and all(len(c['trials'])==10 for c in data['cases'])
    t.release_lock();subprocess.run(t.prefix+['termux-wake-unlock'],check=True,capture_output=True);status('complete')
except Exception as exc:status('failed',error=str(exc));raise
