#!/usr/bin/env python3
"""短历史真机机制验收；可选单次参考成本，不做性能校准或执行生成动作。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.environment import build,native_source_hashes,PKG
from a2o.runtime.protocol import Protocol
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.phone_preflight import require_phone_preflight
from a2o.validation.oracle import active_model_processes
from a2o.validation.contracts import check_response


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True);parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--self-calibration',action='store_true',help='同一进程增加3个短条件，验收真实耗时的在线修正')
    a=parser.parse_args();runtime=a.runtime_dir.resolve()
    libraries=('libllama.so','libggml-base.so','libggml-hexagon.so')
    for name in libraries:
        if not (runtime/name).is_file():raise ValueError('private runtime missing '+name)
    binary=build();stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out=ROOT/'results'/(stamp+'-execution-prepare-probe');out.mkdir(parents=True);print(out,flush=True)
    remote='/data/local/tmp/a2o_execution_'+stamp;adb=['adb','-s',a.device]
    def call(*args):return subprocess.run(adb+list(args),capture_output=True,check=True,timeout=30)
    manifest={'device':a.device,'model':'/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf','mode':'hybrid','prefill':'HTP','decode':'CPU','H':256,'context':1024,'batch':64,
              'scope':'text mechanism validation; no latency calibration; no generated actions executed','timing_valid_for_performance':False,
              'self_calibration':a.self_calibration,'cost_baseline':'single actual measurement per path; mechanism only, not production calibration',
              'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_files_sha256':native_source_hashes(),
              'libraries':{n:hashlib.sha256((runtime/n).read_bytes()).hexdigest() for n in libraries}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    proc=None;protocol=None;complete=False
    try:
        call('get-state');ps=call('shell','ps -A -o PID,ARGS').stdout.decode();(out/'device-processes-before.txt').write_text(ps)
        if active_model_processes(ps):raise RuntimeError('another model experiment is running')
        power=call('shell','dumpsys','power').stdout.decode();(out/'device-power-before.txt').write_text(power)
        if 'mWakefulness=Awake' not in power:raise RuntimeError('device is not awake; waiting for manual unlock is required')
        call('shell','mkdir',remote);call('push',str(binary),remote+'/agent');call('shell','chmod','755',remote+'/agent')
        (out/'empty.txt').write_text('');call('push',str(out/'empty.txt'),remote+'/empty.txt')
        for name in libraries:
            call('push',str(runtime/name),remote+'/'+name)
            if call('shell','sha256sum',remote+'/'+name).stdout.decode().split()[0]!=manifest['libraries'][name]:raise RuntimeError('private library hash mismatch')
        args=[remote+'/agent','--execution-prepare-probe','1','--model',manifest['model'],'--mode','hybrid','--policy','none','--pre-move','0','--cpu-repack','0','--vision-backend','cpu','--ctx-size','1024','--batch-size','64','--out',remote+'/cases.jsonl']
        args+=['--self-calibration-probe',str(int(a.self_calibration))]
        for key in ('system','history','previous','prompt'):args+=['--'+key,remote+'/empty.txt']
        env='export LD_LIBRARY_PATH='+shlex.quote(remote+':/data/local/tmp/llama.cpp/lib')+' ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1; '
        shell='cd /data/local/tmp/llama.cpp; '+env+'echo $$ > '+remote+'/pid; exec '+shlex.join(args)
        (out/'command.txt').write_text(shell)
        with (out/'runtime.log').open('w') as log:
            require_phone_preflight(DeviceTransport(ROOT, 'adb', device=a.device), out)
            proc=subprocess.Popen(adb+['shell','-T',shell],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            protocol=Protocol(proc,out);protocol.wait('session_ready',timeout=180)
            rows=[]
            for i in range(14 if a.self_calibration else 11):
                row=protocol.wait('execution_case_completed',i,120)['result'];row['response_check']=check_response(row,{'check':{'type':'json_action','action':'wait'}})
                rows.append(row);(out/'checked-cases.json').write_text(json.dumps(rows,indent=2))
                print(row['name'],'graph_hits=',row['graph_hits'],'descriptor_hits=',row['descriptor_hits'],'passed=',row['passed'] and row['response_check']['passed'],flush=True)
                if not row['passed'] or not row['response_check']['passed']:raise RuntimeError('case response failed')
            protocol.wait('session_closed',timeout=30)
            if proc.wait(timeout=30):raise RuntimeError('native exit failed')
            observed=[e for e in protocol.events if e['event']=='self_calibration_observed']
            if a.self_calibration and not (
                    [e['samples'] for e in observed]==[1,2,3]
                    and [e['htp_source'] for e in observed]==['offline','offline','online_corrected']
                    and all(e['cpu_source']=='offline' for e in observed)):
                raise RuntimeError('online correction event sequence failed')
            complete=True;(out/'completion.json').write_text(json.dumps({'passed':True,'cases':len(rows),
                'self_calibration_observations':observed,'timing_valid_for_performance':False},indent=2))
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)},indent=2));raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                call('shell',f'p=$(cat {remote}/pid); if tr "\\0" " " < /proc/$p/cmdline | grep -F -q "{remote}/agent "; then kill -TERM "$p"; fi')
                proc.wait(timeout=30)
            except Exception as exc:(out/'stop-error.txt').write_text(str(exc))
        if protocol:
            protocol.reader.join(timeout=5);(out/'native-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in protocol.events))
        try:
            call('pull',remote,str(out/'device'))
            if complete:call('shell','rm','-rf',remote)
        except Exception as exc:(out/'cleanup-error.txt').write_text(str(exc))

if __name__=='__main__':main()
