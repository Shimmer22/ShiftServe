#!/usr/bin/env python3
"""Termux SSH串行decode隔离测量；复用已验证文本夹具，不执行生成动作。"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.environment import build,native_source_hashes
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.protocol import Protocol
from a2o.validation.contracts import check_response
from a2o.validation.oracle import active_model_processes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/decode-backends-termux.json');a=ap.parse_args()
    c=json.loads(a.config.read_text())
    if (c['mode']!='hybrid' or c['kv_policy']!='none' or c['pre_move'] or c['cpu_repack']
            or c['backends']!=['cpu','htp','opencl'] or c['repetitions']<3
            or not 8<=c['horizon']<=256 or c['batch_size']!=128):raise ValueError('要求显式三后端、无整理/预搬移/repack及合法decode长度')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out=ROOT/'results'/(stamp+'-'+c.get('tag','decode-backends'));out.mkdir();print(out,flush=True)
    lock=(ROOT/'results/.termux-model.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binary=ROOT/c['agent_binary'] if c.get('agent_binary') else build()
    fixtures=out/'fixtures';fixtures.mkdir();source_hashes={}
    for h,source in c['history_fixtures'].items():
        dest=fixtures/f'H{h}.txt';shutil.copyfile(ROOT/source,dest);source_hashes[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    for p,source in c['input_fixtures'].items():
        dest=fixtures/f'P{p}.txt';shutil.copyfile(ROOT/source,dest);source_hashes[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
    remote=t.remote_root+'/decode_'+stamp;runtime=t.config['runtime_library_dir'];proc=None;protocol=None;complete=False
    expected=len(c['history_fixtures'])*len(c['input_fixtures'])*(1+3+3*c['repetitions'])
    manifest=dict(config=c,transport='ssh',clock_id='CLOCK_BOOTTIME',device=t.identity,
        model='/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        source_files_sha256=native_source_hashes(),fixture_sha256=source_hashes,expected_trials=expected,
        numeric_acceptance=dict(min_cosine=.99,max_nrmse=None),kv_type='F16',flash_attention=False,
        semantics=f"natural JSON until EOG; separate fixed {c['horizon']}-step compute replay; replay is not a natural response; migration and numerical audits excluded from decode sums")
    def save():(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    save()
    log=(out/'runtime.log').open('w')
    try:
        t.prepare_ssh(out)
        processes=t.call('shell','ps -A -o PID,ARGS').stdout.decode();(out/'device-processes-before.txt').write_text(processes)
        if active_model_processes(processes):raise RuntimeError('设备已有模型进程，拒绝争抢')
        hashes=t.call('shell','sha256sum '+shlex.quote(runtime)+'/*.so').stdout.decode();(out/'runtime-sha256.txt').write_text(hashes)
        t.call('shell','mkdir',remote);t.call('push',str(binary),remote+'/agent');t.call('shell','chmod','755',remote+'/agent');t.call('push',str(fixtures),remote+'/fixtures')
        (out/'empty.txt').write_text('');t.call('push',str(out/'empty.txt'),remote+'/empty.txt')
        args=[remote+'/agent','--decode-profile','1','--model',manifest['model'],'--mode','hybrid','--policy','none','--pre-move','0','--cpu-repack','0','--vision-backend','cpu',
            '--ctx-size',str(c['context_size']),'--batch-size',str(c['batch_size']),'--decode-repeats',str(c['repetitions']),'--decode-horizon',str(c['horizon']),
            '--decode-histories',','.join(c['history_fixtures']),'--decode-inputs',','.join(c['input_fixtures']),'--decode-fixtures',remote+'/fixtures','--out',remote+'/trials.jsonl']
        for name in ['system','history','previous','prompt']:args+=['--'+name,remote+'/empty.txt']
        env='LD_LIBRARY_PATH='+shlex.quote(remote+':'+runtime+':/vendor/lib64:/vendor/lib64/egl')+' ADSP_LIBRARY_PATH='+shlex.quote(runtime)+' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1'
        shell='export '+env+'; echo $$ > '+shlex.quote(remote+'/pid')+'; exec '+shlex.join(args)
        (out/'command.txt').write_text(shell+'\n');manifest['command']=shell;save()
        proc=subprocess.Popen(t.shell_command(shell),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        protocol=Protocol(proc,out)
        for i in range(expected):
            e=protocol.wait('decode_trial_completed',i,timeout=c['timeout_s'])
            print(f"{i} {e['phase']} {e['backend']} H={e['H']} P={e['P']} passed={e['passed']}",flush=True)
            if not e.get('natural_passed',e['passed']):raise RuntimeError('decode natural response/numerical failure')
        protocol.wait('decode_session_completed',timeout=60)
        if proc.wait(timeout=60):raise RuntimeError('native exit failed')
        t.call('pull',remote,str(out/'device'))
        raw=[json.loads(x) for x in (out/'device/trials.jsonl').read_text().splitlines()]
        assert len(raw)==expected
        for r in raw:
            r['response_check']=check_response(r,{'check':{'type':'json_action','action':'wait'}})
            if not r.get('natural_passed',r['passed']) or not r['response_check']['passed'] or not r['eog'] or not r['generated_ids_equal']:raise RuntimeError('完整响应/迁移验收失败')
        (out/'checked-rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw))
        (out/'completion.json').write_text(json.dumps(dict(completed=True,passed=all(r['passed'] for r in raw),response_and_state_passed=True,all_numerical_passed=all(r['passed'] for r in raw),trials=len(raw),measurement_trials=sum(r['phase']=='measurement' for r in raw)),indent=2)+'\n');complete=True
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)},ensure_ascii=False,indent=2)+'\n');raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                t.call('shell','p=$(cat '+shlex.quote(remote+'/pid')+'); if [ "$(readlink /proc/$p/exe)" = '+shlex.quote(remote+'/agent')+' ]; then kill -TERM "$p"; fi')
                proc.wait(timeout=30)
            except Exception as exc:(out/'stop-error.txt').write_text(repr(exc))
        if protocol:
            protocol.reader.join(timeout=5);(out/'native-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in protocol.events))
        if not complete:
            try:t.call('pull',remote,str(out/'failure-device'))
            except Exception as exc:(out/'pull-error.txt').write_text(repr(exc))
        if proc is None or proc.poll() is not None:
            t.release_lock();p=subprocess.run(t.prefix+['termux-wake-unlock'],capture_output=True,timeout=30);(out/'wake-unlock.log').write_bytes(p.stdout+p.stderr)
        log.close()
    print(json.dumps({'run':str(out),'passed':all(r['passed'] for r in raw),'response_and_state_passed':True,'trials':expected}),flush=True)

if __name__=='__main__':main()
