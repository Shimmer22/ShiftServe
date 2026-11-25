#!/usr/bin/env python3
"""仅短decode归因，复用历史快照，CPU/HTP条件持锁串行执行。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.phone_preflight import require_phone_preflight


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/decode-attribution.json');p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads(a.config.read_text());out=(a.output or ROOT/'results'/plan['name']).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise RuntimeError('禁止覆盖已完成结果')
    build=ROOT/plan.get('build','results/20261113-decode-attribution/build');fixture=json.loads((ROOT/plan['fixture']).read_text())
    model=plan.get('model_path','/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf')
    t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json');q=shlex.quote;remote=t.remote_root+'/'+out.name
    def status(phase,**kw):save(out/'status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**kw))
    held=False;running=False
    try:
        ps=t.call('shell','ps -A -o PID,ARGS').stdout.decode();(out/'processes-before.txt').write_text(ps)
        active=[s for s in ps.splitlines() if len(s.split())>1 and s.split()[1].rsplit('/',1)[-1] in ['probe','agent','service','llama-server']]
        if active and (not a.resume or any(remote+'/' not in line for line in active)):raise RuntimeError('已有模型进程：'+str(active))
        running=bool(active)
        dsp=json.loads((ROOT/'results/20261111-htp-platform-exploration/probe-build-v75/completion.json').read_text())
        status('acquiring_lock');t.prepare_ssh(out,resume_lock=a.resume);held=True
        if plan.get('model_sha256'):
            actual=t.call('shell','sha256sum '+q(model)).stdout.decode().split()[0]
            if actual!=plan['model_sha256']:raise RuntimeError('模型SHA-256不匹配')
        t.call('shell','mkdir','-p',remote+'/instrumented',remote+'/probe-dsp',remote+'/diagnostic')
        if not a.resume:
            for local,name in [(build/'probe','probe'),(build/'libggml-base.so','instrumented/libggml-base.so'),
                (ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so','libggml.so')]:t.call('push',str(local),remote+'/'+name)
            if (build/'libllama.so').exists():t.call('push',str(build/'libllama.so'),remote+'/diagnostic/libllama.so')
            if (build/'libggml-cpu.so').exists():t.call('push',str(build/'libggml-cpu.so'),remote+'/instrumented/libggml-cpu.so')
            t.call('push',dsp['binary'],remote+'/probe-dsp/libggml-htp-v75.so');t.call('shell','chmod','755',remote+'/probe')
        identities=t.call('shell','sha256sum '+shlex.join([remote+'/probe',remote+'/instrumented/libggml-base.so',remote+'/probe-dsp/libggml-htp-v75.so',t.config['runtime_library_dir']+'/libllama.so',t.config['runtime_library_dir']+'/libggml-base.so',t.config['runtime_library_dir']+'/libggml-hexagon.so',t.config['runtime_library_dir']+'/libggml-htp-v75.so'])).stdout.decode()
        (out/'device-identities.txt').write_text(identities)
        assert identities.split()[0]==hashlib.sha256((build/'probe').read_bytes()).hexdigest()
        assert fixture['expected_libllama_sha256'] in identities
        save(out/'manifest.json',dict(plan=plan,build=str(build),fixture_source=plan['fixture'],private_dsp=dsp,
            scope='原版计时、主机归因、私有DSP路由分开；无KV整理/预搬移；固定token不执行动作；不作任务质量结论'))
        for case in plan['cases']:
            label=case['name'];folder=out/label;folder.mkdir(exist_ok=True)
            cfg=dict(fixture,**{k:v for k,v in case.items() if k not in ['name','backend']})
            desired=case.get('D',plan['D'])
            available=fixture['generated_ids']
            if desired>len(available):
                available=json.loads((ROOT/fixture['source']).read_text())['choices'][0]['token_ids']
                assert available[:len(fixture['generated_ids'])]==fixture['generated_ids']
            cfg['generated_ids']=available[:desired];assert len(cfg['generated_ids'])==desired
            cfg['rounds']=case.get('rounds',plan['rounds']);cfg['keep_ratio']=case.get('keep_ratio',1.0)
            cfg['host_instrumentation']=case.get('host_instrumentation',False)
            save(folder/'config.json',cfg);t.call('push',str(folder/'config.json'),remote+'/'+label+'.config.json')
            lib=(remote+'/diagnostic:' if case.get('cpu_fused_gdn') else '')+(remote+'/instrumented:' if cfg['host_instrumentation'] else '')+remote+':'+t.remote_root+'/a2o_oracle_20261106-143411-669461:'+t.config['runtime_library_dir']+':/vendor/lib64'
            adsp=remote+'/probe-dsp' if case.get('private_dsp') else t.config['runtime_library_dir']
            env=dict(LD_LIBRARY_PATH=lib,ADSP_LIBRARY_PATH=adsp,GGML_HEXAGON_EXPERIMENTAL='1',GGML_HEXAGON_OPQUEUE=str(case.get('opqueue',1)),GGML_HEXAGON_OPBATCH=str(case.get('opbatch',1)),GGML_HEXAGON_USE_HMX='1',GGML_HEXAGON_PROFILE=str(case.get('profile',0)))
            if case.get('cpu_fused_gdn'):env['A2O_DIAG_CPU_FUSED_GDN']='1'
            command='export '+' '.join(k+'='+q(v) for k,v in env.items())+'; '+shlex.join([remote+'/probe',model,remote+'/'+label+'.config.json',case['backend'],remote+'/'+label+'.json'])+'; rc=$?; echo "$rc" > '+q(remote+'/'+label+'.exit')
            save(folder/'command.json',dict(command=command,environment=env));status('running',case=label)
            already=t.call('shell','test -f '+q(remote+'/'+label+'.log')+' && echo yes || echo no').stdout.decode().strip()=='yes'
            if not (a.resume and already):
                t.call('shell','test ! -f '+q(remote+'/'+label+'.exit'))
                require_phone_preflight(t, folder)
                t.call('shell','nohup /system/bin/sh -c '+q(command)+' > '+q(remote+'/'+label+'.log')+' 2>&1 < /dev/null & echo $! > '+q(remote+'/pid'))
            running=True;begin=time.monotonic();mapped=False;d=None
            while True:
                rc=t.call('shell','if [ -f '+q(remote+'/'+label+'.exit')+' ]; then cat '+q(remote+'/'+label+'.exit')+'; else echo running; fi').stdout.decode().strip()
                exists=t.call('shell','test -f '+q(remote+'/'+label+'.json')+' && echo yes || echo no').stdout.decode().strip()
                if exists=='yes':
                    t.call('pull',remote+'/'+label+'.json',str(folder/'response.json'));d=json.loads((folder/'response.json').read_text());status('running',case=label,rows=len(d['rows']),elapsed_s=time.monotonic()-begin)
                    if not mapped:
                        ps=t.call('shell','ps -A -o PID,ARGS').stdout.decode();lines=[s for s in ps.splitlines() if len(s.split())>1 and s.split()[1].rsplit('/',1)[-1]=='probe' and remote+'/'+label+'.config.json' in s]
                        if lines:(folder/'maps.txt').write_bytes(t.call('shell','cat /proc/'+lines[0].split()[0]+'/maps').stdout);mapped=True
                if rc!='running':
                    running=False;t.call('pull',remote+'/'+label+'.log',str(folder/'runtime.log'));save(folder/'exit.json',dict(exit_code=int(rc)))
                    if rc!='0':raise RuntimeError(label+'失败，退出码'+rc)
                    assert d['completed'] and d['normal_response']['passed'] and d['state_restore_verified']
                    if cfg.get('spec_oracle'):
                        assert len([r for r in d['rows'] if not r['warmup']])==len(cfg['oracle_k'])*len(cfg.get('oracle_modes',['cpu_decode','verify_only','draft_only','serial','overlap']))*cfg['rounds']
                        break
                    assert len([r for r in d['rows'] if not r['warmup']])==cfg['rounds']
                    assert d.get('warmup_elapsed_ms',0)>=1000.*cfg.get('minimum_warmup_seconds',0)
                    if cfg['host_instrumentation']:assert all(r['backend_timing'] for r in d['rows'])
                    if cfg.get('save_logits'):t.call('pull',remote+'/'+label+'.json.logits.f32',str(folder/'logits.f32'))
                    if cfg.get('check_checkpoint_replay'):assert d['checkpoint_replay']['final_state_bitwise_equal']
                    if cfg.get('check_json_memory'):assert d['json_memory']['passed']
                    break
                if time.monotonic()-begin>case.get('timeout_seconds',plan.get('timeout_seconds',600)):raise RuntimeError('超时，保留进程和锁')
                time.sleep(3)
        status('complete');save(out/'completion.json',dict(completed=True,cases=[c['name'] for c in plan['cases']]))
    except Exception as exc:status('failed',error=str(exc),process_may_be_running=running);raise
    finally:
        if held and not running:
            t.release_lock();subprocess.run(t.prefix+['termux-wake-unlock'],check=True,capture_output=True,timeout=30)
            (out/'processes-after.txt').write_bytes(t.call('shell','ps -A -o PID,ARGS').stdout)


if __name__=='__main__':main()
