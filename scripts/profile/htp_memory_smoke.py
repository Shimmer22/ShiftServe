#!/usr/bin/env python3
"""Termux HTP 短测：阶段缺页计数、进程映射；不推断物理连续性。"""
import fcntl,json,shlex,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
out=ROOT/'results'/(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')+'-htp-memory-smoke');out.mkdir()
lock=(ROOT/'results/.termux-model.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json')
r=t.remote_root+'/'+out.name;q=shlex.quote
try:
    ps=t.call('shell','ps -A -o PID,ARGS').stdout.decode();(out/'processes-before.txt').write_text(ps)
    from a2o.validation.oracle import active_model_processes
    if active_model_processes(ps):raise RuntimeError('设备存在模型进程')
    t.prepare_ssh(out);t.call('shell','mkdir',r)
    (out/'system.txt').write_text('You are a GUI agent. Return only a JSON object with action set to wait.')
    (out/'prompt.txt').write_text('Return {"action":"wait"}.')
    (out/'empty.txt').write_text('')
    for n in ['system','prompt','empty']:t.call('push',str(out/(n+'.txt')),r+'/'+n+'.txt')
    binary='/data/data/com.termux/files/home/.local/a2o/runs/decode_20261106-161056-356843/agent'
    args=[binary,'--model','/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf','--mode','npu','--policy','none','--pre-move','0','--cpu-repack','0','--vision-backend','cpu','--ctx-size','6144','--batch-size','128','--max-tokens','64','--system',r+'/system.txt','--prompt',r+'/prompt.txt','--history',r+'/empty.txt','--previous',r+'/empty.txt','--out',r+'/result.json']
    env='export LD_LIBRARY_PATH='+q(t.config['runtime_library_dir']+':/vendor/lib64:/vendor/lib64/egl')+' ADSP_LIBRARY_PATH='+q(t.config['runtime_library_dir'])+' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1; '
    shell=env+shlex.join(args)+' > '+q(r+'/stdout.log')+' 2> '+q(r+'/runtime.log')+' & p=$!; echo $p > '+q(r+'/pid')+'; while kill -0 $p 2>/dev/null; do cat /proc/$p/stat >> '+q(r+'/stat.log')+'; if [ ! -f '+q(r+'/maps.txt')+' ] && grep -q "PHASE contexts_ready" '+q(r+'/runtime.log')+'; then cat /proc/$p/maps > '+q(r+'/maps.txt')+'; cat /proc/$p/smaps > '+q(r+'/smaps.txt')+' 2> '+q(r+'/smaps-error.txt')+'; dd if=/proc/$p/pagemap of='+q(r+'/pagemap-first.bin')+' bs=8 count=1 2> '+q(r+'/pagemap-access.txt')+'; fi; sleep 0.1; done; wait $p'
    (out/'command.txt').write_text(shell)
    with (out/'ssh.log').open('w') as log:result=subprocess.run(t.shell_command(shell),stdout=log,stderr=subprocess.STDOUT,timeout=180)
    t.call('pull',r,str(out/'device'))
    if result.returncode:raise RuntimeError('native exit '+str(result.returncode))
    data=json.loads((out/'device/result.json').read_text());res=data['resources']; stages={}
    for name,a,b in [('prefill','observation_arrival','decode_start'),('decode','decode_start','decode_end')]:
        stages[name]={k:res[b][k]-res[a][k] for k in res[b] if 'fault' in k};stages[name]['swap_end_kib']=res[b].get('VmSwap_kib')
    passed=data['eog'] and json.loads(data['output']).get('action')=='wait'
    summary=dict(passed=passed,backend='HTP0',context=6144,observation_tokens=data['observation_tokens'],stages=stages,timing_ms=data['timing_ms'],output=data['output'],physical_contiguity='not measured; pagemap PFN requires permission')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(out);print(json.dumps(summary,ensure_ascii=False))
except BaseException as e:
    (out/'failure.json').write_text(json.dumps({'error':repr(e)}));raise
finally:
    t.release_lock()
    subprocess.run(t.prefix+['termux-wake-unlock'],capture_output=True)
