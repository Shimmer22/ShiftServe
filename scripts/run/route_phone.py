#!/usr/bin/env python3
"""持锁启动/停止私有 CPU、HTP offload 或 route 手机研究服务。"""
import argparse,json,shlex,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.validation.oracle import active_model_processes
from a2o.runtime.network_retry import retry_network
p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','stop']);p.add_argument('--condition',choices=['cpu','htp','route'],required=True);p.add_argument('--binary',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--costs',type=Path);p.add_argument('--registry',type=Path);p.add_argument('--retry-network',action='store_true');p.add_argument('--context-size',type=int,default=4096);p.add_argument('--replay-cache',action='store_true');p.add_argument('--cpu-threads',type=int,default=4,choices=range(1,17));p.add_argument('--probe-skel',type=Path);p.add_argument('--htp-profile',type=int,choices=[0,2],default=0);a=p.parse_args()
if bool(a.probe_skel)!=(a.htp_profile==2):p.error('私有DSP探针必须同时使用--probe-skel与--htp-profile 2')
t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json');out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);remote=t.remote_root+'/'+out.name;q=shlex.quote
def retry(operation):
 return retry_network(operation,report=lambda message:print(message,flush=True)) if a.retry_network else operation()
original_call=t.call
t.call=lambda *args:retry(lambda:original_call(*args))
# 每次启动/停止均核对锁归属和具体服务路径，重连不重复启动或误杀复用PID。
owner_check='test "$(cat '+q(t.remote_root+'/model.lock/owner')+')" = '+q(out.name)
pid_alive='p=$(cat '+q(remote+'/pid')+' 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && tr "\\000" " " < /proc/"$p"/cmdline | /system/bin/grep -F '+q(remote+'/service')+' >/dev/null'
if a.action=='start':
 ps=t.call('shell','ps -A -o PID,ARGS').stdout.decode();(out/'processes-before.txt').write_text(ps)
 others=[line for line in ps.splitlines() if remote+'/service' not in line]
 if active_model_processes('\n'.join(others)):raise RuntimeError('设备已有其他模型进程')
 retry(lambda:t.prepare_ssh(out,resume_lock=a.retry_network));t.call('shell','mkdir','-p',remote)
 exists=t.call('shell','if '+pid_alive+'; then echo running; else echo stopped; fi').stdout.decode().strip()
 if exists!='running':
  t.call('push',str(a.binary),remote+'/service');t.call('shell','chmod','755',remote+'/service')
 if a.registry and exists!='running':t.call('push',str(a.registry),remote+'/libggml.so')
 if a.probe_skel and exists!='running':t.call('push',str(a.probe_skel),remote+'/'+a.probe_skel.name)
 costs='none'
 if a.costs:t.call('push',str(a.costs),remote+'/costs.json');costs=remote+'/costs.json'
 model='/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf'
 lib=remote+':'+t.remote_root+'/a2o_oracle_20261106-143411-669461:'+t.config['runtime_library_dir']+':/vendor/lib64:/vendor/lib64/egl'
 cache_env=(' A2O_REPLAY_CACHE_DIR='+q(remote+'/replay-cache')) if a.replay_cache else ''
 command='export LD_LIBRARY_PATH='+q(lib)+' ADSP_LIBRARY_PATH='+q(remote if a.probe_skel else t.config['runtime_library_dir'])+' GGML_HEXAGON_PROFILE='+str(a.htp_profile)+' GGML_HEXAGON_USE_HMX=1 A2O_CPU_THREADS='+str(a.cpu_threads)+' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1'+cache_env+'; exec '+shlex.join([remote+'/service',model,a.condition,str(a.context_size),'18780',costs])
 launch='nohup /system/bin/sh -c '+q(command)+' > '+q(remote+'/runtime.log')+' 2>&1 < /dev/null & echo $! > '+q(remote+'/pid')
 t.call('shell',owner_check+' && { if '+pid_alive+'; then :; else '+launch+'; fi; }')
 (out/'manifest.json').write_text(json.dumps(dict(condition=a.condition,remote=remote,command=command,context_size=a.context_size,replay_cache=a.replay_cache,cpu_threads=a.cpu_threads,cpu_threads_batch=a.cpu_threads,htp_host_threads=4,htp_profile=a.htp_profile,probe_skel=str(a.probe_skel),batch_size=128,kv_offload=False,kv_policy='none',pre_move=False,costs=str(a.costs)),indent=2)+'\n')
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  try:
   with urllib.request.urlopen('http://10.29.131.50:18780/health',timeout=3) as r:health=json.load(r)
   if health.get('cpu_threads',4)!=a.cpu_threads:raise RuntimeError('CPU线程数与请求不一致')
   (out/'health.json').write_text(json.dumps(health));print(health);break
  except OSError:
   # 仅网络等待不消耗初始化预算；SSH连通后继续检查服务。
   before=time.monotonic()
   retry(lambda:original_call('get-state'))
   state=t.call('shell','if '+pid_alive+'; then echo running; else echo stopped; fi').stdout.decode().strip()
   if state!='running':raise RuntimeError('私有服务已退出，检查runtime.log；锁与失败现场保留')
   deadline+=time.monotonic()-before
   time.sleep(1)
 else:raise RuntimeError('服务未就绪，锁和现场保留')
else:
 t.lock_owner=out.name
 command='if [ -d '+q(t.remote_root+'/model.lock')+' ]; then '+owner_check+' && { if '+pid_alive+'; then kill "$p"; else :; fi; }; fi'
 t.call('shell',command)
 t.call('pull',remote+'/.',str(out/'device'));t.release_lock()
 retry(lambda:subprocess.run(t.prefix+['termux-wake-unlock'],check=True,capture_output=True,timeout=30))
