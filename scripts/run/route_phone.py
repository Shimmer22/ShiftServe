#!/usr/bin/env python3
"""持锁启动/停止私有 CPU、HTP offload 或 route 手机研究服务。"""
import argparse,json,shlex,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.validation.oracle import active_model_processes
from a2o.runtime.network_retry import retry_network
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',default='/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf')
p.add_argument('--model-id',default='qwen35-9b-q4_0')
p.add_argument('--model-sha256')
p.add_argument('--opbatch',type=int,default=1)
p.add_argument('--opqueue',type=int,default=1)
p.add_argument('--kv-offload',type=int,choices=[0,1],default=0)
p.add_argument('--flash-attention',type=int,choices=[0,1],default=0)
p.add_argument('--single-resident',action='store_true')
p.add_argument('action',choices=['start','stop'])
p.add_argument('--condition',choices=['cpu','htp','route'],required=True)
p.add_argument('--binary',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--costs',type=Path)
p.add_argument('--route-costs',type=Path)
p.add_argument('--kv-costs',type=Path)
p.add_argument('--registry',type=Path)
p.add_argument('--retry-network',action='store_true')
p.add_argument('--context-size',type=int,default=4096)
p.add_argument('--replay-cache',action='store_true')
p.add_argument('--logs-only',action='store_true',help='停止时只收回日志和PID，大缓存与二进制保留在手机')
p.add_argument('--cpu-threads',type=int,default=4,choices=range(1,17))
p.add_argument('--cpu-repack',type=int,choices=[0,1],default=0)
p.add_argument('--kv-reduce',type=int,choices=[0,1],default=0)
p.add_argument('--kv-policy',choices=['prefill_prior','recent','adaptive_cost'],default='prefill_prior')
p.add_argument('--kv-keep-tokens',type=int,default=512)
p.add_argument('--kv-sink-tokens',type=int,default=4)
p.add_argument('--kv-recent-tokens',type=int,default=64)
p.add_argument('--kv-pool-window',type=int,default=5)
p.add_argument('--kv-tail-tokens',type=int,default=16)
p.add_argument('--compact-anchor',type=int,choices=[0,1],default=0)
p.add_argument('--kv-hwm-trim',type=int,choices=[0,1],default=1)
p.add_argument('--expected-decode',type=int,default=96)
p.add_argument('--probe-skel',type=Path)
p.add_argument('--htp-profile',type=int,choices=[0,2],default=0)
a=p.parse_args()
if bool(a.probe_skel)!=(a.htp_profile==2):p.error('私有DSP探针必须同时使用--probe-skel与--htp-profile 2')
if a.kv_reduce and a.kv_policy=='adaptive_cost' and not a.kv_costs:p.error('--kv-policy adaptive_cost 需要 --kv-costs')
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
 model=a.model
 if a.model_sha256:
  digest=t.call('shell','sha256sum '+q(model)).stdout.decode().split()[0]
  if digest!=a.model_sha256:raise RuntimeError('模型SHA-256不匹配，拒绝启动')
 lib=remote+':'+t.remote_root+'/a2o_oracle_20261106-143411-669461:'+t.config['runtime_library_dir']+':/vendor/lib64:/vendor/lib64/egl'
 cache_env=(' A2O_REPLAY_CACHE_DIR='+q(remote+'/replay-cache')) if a.replay_cache else ''
 command='export LD_LIBRARY_PATH='+q(lib)+' ADSP_LIBRARY_PATH='+q(remote if a.probe_skel else t.config['runtime_library_dir'])+' GGML_HEXAGON_PROFILE='+str(a.htp_profile)+' GGML_HEXAGON_USE_HMX=1 A2O_CPU_THREADS='+str(a.cpu_threads)+' A2O_CPU_REPACK='+str(a.cpu_repack)+' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE='+str(a.opqueue)+' GGML_HEXAGON_OPBATCH='+str(a.opbatch)+' A2O_HTP_KV_OFFLOAD='+str(a.kv_offload)+' A2O_FLASH_ATTN='+str(a.flash_attention)+' A2O_KV_REDUCE='+str(a.kv_reduce)+' A2O_KV_POLICY='+q(a.kv_policy)+' A2O_KV_KEEP_TOKENS='+str(a.kv_keep_tokens)+' A2O_KV_SINK_TOKENS='+str(a.kv_sink_tokens)+' A2O_KV_RECENT_TOKENS='+str(a.kv_recent_tokens)+' A2O_KV_POOL_WINDOW='+str(a.kv_pool_window)+' A2O_KV_TAIL_TOKENS='+str(a.kv_tail_tokens)+' A2O_COMPACT_ANCHOR='+str(a.compact_anchor)+' A2O_KV_HWM_TRIM='+str(a.kv_hwm_trim)+' A2O_EXPECTED_DECODE='+str(a.expected_decode)+' A2O_SINGLE_RESIDENT='+str(int(a.single_resident))+' A2O_MODEL_ID='+q(a.model_id)+cache_env+'; exec '+shlex.join([remote+'/service',model,a.condition,str(a.context_size),'18780',costs])
 launch='nohup /system/bin/sh -c '+q(command)+' > '+q(remote+'/runtime.log')+' 2>&1 < /dev/null & echo $! > '+q(remote+'/pid')
 t.call('shell',owner_check+' && { if '+pid_alive+'; then :; else '+launch+'; fi; }')
 (out/'manifest.json').write_text(json.dumps(dict(condition=a.condition,remote=remote,command=command,context_size=a.context_size,replay_cache=a.replay_cache,cpu_threads=a.cpu_threads,cpu_threads_batch=a.cpu_threads,cpu_repack=bool(a.cpu_repack),htp_host_threads=4,htp_profile=a.htp_profile,probe_skel=str(a.probe_skel),batch_size=128,single_resident=a.single_resident,model=model,model_id=a.model_id,model_sha256=a.model_sha256,opbatch=a.opbatch,opqueue=a.opqueue,flash_attention=bool(a.flash_attention),kv_offload=bool(a.kv_offload),kv_policy=(a.kv_policy if a.kv_reduce else 'none'),kv_reduce=bool(a.kv_reduce),kv_keep_tokens=(a.kv_keep_tokens if a.kv_reduce else None),kv_sink_tokens=(a.kv_sink_tokens if a.kv_reduce else None),kv_recent_tokens=(a.kv_recent_tokens if a.kv_reduce else None),kv_pool_window=(a.kv_pool_window if a.kv_reduce else None),kv_tail_tokens=(a.kv_tail_tokens if a.kv_reduce else None),compact_anchor=(bool(a.compact_anchor) if a.kv_reduce else False),expected_decode_tokens=a.expected_decode,kv_costs=(str(a.kv_costs) if a.kv_costs else None),route_costs=(str(a.route_costs) if a.route_costs else None),pre_move=False,costs=str(a.costs)),indent=2)+'\n')
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  try:
   with urllib.request.urlopen('http://10.29.131.50:18780/health',timeout=3) as r:health=json.load(r)
   if health.get('cpu_threads',4)!=a.cpu_threads:raise RuntimeError('CPU线程数与请求不一致')
   if 'model' in health and health['model']!=a.model_id:raise RuntimeError('模型身份与请求不一致')
   if 'htp_kv_offload' in health and health['htp_kv_offload']!=bool(a.kv_offload):raise RuntimeError('KV offload配置不一致')
   cost_payload={}
   if a.kv_costs:cost_payload['kv_model']=json.loads(a.kv_costs.read_text())
   if a.route_costs:
    rc=json.loads(a.route_costs.read_text())
    cost_payload['model']=rc['model'] if ('model' in rc and isinstance(rc['model'],dict)) else rc
   if cost_payload:
    req=urllib.request.Request('http://10.29.131.50:18780/a2o/costs',data=json.dumps(cost_payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=15) as kr:health['cost_install']=json.load(kr)
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
 for _ in range(30):
  state=t.call('shell','if '+pid_alive+'; then echo running; else echo stopped; fi').stdout.decode().strip()
  if state=='stopped':break
  time.sleep(1)
 else:raise RuntimeError('私有服务未退出，保留锁；禁止并行启动下一模型')
 if a.logs_only:
  device=out/'device';device.mkdir(exist_ok=True)
  for name in ['runtime.log','pid']:t.call('pull',remote+'/'+name,str(device/name))
  (out/'remote-artifacts.json').write_text(json.dumps(dict(remote=remote,replay_cache=remote+'/replay-cache',retained_on_phone=True,downloaded=['runtime.log','pid']),indent=2)+'\n')
 else:t.call('pull',remote+'/.',str(out/'device'))
 t.release_lock()
 retry(lambda:subprocess.run(t.prefix+['termux-wake-unlock'],check=True,capture_output=True,timeout=30))
