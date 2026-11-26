#!/usr/bin/env python3
"""持锁启动/停止私有 CPU、HTP offload 或 route 手机研究服务。"""
import argparse,hashlib,json,os,shlex,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.validation.oracle import active_model_processes
from a2o.runtime.network_retry import retry_network
from a2o.runtime.model_digest import verify_model_digest
from a2o.runtime.phone_preflight import PhonePreflightError, require_phone_preflight
BASE=os.environ.get('A2O_PHONE_HTTP_BASE','http://10.29.131.50:18780').rstrip('/')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',default='/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf')
p.add_argument('--model-id',default='qwen35-9b-q4_0')
p.add_argument('--model-sha256')
p.add_argument('--rehash-model',action='store_true',help='启动前重新读取整个模型进行 SHA-256 校验')
p.add_argument('--verify-kv-state',type=int,choices=[0,1],help='恢复和交接后全量回读核验 KV')
p.add_argument('--opbatch',type=int,default=1024)
p.add_argument('--opqueue',type=int,default=16)
p.add_argument('--kv-offload',type=int,choices=[0,1],default=0)
p.add_argument('--flash-attention',type=int,choices=[0,1],default=0)
p.add_argument('--single-resident',action='store_true')
p.add_argument('action',choices=['start','stop'])
p.add_argument('--condition',choices=['cpu','htp','route'],required=True)
p.add_argument('--binary',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--costs',type=Path)
p.add_argument('--route-costs',type=Path)
p.add_argument('--cold-costs',type=Path,help='H=0 实测 cold_route_v1，points 包含 P/cc_ms/hc_ms')
p.add_argument('--kv-costs',type=Path)
p.add_argument('--registry',type=Path)
p.add_argument('--runtime-overlay',type=Path,help='私有构建目录，含 agent、CPU/llama 动态库与校验清单')
p.add_argument('--shared-weights',action='store_true',help='CPU 直接读取 HTP 量化权重')
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
p.add_argument('--kv-layer-stride',type=int,default=4)
p.add_argument('--kv-min-evict-tokens',type=int,default=256)
p.add_argument('--compact-anchor',type=int,choices=[0,1],default=0)
p.add_argument('--kv-hwm-trim',type=int,choices=[0,1])
p.add_argument('--memory-ledger',action='store_true',help='采集加载与逐轮端侧内存账本')
p.add_argument('--m1-explorer',type=int,choices=[0,1],default=0)
p.add_argument('--device',default='xiaomi14pro')
p.add_argument('--gap-ms',type=int,default=-1)
p.add_argument('--expected-decode',type=int,default=96)
p.add_argument('--probe-skel',type=Path)
p.add_argument('--htp-profile',type=int,choices=[0,1,2],default=0)
p.add_argument('--htp-big-cores',type=int,choices=[0,1],default=0)
p.add_argument('--max-quiet-c',type=float,default=None)
p.add_argument('--min-prime-khz',type=int,default=None)
p.add_argument('--min-mem-avail-kib',type=int,default=4000000)
p.add_argument('--cooldown-timeout-s',type=int,default=0)
a=p.parse_args()
if a.action=='start' and a.memory_ledger and a.kv_hwm_trim is None:
 p.error('--memory-ledger 需要显式指定 --kv-hwm-trim 0 或 1')
if a.kv_hwm_trim is None:a.kv_hwm_trim=1
if bool(a.probe_skel)!=(a.htp_profile==2):p.error('私有DSP探针必须同时使用--probe-skel与--htp-profile 2')
if a.kv_reduce and a.kv_policy=='adaptive_cost' and not a.kv_costs:p.error('--kv-policy adaptive_cost 需要 --kv-costs')
if a.shared_weights and (a.condition!='route' or a.single_resident or a.cpu_repack):
 p.error('--shared-weights 需要 route、双上下文和 --cpu-repack 0')
overlay={}
hexagon_sha256=None
build_meta={}
if a.action=='start':
 if a.cold_costs and (a.condition!='route' or a.single_resident):p.error('--cold-costs 需要 route 和双上下文')
 if a.shared_weights and not a.runtime_overlay:p.error('--shared-weights 需要 --runtime-overlay')
 if a.runtime_overlay:
  build=a.runtime_overlay.resolve()
  build_manifest=json.loads((build/'manifest.json').read_text())
  declared=build_manifest['artifacts']
  hexagon_sha256=build_manifest.get('hexagon_sha256')
  build_meta={'git_commit':build_manifest.get('git_commit'),'build_id':build_manifest.get('build_id')}
  if a.shared_weights and not hexagon_sha256:p.error('构建清单缺少 Hexagon 布局兼容校验')
  for name in ('agent','libggml-cpu.so','libllama.so'):
   artifact=build/name
   digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
   if digest!=declared[name]:p.error(f'构建产物校验失败: {artifact}')
   overlay[name]=digest
  if a.binary and a.binary.resolve()!=build/'agent':p.error('--binary 与 --runtime-overlay 构建不一致')
  a.binary=build/'agent'
 if not a.binary:p.error('start 需要 --binary 或 --runtime-overlay')
t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json');out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);remote=t.remote_root+'/'+out.name;q=shlex.quote
def retry(operation):
 return retry_network(operation,report=lambda message:print(message,flush=True)) if a.retry_network else operation()
original_call=t.call
t.call=lambda *args:retry(lambda:original_call(*args))
# 每次启动/停止均核对锁归属和具体服务路径，重连不重复启动或误杀复用PID。
owner_check='test "$(cat '+q(t.remote_root+'/model.lock/owner')+')" = '+q(out.name)
pid_alive='p=$(cat '+q(remote+'/pid')+' 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && /system/bin/grep -a -F '+q(remote+'/service')+' /proc/"$p"/cmdline >/dev/null 2>&1'
if a.action=='start':
 ps=t.call('shell','ps -A -o PID,ARGS').stdout.decode();(out/'processes-before.txt').write_text(ps)
 others=[line for line in ps.splitlines() if remote+'/service' not in line]
 if active_model_processes('\n'.join(others)):raise RuntimeError('设备已有其他模型进程')
 retry(lambda:t.prepare_ssh(out,resume_lock=a.retry_network));t.call('shell','mkdir','-p',remote)
 exists=t.call('shell','if '+pid_alive+'; then echo running; else echo stopped; fi').stdout.decode().strip()
 if exists!='running':
  t.call('push',str(a.binary),remote+'/service');t.call('shell','chmod','755',remote+'/service')
 if overlay:
  for name,digest in overlay.items():
   dest=remote+('/service' if name=='agent' else '/'+name)
   if name!='agent' and exists!='running':t.call('push',str(a.runtime_overlay/name),dest)
   remote_digest=t.call('shell','sha256sum '+q(dest)).stdout.decode().split()[0]
   if remote_digest!=digest:raise RuntimeError(f'手机运行时产物校验失败: {name}')
 if a.registry and exists!='running':t.call('push',str(a.registry),remote+'/libggml.so')
 if a.probe_skel and exists!='running':t.call('push',str(a.probe_skel),remote+'/'+a.probe_skel.name)
 costs='none'
 if a.costs:t.call('push',str(a.costs),remote+'/costs.json');costs=remote+'/costs.json'
 model=a.model
 model_verification=None
 if a.model_sha256:
  model_verification=verify_model_digest(lambda command:t.call('shell',command).stdout.decode(),
                                         t.identity,model,a.model_sha256,force=a.rehash_model)
 library_dirs=[remote,t.remote_root+'/a2o_oracle_20261106-143411-669461',t.config['runtime_library_dir']]
 if a.shared_weights:
  candidates=shlex.join([path+'/libggml-hexagon.so' for path in library_dirs])
  found=t.call('shell','for f in '+candidates+'; do if [ -r "$f" ]; then sha256sum "$f"; exit; fi; done; exit 1').stdout.decode().split(maxsplit=1)
  if found[0]!=hexagon_sha256:raise RuntimeError('手机 Hexagon 动态库与构建时的权重布局来源不同')
 lib=':'.join(library_dirs+['/vendor/lib64','/vendor/lib64/egl'])
 cache_env=(' A2O_REPLAY_CACHE_DIR='+q(remote+'/replay-cache')) if a.replay_cache else ''
 verify_env='' if a.verify_kv_state is None else ' A2O_VERIFY_KV_STATE='+str(a.verify_kv_state)
 command='export LD_LIBRARY_PATH='+q(lib)+' ADSP_LIBRARY_PATH='+q(remote if a.probe_skel else t.config['runtime_library_dir'])+' GGML_HEXAGON_PROFILE='+str(a.htp_profile)+' GGML_HEXAGON_USE_HMX=1 A2O_CPU_THREADS='+str(a.cpu_threads)+' A2O_CPU_REPACK='+str(a.cpu_repack)+' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE='+str(a.opqueue)+' GGML_HEXAGON_OPBATCH='+str(a.opbatch)+' A2O_HTP_KV_OFFLOAD='+str(a.kv_offload)+' A2O_HTP_BIG_CORES='+str(a.htp_big_cores)+' A2O_FLASH_ATTN='+str(a.flash_attention)+' A2O_KV_REDUCE='+str(a.kv_reduce)+' A2O_KV_POLICY='+q(a.kv_policy)+' A2O_KV_KEEP_TOKENS='+str(a.kv_keep_tokens)+' A2O_KV_SINK_TOKENS='+str(a.kv_sink_tokens)+' A2O_KV_RECENT_TOKENS='+str(a.kv_recent_tokens)+' A2O_KV_POOL_WINDOW='+str(a.kv_pool_window)+' A2O_KV_TAIL_TOKENS='+str(a.kv_tail_tokens)+' A2O_KV_LAYER_STRIDE='+str(a.kv_layer_stride)+' A2O_KV_MIN_EVICT_TOKENS='+str(a.kv_min_evict_tokens)+' A2O_COMPACT_ANCHOR='+str(a.compact_anchor)+' A2O_KV_HWM_TRIM='+str(a.kv_hwm_trim)+' A2O_MEMORY_LEDGER='+str(int(a.memory_ledger))+' A2O_M1_EXPLORER='+str(a.m1_explorer)+' A2O_DEVICE='+q(a.device)+' A2O_GAP_MS='+str(a.gap_ms)+' A2O_EXPECTED_DECODE='+str(a.expected_decode)+' A2O_SINGLE_RESIDENT='+str(int(a.single_resident))+' A2O_SHARED_WEIGHTS='+str(int(a.shared_weights))+' A2O_MODEL_ID='+q(a.model_id)+cache_env+verify_env+'; exec '+shlex.join([remote+'/service',model,a.condition,str(a.context_size),'18780',costs])
 launch='nohup /system/bin/sh -c '+q(command)+' > '+q(remote+'/runtime.log')+' 2>&1 < /dev/null & echo $! > '+q(remote+'/pid')
 try:
  require_phone_preflight(
   t, out, attached=exists=='running', expected_owner=out.name, call=t.call,
   max_quiet_c=a.max_quiet_c, min_prime_khz=a.min_prime_khz,
   min_mem_avail_kib=(None if exists=='running' else a.min_mem_avail_kib),
   cooldown_timeout_s=a.cooldown_timeout_s,
  )
 except PhonePreflightError:
  if exists!='running':
   try:
    t.call('shell',owner_check+' && PATH=/data/data/com.termux/files/usr/bin:"$PATH" termux-wake-unlock')
   finally:
    t.release_lock()
  raise
 t.call('shell',owner_check+' && { if '+pid_alive+'; then :; else sleep 0.2; '+launch+'; fi; }')
 meta_header=dict(git_commit=build_meta.get('git_commit') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),build_id=build_meta.get('build_id'),device=a.device,model=a.model_id,shared_weights=a.shared_weights,compact_anchor=(bool(a.compact_anchor) if a.kv_reduce else False),kv_hwm_trim=bool(a.kv_hwm_trim),memory_ledger=a.memory_ledger,kv_layer_stride=a.kv_layer_stride,kv_min_evict_tokens=a.kv_min_evict_tokens,gap_ms=a.gap_ms)
 print('A2O_META',json.dumps(meta_header,ensure_ascii=False),flush=True)
 manifest=dict(**meta_header,condition=a.condition,runtime_overlay=overlay,hexagon_sha256=hexagon_sha256,remote=remote,command=command,context_size=a.context_size,replay_cache=a.replay_cache,cpu_threads=a.cpu_threads,cpu_threads_batch=a.cpu_threads,cpu_repack=bool(a.cpu_repack),htp_host_threads=4,htp_profile=a.htp_profile,htp_big_cores=bool(a.htp_big_cores),probe_skel=str(a.probe_skel),batch_size=128,single_resident=a.single_resident,model_path=model,model_id=a.model_id,model_sha256=a.model_sha256,model_verification=model_verification,requested_verify_kv_state=a.verify_kv_state,opbatch=a.opbatch,opqueue=a.opqueue,flash_attention=bool(a.flash_attention),kv_offload=bool(a.kv_offload),kv_policy=(a.kv_policy if a.kv_reduce else 'none'),kv_reduce=bool(a.kv_reduce),kv_keep_tokens=(a.kv_keep_tokens if a.kv_reduce else None),kv_sink_tokens=(a.kv_sink_tokens if a.kv_reduce else None),kv_recent_tokens=(a.kv_recent_tokens if a.kv_reduce else None),kv_pool_window=(a.kv_pool_window if a.kv_reduce else None),kv_tail_tokens=(a.kv_tail_tokens if a.kv_reduce else None),m1_explorer=bool(a.m1_explorer),expected_decode_tokens=a.expected_decode,kv_costs=(str(a.kv_costs) if a.kv_costs else None),route_costs=(str(a.route_costs) if a.route_costs else None),cold_costs=(str(a.cold_costs) if a.cold_costs else None),pre_move=False,costs=str(a.costs))
 manifest_path=out/'manifest.json'
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
 text_busy_retried=False
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  try:
   with urllib.request.urlopen(BASE+'/health',timeout=3) as r:health=json.load(r)
   if health.get('shared_weights',False)!=a.shared_weights:raise RuntimeError('共享权重配置不一致')
   if health.get('cpu_threads',4)!=a.cpu_threads:raise RuntimeError('CPU线程数与请求不一致')
   if 'model' in health and health['model']!=a.model_id:raise RuntimeError('模型身份与请求不一致')
   if 'htp_kv_offload' in health and health['htp_kv_offload']!=bool(a.kv_offload):raise RuntimeError('KV offload配置不一致')
   if a.verify_kv_state is not None and health.get('verify_kv_state') is not bool(a.verify_kv_state):
    raise RuntimeError('服务不支持请求的 KV 核验模式，或启动配置不一致')
   manifest['verify_kv_state']=health.get('verify_kv_state')
   manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
   if a.memory_ledger:
    if health.get('memory_ledger_version')!=1 or health.get('kv_hwm_trim') is not bool(a.kv_hwm_trim):
     raise RuntimeError('内存账本或 KV HWM trim 配置不一致')
    with urllib.request.urlopen(BASE+'/a2o/memory-ledger',timeout=10) as r:
     ledger=json.load(r)
    (out/'startup-memory.json').write_text(json.dumps(ledger,indent=2)+'\n')
   cost_payload={}
   if a.cold_costs:cost_payload['cold_model']=json.loads(a.cold_costs.read_text())
   if a.kv_costs:cost_payload['kv_model']=json.loads(a.kv_costs.read_text())
   if a.route_costs:
    rc=json.loads(a.route_costs.read_text())
    cost_payload['model']=rc['model'] if ('model' in rc and isinstance(rc['model'],dict)) else rc
   if cost_payload:
    req=urllib.request.Request(BASE+'/a2o/costs',data=json.dumps(cost_payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=15) as kr:health['cost_install']=json.load(kr)
    if a.cold_costs:
     if not health['cost_install'].get('cold_cost_loaded'):raise RuntimeError('首轮选路成本安装失败')
     health['cold_cost_loaded']=True
   (out/'health.json').write_text(json.dumps(health));print(health);break
  except urllib.error.HTTPError:
   raise
  except OSError:
   # 仅网络等待不消耗初始化预算；SSH连通后继续检查服务。
   before=time.monotonic()
   retry(lambda:original_call('get-state'))
   state=t.call('shell','if '+pid_alive+'; then echo running; else echo stopped; fi').stdout.decode().strip()
   if state!='running':
    rlog=t.call('shell','cat '+q(remote+'/runtime.log')+' 2>/dev/null || true').stdout.decode(errors='replace')
    if not text_busy_retried and 'Text file busy' in rlog:
     text_busy_retried=True
     time.sleep(1)
     t.call('shell',owner_check+' && { if '+pid_alive+'; then :; else '+launch+'; fi; }')
     deadline+=time.monotonic()-before
     continue
    raise RuntimeError('私有服务已退出，检查runtime.log；锁与失败现场保留')
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
 try:
  if a.logs_only:
   device=out/'device';device.mkdir(exist_ok=True)
   downloaded,missing=[],[]
   for name in ('runtime.log','pid'):
    source=remote+'/'+name
    present=t.call('shell','if [ -f '+q(source)+' ]; then echo yes; fi').stdout.decode().strip()
    if present=='yes':
     t.call('pull',source,str(device/name));downloaded.append(name)
    else:missing.append(name)
   (out/'remote-artifacts.json').write_text(json.dumps(dict(remote=remote,replay_cache=remote+'/replay-cache',retained_on_phone=True,downloaded=downloaded,missing=missing),indent=2)+'\n')
  elif t.call('shell','if [ -d '+q(remote)+' ]; then echo yes; fi').stdout.decode().strip()=='yes':
   t.call('pull',remote+'/.',str(out/'device'))
 finally:
  # Termux's am wrapper works from the SSH app UID. Android's am does not.
  try:t.call('shell','if '+owner_check+'; then PATH=/data/data/com.termux/files/usr/bin:"$PATH" termux-wake-unlock; fi')
  finally:t.release_lock()
