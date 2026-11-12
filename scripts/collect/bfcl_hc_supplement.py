#!/usr/bin/env python3
"""补齐冻结边界的小P强制HC路径；同轮重测CC，保留原始三条件响应。"""
import fcntl,hashlib,json,shlex,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network
OUT=ROOT/'results/20261110-bfcl-small-p-hc-supplement';OUT.mkdir(parents=True,exist_ok=True)
def save(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def status(phase,**kw):save('status.json',dict(phase=phase,updated_unix=time.time(),**kw))
def main():
 lock=(OUT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 t=DeviceTransport(ROOT,'ssh',config=ROOT/'configs/termux-ssh.json');q=shlex.quote
 service=OUT/(OUT.name+'-service');remote=t.remote_root+'/'+service.name
 def ssh(cmd):return retry_network(lambda:t.call('shell',cmd)).stdout.decode()
 binary=ROOT/'results/20261107-toolsandbox-route-e2e/build-service-165732/agent'
 old=ROOT/'results/20261109-bfcl-cached-repeats'
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==json.loads((old/'plan.json').read_text())['binary_sha256']
 requests=[]
 for f in sorted((old/'measurements').glob('r00-b*.json')):
  d=json.loads(f.read_text());b=d['request']
  if b['P']<=39:
   b={k:v for k,v in b.items() if k not in ('condition_order','cache_only')};b['original_boundary']=f.stem.split('-')[1];requests.append(b)
 assert len(requests)==15
 save('requests.json',requests)
 original=json.loads((ROOT/'results/20261108-bfcl-text-pilot/phone-frozen-retest/cost-model.json').read_text())
 model=json.loads(json.dumps(original['model']))
 model['routing_p_min']=0
 model['cpu_restore'].update(intercept_ms=1e12,slope_ms_per_token=0)
 model['cpu_prefill'].update(intercept_ms=1e12,slope_ms_per_token=0)
 save('forced-hc-costs.json',dict(model=model,scope='仅强制HC测量，不是拟合模型，不用于部署',original_model=original))
 status('starting_service',boundaries=15,rounds=3)
 with (OUT/'service-launch.log').open('a') as log:
  subprocess.run([sys.executable,str(ROOT/'scripts/run/route_phone.py'),'start','--condition','route','--binary',str(binary),'--output',str(service),'--context-size','8192','--replay-cache','--retry-network'],check=True,stdout=log,stderr=subprocess.STDOUT)
 owner='test "$(cat '+q(t.remote_root+'/model.lock/owner')+')" = '+q(service.name)
 cache=remote+'/replay-cache';oldcache=t.remote_root+'/20261109-bfcl-cached-repeats-service/replay-cache'
 ssh(owner+' && test ! -e '+q(cache)+' && ln -s '+q(oldcache)+' '+q(cache))
 save('plan.json',dict(boundaries=15,rounds=3,condition_evaluations=135,scope='强制HC与同轮CC；原H/P/D完整不变；三条件轮换；准备及夹具恢复不计时；teacher forcing不测质量',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cache=oldcache))
 status('collecting',boundaries=15,rounds=3)
 with (OUT/'collector.log').open('a') as log:
  subprocess.run([sys.executable,str(ROOT/'scripts/collect/bfcl_phone_cached.py'),'--requests',str(OUT/'requests.json'),'--cost-model',str(OUT/'forced-hc-costs.json'),'--output',str(OUT/'measurements'),'--rounds','3'],check=True,stdout=log,stderr=subprocess.STDOUT)
 records=[json.loads(f.read_text()) for f in (OUT/'measurements').glob('r*-b*.json') if '.raw.' not in f.name]
 assert len(records)==45
 assert all(next(x for x in d['response']['results'] if x['condition']=='route')['route']=='htp' for d in records)
 status('stopping_service',completed=45)
 pid=int(ssh('cat '+q(remote+'/pid')).strip());ticks=ssh("awk '{print $22}' /proc/"+str(pid)+'/stat').strip()
 guard=owner+' && test "$(awk \'{print $22}\' /proc/'+str(pid)+'/stat)" = '+q(ticks)+' && tr "\\000" " " < /proc/'+str(pid)+'/cmdline | /system/bin/grep -F '+q(remote+'/service')
 ssh(guard+' && kill '+str(pid))
 for _ in range(30):
  if ssh('if kill -0 '+str(pid)+' 2>/dev/null; then echo alive; else echo stopped; fi').strip()=='stopped':break
  time.sleep(1)
 else:raise RuntimeError('服务未退出，保留锁')
 t.call('pull',remote+'/runtime.log',str(OUT/'runtime.log'));t.lock_owner=service.name;t.release_lock()
 subprocess.run(t.prefix+['termux-wake-unlock'],check=True,capture_output=True)
 save('summary.json',dict(boundaries=15,rounds=3,completed=45,forced_hc_verified=True))
 status('complete',completed=45)
if __name__=='__main__':
 try:main()
 except Exception:
  status('failed',error=traceback.format_exc());raise
