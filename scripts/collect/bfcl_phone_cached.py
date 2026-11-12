#!/usr/bin/env python3
"""Direct phone frozen replay; persistent fixture cache, resumable rounds/order rotation."""
import argparse,fcntl,hashlib,json,time,urllib.request,urllib.error,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.runtime.network_retry import is_network_error,retry_network

def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--requests',required=True,type=Path);p.add_argument('--cost-model',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path);p.add_argument('--endpoint',default='http://10.29.131.50:18780')
 p.add_argument('--rounds',type=int,default=1);p.add_argument('--cache-only',action='store_true')
 p.add_argument('--backup-command-json',help='首轮完整落盘后串行运行的缓存备份命令参数数组')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 lock=(a.output/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 requests=json.loads(a.requests.read_text());model=json.loads(a.cost_model.read_text());model=model.get('model',model)
 assert a.rounds>0
 for b in requests:assert 0<b['H']<len(b['prompt_ids']) and len(b['prompt_ids'])-b['H']==b['P'] and len(b['generated_ids'])==b['D']
 def send(path,b):
  req=urllib.request.Request(a.endpoint+path,json.dumps(b).encode(),{'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=2400) as q:return json.load(q)
  except urllib.error.HTTPError as e:raise RuntimeError(e.read().decode()) from e
 def get_health():
  with urllib.request.urlopen(a.endpoint+'/health',timeout=2400) as q:return json.load(q)
 health=retry_network(get_health)
 def post(path,b):
  while True:
   try:return send(path,b)
   except Exception as exc:
    if not is_network_error(exc):raise
    # Lost HTTP responses do not cancel native evaluation. Wait for the same
    # service mutex to become idle before retrying this frozen fixture.
    with (a.output/'network-events.jsonl').open('a') as f:
     f.write(json.dumps(dict(time=time.time(),path=path,error=str(exc),H=b.get('H'),P=b.get('P'),D=b.get('D')))+'\n')
    print('连接中断，等待同一服务完成当前请求后重试冻结边界：'+str(exc),flush=True)
    ready=retry_network(get_health)
    if any(ready.get(k)!=health.get(k) for k in ['build_id','context_size','condition','replay_cache_version']):
     raise RuntimeError('service identity changed after network interruption') from exc
 if health.get('replay_cache_version')!=2 or health.get('condition')!='route':
  raise RuntimeError('requires route service with replay_cache_version=2; refusing legacy service')
 assert all(len(b['prompt_ids'])+b['D']<health['context_size'] for b in requests),'context overflow'
 identity=hashlib.sha256(json.dumps(dict(requests=requests,model=model,rounds=a.rounds,cache_only=a.cache_only,health={k:health[k] for k in ['build_id','context_size','condition','replay_cache_version']},endpoint=a.endpoint),sort_keys=True).encode()).hexdigest()
 manifest=a.output/'manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())['id']==identity,'refuse mixing experiment configurations'
 else:write(manifest,dict(id=identity,endpoint=a.endpoint,health=health,rounds=a.rounds,cache_only=a.cache_only,fixture='outside timing; prefix cache requires A2O_REPLAY_CACHE_DIR on phone',order='trajectory-grouped first pass; rotate condition order by boundary and round'))
 def status(**kw):write(a.output/'status.json',dict(updated_unix=time.time(),**kw))
 completed=0;start=time.time()
 try:
  post('/a2o/costs',{'model':model})
  # Exact-prefix extension is safe only when tokens actually match.
  order=sorted(range(len(requests)),key=lambda i:(requests[i]['task'],requests[i]['step']))
  for rep in range(a.rounds):
   for j,i in enumerate(order):
    b=dict(requests[i]);shift=(j+rep)%3;conditions=['cpu','htp','route'];b['condition_order']=conditions[shift:]+conditions[:shift];b['cache_only']=a.cache_only
    dest=a.output/f'r{rep:02d}-b{i:03d}.json'
    if dest.exists():
     saved=json.loads(dest.read_text());assert saved['request']==b
     completed+=1;continue
    status(phase='running',completed=completed,total=a.rounds*len(requests),round=rep,boundary=i,H=b['H'],P=b['P'],D=b['D'])
    result=post('/a2o/replay',b)
    # Preserve responses even when semantic validation fails.
    raw=dest.with_suffix('.raw.json');write(raw,dict(request=b,response=result))
    assert result['cache']['prefix_tokens']==b['H']
    if not a.cache_only:
     assert [x['condition'] for x in result['results']]==b['condition_order']
     assert all(all(x[k]==b[k] for k in ['H','P','D']) for x in result['results'])
    else:assert result['mode']=='cache_only' and result['H']==b['H']
    raw.replace(dest);completed+=1
   if rep==0 and a.backup_command_json:
    import subprocess
    status(phase='backing_up_cache',completed=completed,total=a.rounds*len(requests))
    subprocess.run(json.loads(a.backup_command_json),check=True)
  status(phase='complete',completed=completed,elapsed_s=time.time()-start)
 except Exception as e:
  status(phase='failed',completed=completed,error=str(e));raise
if __name__=='__main__':main()
