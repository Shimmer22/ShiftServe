"""Run on wins: durable paired warm replay smoke, no trajectory generation."""
import json,time,urllib.request,traceback,fcntl
lock=open("/home/wins/ai-workspace/private/a2o-text-pilot/paired-retest.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from a2o.runtime.phone_preflight import require_attached_phone_preflight
base=Path('/home/wins/ai-workspace/private/a2o-text-pilot')
out=Path('/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261108-bfcl-text-pilot/phone-frozen-retest');out.mkdir(exist_ok=True)
def post(path,b):
 req=urllib.request.Request('http://10.29.131.50:18780'+path,json.dumps(b).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=2400) as q:return json.load(q)
def status(**kw):
 t=out/'status.tmp';t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2));t.replace(out/'status.json')
try:
 status(phase='preflight')
 require_attached_phone_preflight(ROOT, out)
 status(phase='waiting_service')
 for i in range(120):
  try:
   with urllib.request.urlopen('http://10.29.131.50:18780/health',timeout=3) as q:h=json.load(q)
   if h['condition']=='route':break
  except Exception:pass
  time.sleep(2)
 else:raise RuntimeError('route service unavailable')
 model=json.loads((base/'cost-model.json').read_text());post('/a2o/costs',{'model':model.get('model',model)})
 requests=json.loads((base/'retest-requests.json').read_text());start=time.time()
 for i,b in enumerate(requests):
  if (out/f'{i:02d}.json').exists():continue
  status(phase='running',completed=i,total=len(requests),current={k:b[k] for k in ['task','step','H','P','D']},elapsed_s=time.time()-start)
  d=post('/a2o/replay',b);(out/f'{i:02d}.json').write_text(json.dumps(dict(request={k:b[k] for k in ['task','step','H','P','D']},response=d),indent=2))
  assert len(d['results'])==3 and all(x['H']==b['H'] and x['P']==b['P'] and x['D']==b['D'] for x in d['results'])
 status(phase='complete',completed=len(requests),elapsed_s=time.time()-start)
except Exception:
 status(phase='failed',error=traceback.format_exc())
 raise
