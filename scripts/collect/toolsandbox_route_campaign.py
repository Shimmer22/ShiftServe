#!/usr/bin/env python3
"""串行完成已授权的CPU/HTP/route十题比较，失败停在现场。"""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.workloads.toolsandbox import post,save
from a2o.runtime.phone_preflight import docker_preflight_args
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--start-from',choices=['htp','route']);a=p.parse_args()
out=ROOT/'results/20261107-toolsandbox-route-e2e';remote='/home/wins/ai-workspace/outputs/a2o-gap-profiling';workspace='/workspace/projects/a2o-gap-profiling';python='/workspace/tools/a2o-toolsandbox-py311-env/bin/python';endpoint='http://10.29.131.50:18780'
def run(cmd,name):
 with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def phone(action,condition,folder):
 cmd=[sys.executable,'scripts/run/route_phone.py',action,'--condition',condition,'--output',str(out/folder)]
 if action=='start':cmd+=['--binary',str(a.binary),'--registry',str(a.registry)]
 run(cmd,folder+'-'+action)
def collect(condition,name,owner):
 gate=shlex.join(docker_preflight_args(ROOT,out/'preflight',expected_owner=owner))
 run(['ssh','wins',f'docker exec {gate} a2o-toolsandbox-py311 {python} {workspace}/scripts/collect/toolsandbox.py --config {workspace}/configs/toolsandbox-qwen35-ten-{condition}.json --output /workspace/outputs/a2o-gap-profiling/{name}'],name)
 run(['rsync','-a','wins:'+remote+'/'+name+'/',str(out/name)+'/'],name+'-download')
 s=json.loads((out/name/'summary.json').read_text())
 if s['completed']!=10:raise RuntimeError('正式任务未全部尝试：'+name)
 if s['trace_complete']!=10:save(out/(name+'-quality-failures.json'),dict(trace_complete=s['trace_complete'],completed=s['completed'],failed=[r for r in s['results'] if not r['trace_complete']]))
 run([sys.executable,'scripts/analyze/toolsandbox.py',str(out/name)],name+'-analysis')
try:
 if not a.resume:
  # 首轮CPU正在运行，等待其完成后才切换服务，不与之争抢设备。
  while True:
   r=subprocess.run(['ssh','wins','test -f '+remote+'/20261107-route-e2e-cpu-ten/completion.json'],capture_output=True)
   if r.returncode==0:break
   time.sleep(10)
  run(['rsync','-a','wins:'+remote+'/20261107-route-e2e-cpu-ten/',str(out/'cpu-bootstrap')+'/'],'bootstrap-download')
  phone('stop','cpu','service-cpu-v2')
  # 校准和正式轮使用最终同一二进制。校准数据独立于正式策略轮。
  phone('start','route','service-calibration')
  run([sys.executable,'scripts/collect/route_calibration.py','--cpu-run',str(out/'cpu-bootstrap'),'--output',str(out/'calibration')],'calibration')
  phone('stop','route','service-calibration')
 for condition in (['route'] if a.start_from=='route' else ['htp','route'] if a.resume else ['cpu','htp','route']):
  folder='service-formal-'+condition+('-resume' if a.resume else '');phone('start',condition,folder)
  if condition=='route':
   costs=json.loads((out/'calibration/costs.json').read_text());save(out/'route-costs-installed.json',post(endpoint,'/a2o/costs',costs))
  gate=shlex.join(docker_preflight_args(ROOT,out/'preflight',expected_owner=folder))
  run(['ssh','wins',f'docker exec {gate} a2o-toolsandbox-py311 {python} {workspace}/scripts/validate/phone_server.py --config {workspace}/configs/toolsandbox-qwen35-ten-{condition}.json --output /workspace/outputs/a2o-gap-profiling/20261107-route-e2e-{condition}-validation-final'],condition+'-validation')
  collect(condition,'20261107-route-e2e-'+condition+'-formal-ten',folder)
  phone('stop',condition,folder)
 run([sys.executable,'scripts/analyze/toolsandbox_route.py',str(out)],'final-analysis')
 save(out/'campaign-completion.json',dict(completed=True,conditions=3,scenarios=10))
except BaseException as e:
 save(out/'campaign-failure.json',dict(error=repr(e),note='现场及锁保留，检查日志后恢复；不静默回退'))
 raise
