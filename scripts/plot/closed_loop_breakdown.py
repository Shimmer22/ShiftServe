import json, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path('results/20261107-toolsandbox-full'); OUT=ROOT/'closed-loop-breakdown'; OUT.mkdir(exist_ok=True)
conds=['cpu','htp','route']; labels=['CPU','HTP','Route +\nalways prepare']
stages=['Observation','Blocking preparation','Prefill','State handoff','Decode','Other']
colors={'Observation':'#bdbdbd','Blocking preparation':'#cc79a7','Prefill':'#56b4e9','State handoff':'#e69f00','Decode':'#009e73','Other':'#7f7f7f'}
common=json.load(open(ROOT/'first-round-summary.json'))['common_complete']
raw={c:{k:0.0 for k in stages} for c in conds}; excl={c:{k:0.0 for k in stages} for c in conds}
ns={c:0 for c in conds}
for c in conds:
  for name in common:
    p=ROOT/c/name/'metrics.json'
    if not p.exists(): continue
    d=json.load(open(p)); inf=d.get('inference',[]); tools=d.get('tools',[])
    if not inf: continue
    ns[c]+=1
    for j,x in enumerate(inf):
      bt=x.get('backend_timings') or {}
      host=float(x.get('host_request_ms',0)); mp=float(x.get('measurement_preparation_ms',0))
      vals={
        'Observation':float(bt.get('observation_ms',0)),
        'Blocking preparation':mp+float(bt.get('prepare_ms',0)),
        'Prefill':float(bt.get('prompt_ms',0)),
        'State handoff':float(bt.get('restore_ms',0))+float(bt.get('handoff_ms',0))+float(bt.get('snapshot_ms',0)),
        'Decode':float(bt.get('predicted_ms',0)),
      }
      vals['Other']=max(0.0,host-sum(vals.values()))
      for k,v in vals.items():
        raw[c][k]+=v
        # Right panel: remove first request and common observation/host residual.
        if j>0:
          if k not in ('Observation','Other'): excl[c][k]+=v
    # tool execution/result gap is part of raw observation; excluded as common observation
    raw[c]['Observation'] += sum(float(t.get('execution_ms',0))+float(t.get('G_ms',0)) for t in tools)

def totals(d): return {c:sum(d[c].values()) for c in conds}
# output machine-readable values
json.dump({'n':ns,'raw_total_s':{c:sum(raw[c].values())/1000 for c in conds},'excluded_total_s':{c:sum(excl[c].values())/1000 for c in conds},'raw_stage_s':{c:{k:v/1000 for k,v in raw[c].items()} for c in conds},'excluded_stage_s':{c:{k:v/1000 for k,v in excl[c].items()} for c in conds}},open(OUT/'data.json','w'),indent=2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8})
fig,axs=plt.subplots(1,2,figsize=(10.5,4.9),sharey=False,constrained_layout=True)
for ax, data, title in [(axs[0],raw,'Original closed loop'),(axs[1],excl,'After removing first request\nand common host overhead')]:
  x=np.arange(3); bottom=np.zeros(3)
  for st in stages:
    vals=np.array([data[c][st]/1000 for c in conds])
    if np.all(vals==0): continue
    ax.bar(x,vals,bottom=bottom,color=colors[st],width=.62,edgecolor='white',linewidth=.4,label=st)
    bottom+=vals
  ax.set_xticks(x,labels); ax.set_title(title)
  ax.set_ylabel('Accumulated time over 63 paired tasks (s)')
  ax.grid(axis='y',color='#dddddd',linewidth=.6); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
  for i,c in enumerate(conds): ax.text(i,bottom[i]*1.015,f'{bottom[i]:.0f}',ha='center',va='bottom',fontsize=8)
  ax.set_ylim(0,max(bottom)*1.16)
handles=[Patch(facecolor=colors[s],edgecolor='none',label=s) for s in stages]
fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.02),frameon=False)
fig.suptitle('True closed-loop time decomposition',y=1.08,fontsize=12)
fig.savefig(OUT/'closed-loop-breakdown.png',dpi=300,bbox_inches='tight'); fig.savefig(OUT/'closed-loop-breakdown.pdf',bbox_inches='tight')
print(json.dumps({'n':ns,'raw_total_s':{c:round(sum(raw[c].values())/1000,1) for c in conds},'excluded_total_s':{c:round(sum(excl[c].values())/1000,1) for c in conds}},indent=2))
