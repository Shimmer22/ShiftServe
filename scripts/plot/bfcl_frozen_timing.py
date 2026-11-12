"""Plot fixed-trajectory measurements; no bootstrap/preparation in timed totals."""
import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0,'/Users/hui/.codex/skills/figure-style')
from kernel import apply_figure_style
apply_figure_style(sizes=(10,9,8))
r=Path('results/20261108-bfcl-text-pilot/phone-frozen-paired');out=r/'figures';out.mkdir(exist_ok=True)
data=[json.load(open(r/f'{i:02d}.json')) for i in range(6)]
conds=['cpu','htp','route'];names=['CPU','HTP','Route'];stages=[('prefill_ms','Prefill','#56b4e9'),('handoff_ms','State handoff','#e69f00'),('decode_ms','Decode','#009e73')]
fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2,height_ratios=[1.12,1],hspace=.5,wspace=.28)
ax=fig.add_subplot(gs[0,:]);agg=fig.add_subplot(gs[1,0]);small=fig.add_subplot(gs[1,1])
fig.subplots_adjust(top=.88,bottom=.1,left=.07,right=.98)
width=.23;x=np.arange(6)
for j,c in enumerate(conds):
 bottom=np.zeros(6)
 for key,label,color in stages:
  vals=np.array([d['response']['results'][j][key]/1000 for d in data]);ax.bar(x+(j-1)*width,vals,width,bottom=bottom,color=color,edgecolor='white',linewidth=.3);bottom+=vals
 for i in range(6):ax.text(x[i]+(j-1)*width,-4,names[j],rotation=55,ha='right',va='top',fontsize=8)
ax.set_xticks(x,[f"P={d['request']['P']}  D={d['request']['D']}\nH={d['request']['H']}" for d in data]);ax.tick_params(axis='x',pad=35,length=0)
ax.set_ylabel('Post-tool latency, preparation fully hidden (s)');ax.set_title('a   Six frozen boundaries: identical H, P and D within each group',loc='left',pad=12);ax.set_ylim(0,110)
# aggregate
bottom=np.zeros(3)
for key,label,color in stages:
 vals=np.array([sum(d['response']['results'][j][key] for d in data)/1000 for j in range(3)])
 agg.bar(np.arange(3),vals,.58,bottom=bottom,color=color,edgecolor='white',linewidth=.3);bottom+=vals
for j in range(3):agg.text(j,bottom[j]+5,f'{bottom[j]:.2f} s',ha='center',fontsize=9)
agg.set_xticks(range(3),names);agg.set_ylim(0,425);agg.set_ylabel('Accumulated latency (s)');agg.set_title('b   Accumulated over six boundaries',loc='left',pad=12)
# Tool-slack sensitivity, not measured overlap. Only Route migrates prefix.
slack=np.linspace(0,.6,241)
prep=np.array([d['response']['results'][2]['prepared_before_measurement_ms']/1000 for d in data])
route_base=sum(d['response']['results'][2][k] for d in data for k,_,_ in stages)/1000
stall=np.maximum(prep[:,None]-slack[None,:],0).sum(axis=0)
small.plot(slack,stall,color='#cc79a7',linewidth=2)
small.axvline(prep.max(),color='#666',linestyle='--',linewidth=.8)
small.text(prep.max()+.01,.9,f'All preparation hidden\nG ≥ {prep.max():.3f} s',fontsize=9,va='top')
small.set_xlabel('Assumed tool slack G at each boundary (s)')
small.set_ylabel('Accumulated unfinished preparation (s)')
small.set_title('c   Preparation work left after tool slack (scenario)',loc='left',pad=12)
small.set_ylim(-.03,max(stall)*1.15)
small.text(.98,.96,'Scenario: Σ max(Tprepare − G, 0)',transform=small.transAxes,ha='right',va='top',fontsize=9)
for a in [ax,agg,small]:a.grid(axis='y',color='#e4e4e4',linewidth=.5);a.set_axisbelow(True)
handles=[Patch(facecolor=co,label=la) for _,la,co in stages]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.985),ncol=3,columnspacing=1.4,handlelength=1.8)
for ext in ['png','pdf','svg']:fig.savefig(out/f'frozen-timing.{ext}',dpi=300)
with open(out/'stage-times.csv','w') as f:
 w=csv.writer(f);w.writerow(['task','step','condition','route','H','P','D','restore_s','prefill_s','handoff_s','decode_s','timed_total_s','excluded_prepare_s'])
 for d in data:
  b=d['request']
  for z in d['response']['results']:w.writerow([b['task'],b['step'],z['condition'],z['route'],b['H'],b['P'],b['D']]+[z[k]/1000 for k in ['restore_ms','prefill_ms','handoff_ms','decode_ms','total_ms','prepared_before_measurement_ms']])
(out/'caption.md').write_text('Resident-prefix accounting, derived from measured frozen replay. Remove benchmark reset/restore and prefix bootstrap for CPU, HTP and Route. Retain genuine HTP→CPU handoff in Route. Panels a,b assume Route preparation fully hidden by tool slack; totals are sums of prefill+handoff+decode, not measured live E2E. Panel c is a sensitivity scenario using each measured Route preparation time: exposed stall=sum(max(Tprepare_i-G,0)); same assumed G per boundary. Preparation was synchronous before measurement, not a measured asynchronous tool-slack overlap. Unfinished preparation is not necessarily blocking latency: a ready-only router may fall back to CPU. Fixed order, single observation per condition; two tasks, six boundaries, H beyond calibration domain.\n')
print(out/'frozen-timing.png')
print('Route prepare seconds:',prep.tolist(),'hide-all slack',prep.max())
print('resident totals:',[sum(d['response']['results'][j][k] for d in data for k,_,_ in stages)/1000 for j in range(3)])
