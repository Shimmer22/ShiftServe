#!/usr/bin/env python3
"""只读校准快照：分项拟合、边界和实测中位数表。不会安装未完成模型。"""
import csv,json,sys,importlib.util
from pathlib import Path
from statistics import median
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.scheduling.linear_route import fit_model,threshold,FIELDS,predict
spec=importlib.util.spec_from_file_location('figure_style',Path.home()/'.codex/skills/figure-style/kernel.py');style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
style.apply_figure_style(sizes=(11,10,9))
source=ROOT/'results/20261107-toolsandbox-full/linear-routing/calibration'
out=source.parent/'fit-preview';out.mkdir(exist_ok=True)
responses=[json.loads(p.read_text()) for p in sorted(source.glob('*.response.json'))]
model=fit_model(responses)
json.dump(dict(model=model,points=len(responses),provisional=len(responses)<15,responses=responses),open(out/'snapshot.json','w'),indent=2)
colors={128:'#0072B2',768:'#E69F00',1536:'#009E73'}
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
names={'cpu_restore':'CPU state restore R(H)','cpu_to_htp':'Prepare NPU state U(H)','cpu_prefill':'CPU prefill C(P)','htp_prefill':'NPU prefill N(P)','htp_to_cpu':'Return state to CPU M(H+P)'}
for ax,(key,(coord,metric)) in zip(axes.flat,FIELDS.items()):
 xs=[]
 for r in responses:
  x=r['H']+r['P'] if coord=='HP' else r[coord];ys=[row[metric] for row in r['rows']];xs.append(x)
  ax.scatter([x]*len(ys),ys,s=17,alpha=.35,color=colors[r['H']])
  ax.scatter(x,median(ys),s=48,marker='D',edgecolors='white',linewidth=.7,color=colors[r['H']],zorder=3)
 grid=np.linspace(min(xs),max(xs),200);fit=model[key]
 ax.plot(grid,fit['intercept_ms']+fit['slope_ms_per_token']*grid,color='#444444',label='Affine fit')
 ax.set(title=names[key],xlabel={'H':'History H (tokens)','P':'Append P (tokens)','HP':'Updated history H+P (tokens)'}[coord],ylabel='Time (ms)')
 ax.text(.04,.94,f"RMSE {fit['rmse_ms']:.1f} ms; R² {fit['r2']:.3f}",transform=ax.transAxes,va='top',fontsize=10)
 ax.margins(.10)
ax=axes.flat[-1]
for h in sorted(set(r['H'] for r in responses)):
 ps=np.linspace(0,384,385);delta=[predict(model,h,p)['cc_ms']-predict(model,h,p)['hc_ms'] for p in ps]
 ax.plot(ps,delta,color=colors[h],label=f'H={h}')
ax.axhline(0,color='#777777',linestyle='--');ax.set(title='Predicted gain from choosing NPU',xlabel='Append P (tokens)',ylabel='CPU − NPU path time (ms)');ax.legend(loc='upper left')
fig.suptitle(f'PROVISIONAL offline fits: {len(responses)}/15 grid points, 3 repeats per point',fontsize=11)
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker='o',linestyle='',color=colors[h],label=f'H={h}') for h in sorted(set(r['H'] for r in responses))]
handles += [Line2D([],[],marker='D',linestyle='',color='#444444',label='Median'),Line2D([],[],marker='o',linestyle='',color='#aaaaaa',label='Raw repeat')]
fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),fontsize=10)
fig.savefig(out/'fits.png',dpi=180);fig.savefig(out/'fits.pdf')
# Check non-tick label collisions and bounds after rendering.
fig.canvas.draw();renderer=fig.canvas.get_renderer();warnings=[]
for i,ax in enumerate(axes.flat):
 texts=[ax.title,ax.xaxis.label,ax.yaxis.label]+list(ax.texts)
 for j,t in enumerate(texts):
  if not t.get_text():continue
  box=t.get_window_extent(renderer)
  if not fig.bbox.contains(box.x0,box.y0) or not fig.bbox.contains(box.x1,box.y1):warnings.append(f'panel {i}: label outside')
  for u in texts[j+1:]:
   if u.get_text() and box.overlaps(u.get_window_extent(renderer)):warnings.append(f'panel {i}: label overlap')
(out/'layout-check.json').write_text(json.dumps(warnings))
rows=[]
for r in responses:
 row=dict(H=r['H'],P=r['P'],**{k:median(x[field] for x in r['rows']) for k,(coord,field) in FIELDS.items()})
 row.update(predict(model,r['H'],r['P']));rows.append(row)
with (out/'observations.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(points=len(responses),fits={k:model[k] for k in FIELDS},thresholds=[threshold(model,h) for h in [128,768,1536]],rows=rows,layout=warnings),indent=2))
