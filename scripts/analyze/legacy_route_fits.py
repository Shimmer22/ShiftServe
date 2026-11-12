#!/usr/bin/env python3
"""旧20点离线校准：总路径成本拟合，质量失败点明确标出。"""
import sys,json,csv,importlib.util
from pathlib import Path
from statistics import median
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('style',Path.home()/'.codex/skills/figure-style/kernel.py');style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style);style.apply_figure_style(sizes=(11,10,9))
base=ROOT/'results/20261107-toolsandbox-route-e2e/calibration';out=base/'fit-preview';out.mkdir(exist_ok=True)
rows=[]
for p in sorted(base.glob('*.response.json')):
 r=json.loads(p.read_text());rows.append(dict(H=r['H'],P=r['P'],passed=all(x['passed'] for x in r['rows']),**{key:median(x[key] for x in r['rows']) for key in ['cc_ms','hc_ms','cc_restore_ms','hc_prepare_ms']}))
# Fit only the 19 accepted table points. Show the excluded point but never use it in fit statistics.
accepted=[r for r in rows if r['passed']]
x=np.array([[1,r['H'],r['P']] for r in accepted]);fits={}
for key in ['cc_ms','hc_ms']:
 y=np.array([r[key] for r in accepted]);coef=np.linalg.lstsq(x,y,rcond=None)[0];res=x@coef-y
 fits[key]=dict(coefficients=coef.tolist(),rmse_ms=float(np.sqrt(np.mean(res**2))),r2=float(1-sum(res**2)/sum((y-y.mean())**2)))
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'cc_ms':'#0072B2','hc_ms':'#E69F00'}
for key,label in [('cc_ms','CPU path CC'),('hc_ms','NPU-to-CPU path HC')]:
 ax=axs[0,0]
 ax.scatter([r['P'] for r in accepted],[r[key]/1000 for r in accepted],color=colors[key],label=label,s=32)
 for r in rows:
  if not r['passed']:ax.scatter(r['P'],r[key]/1000,facecolors='none',edgecolors=colors[key],marker='s',s=80)
ax.set(title='Measured path costs (D=4)',xlabel='Append P (tokens)',ylabel='Time (s)');ax.legend()
ax=axs[0,1]
for key,label in [('cc_ms','CC'),('hc_ms','HC')]:
 coef=np.array(fits[key]['coefficients']);pred=x@coef/1000;y=np.array([r[key]/1000 for r in accepted]);ax.scatter(y,pred,color=colors[key],label=f"{label}: RMSE {fits[key]['rmse_ms']/1000:.2f}s",s=30)
ax.plot([0,26],[0,26],color='#777',linestyle='--');ax.set(title='Affine fit: a + bH + cP',xlabel='Measured cost (s)',ylabel='Fitted cost (s)');ax.legend()
ax=axs[1,0]
for key,label,color in [('cc_restore_ms','CPU restore','#0072B2'),('hc_prepare_ms','NPU preparation','#E69F00')]:
 h=np.array([r['H'] for r in accepted]);y=np.array([r[key] for r in accepted]);coef=np.polyfit(h,y,1);grid=np.linspace(h.min(),h.max(),100)
 ax.scatter(h,y,color=color,s=25,label=label);ax.plot(grid,np.polyval(coef,grid),color=color)
 fits[key]=dict(intercept_ms=float(coef[1]),slope_ms_per_token=float(coef[0]))
 for r in rows:
  if not r['passed']:ax.scatter(r['H'],r[key],facecolors='none',edgecolors=color,marker='s',s=80)
ax.set(title='Restore / preparation vs history',xlabel='History H (tokens)',ylabel='Time (ms)');ax.legend()
ax=axs[1,1]
for h,color in [(285,'#0072B2'),(700,'#E69F00'),(1160,'#009E73')]:
 p=np.linspace(33,202,200);cc=np.array(fits['cc_ms']['coefficients']);hc=np.array(fits['hc_ms']['coefficients']);diff=cc-hc
 ax.plot(p,(diff[0]+diff[1]*h+diff[2]*p)/1000,label=f'H={h}',color=color)
ax.axhline(0,color='#777',linestyle='--');ax.set(title='Fitted path advantage within sampled P range',xlabel='Append P (tokens)',ylabel='CC − HC (s; positive favors HC)');ax.legend()
fig.suptitle('Previous offline calibration: 19 accepted points × 3 repeats; 1 excluded point shown as open squares',fontsize=11)
fig.savefig(out/'legacy-fits.png',dpi=180);fig.savefig(out/'legacy-fits.pdf')
fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[]
for i,ax in enumerate(axs.flat):
 texts=[ax.title,ax.xaxis.label,ax.yaxis.label]
 for j,t in enumerate(texts):
  b=t.get_window_extent(renderer)
  if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1):issues.append(str(i)+' outside')
  for u in texts[j+1:]:
   if b.overlaps(u.get_window_extent(renderer)):issues.append(str(i)+' overlap')
(out/'layout-check.json').write_text(json.dumps(issues))
with (out/'table.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'fits.json').write_text(json.dumps(dict(fits=fits,rows=rows),indent=2))
print(json.dumps(fits,indent=2));print('excluded',[(r['H'],r['P']) for r in rows if not r['passed']]);print('layout',issues)
