"""绘制24个冻结边界的三轮手机复测；从原始JSON读取。"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, '/Users/hui/.codex/skills/figure-style')
from kernel import apply_figure_style
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
apply_figure_style(font='Arial Unicode MS', sizes=(12,10,9))
plt.rcParams['axes.unicode_minus']=False
root=Path('results/20261109-bfcl-cached-repeats'); out=root/'figures';out.mkdir(exist_ok=True)
rows=[]
for f in sorted((root/'measurements').glob('r*-b*.json')):
 if '.raw.' in f.name: continue
 d=json.loads(f.read_text()); q=d['request']; rr={x['condition']:x for x in d['response']['results']}
 assert set(rr)=={'cpu','htp','route'}
 assert all(all(x[k]==q[k] for k in ('H','P','D')) for x in rr.values())
 rows.append((f.name[:3],f.stem.split('-')[1],q,rr))
assert len(rows)==72
colors={'cpu':'#7c8796','htp':'#c38b46','route':'#007c91'}
fig,axs=plt.subplots(1,2,figsize=(12,5.4)); fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.79,wspace=.3)
ax=axs[0]
for c,label in [('cpu','CPU'),('htp','HTP'),('route','Route')]:
 vals=[sum(rr[c]['total_ms'] for r,b,q,rr in rows if r==f'r0{i}')/1000 for i in range(3)]
 ax.plot([1,2,3],vals,'o-',label=label,color=colors[c],lw=2,ms=6)
ax.set(xticks=[1,2,3],xlabel='重复轮次',ylabel='24 个边界累计耗时（秒）',ylim=(0,1400))
ax.set_title('a  三轮 Route 均比 CPU 少约 7%',loc='left',pad=16)
ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.02))
ax=axs[1]
for backend,label,marker in [('cpu','Route 选 CPU','o'),('htp','Route 选 HTP','^')]:
 selected=[(q,rr) for r,b,q,rr in rows if rr['route']['route']==backend]
 ax.scatter([q['P'] for q,rr in selected],[(1-(rr['route']['prefill_ms']+rr['route']['handoff_ms'])/rr['cpu']['prefill_ms'])*100 for q,rr in selected],c=colors[backend],marker=marker,s=30,alpha=.72,label=label,edgecolors='white',linewidths=.3)
ax.axhline(0,color='#555555',lw=.8,ls='--')
ax.set(xlabel='返回后待处理 token 数 P',ylabel='Prefill＋迁移相对 CPU 耗时减少（%）',xlim=(18,102))
ax.set_title('b  按 P 看返回后处理收益',loc='left',pad=16)
ax.legend(loc='lower right')
for ax in axs:ax.grid(axis='y',color='#e6e6e6',lw=.6);ax.set_axisbelow(True)
fig.suptitle('手机冻结边界复测：Route 累计耗时减少 6.73%（相对 CPU）',x=.075,y=.96,ha='left',fontsize=12)
fig.text(.075,.145,'24 个边界 × 3 轮 × 3 条件，共 216 次测量；右图每点为一个边界的一轮实测。',fontsize=10)
fig.text(.075,.095,'固定 token 重放；准备、夹具恢复与初始化不计入耗时；未测真实 Tool Gap 异步重叠或生成质量。',fontsize=10)
fig.text(.075,.045,'全部选路均超出原成本模型校准范围；不同 P 的 H、D 不同，右图不代表单独改变 P 的因果曲线。',fontsize=10)
fig.canvas.draw()
texts=[(t,t.get_window_extent()) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
issues=[]
for i,(t,a) in enumerate(texts):
 for u,b in texts[i+1:]:
  if a.overlaps(b):issues.append((t.get_text(),u.get_text()))
(out/'layout-check.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2))
for ext in ('png','pdf'):fig.savefig(out/f'cached-repeats.{ext}',dpi=300)
print('overlaps',issues)
print(out/'cached-repeats.png')
