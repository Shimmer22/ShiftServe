"""真实边界的实测路径包络；缺失HC不插值。"""
exec(open('scripts/plot/bfcl_cached_repeats.py').read().split('colors=')[0])
from collections import defaultdict
by=defaultdict(list)
for r,b,q,rr in rows:
 if q['P']>=40: by[b].append((q,rr))
supplement=Path('results/20261110-bfcl-small-p-hc-supplement')
for f in sorted((supplement/'measurements').glob('r*-b*.json')):
 if '.raw.' in f.name: continue
 d=json.loads(f.read_text());q=d['request'];rr={v['condition']:v for v in d['response']['results']};by[q['original_boundary']].append((q,rr))
pts=[]
for b,items in by.items():
 q=items[0][0];cc=np.median([rr['cpu']['prefill_ms'] for q,rr in items])/1000
 hc=np.median([rr['route']['prefill_ms']+rr['route']['handoff_ms'] for q,rr in items])/1000 if all(rr['route']['route']=='htp' for q,rr in items) else np.nan
 pts.append(dict(boundary=b,P=q['P'],H=q['H'],D=q['D'],cc_s=cc,hc_s=None if np.isnan(hc) else hc))
# 同P存在不同真实历史：每个边界先取三轮中位数，再同P等权平均。
ps=sorted({p['P'] for p in pts});cc=[];hc=[]
for p in ps:
 group=[v for v in pts if v['P']==p];cc.append(np.mean([v['cc_s'] for v in group]));hv=[v['hc_s'] for v in group if v['hc_s'] is not None];hc.append(np.mean(hv) if len(hv)==len(group) else np.nan)
cc=np.array(cc);hc=np.array(hc);best=np.minimum(cc,hc)
fig,axs=plt.subplots(1,2,figsize=(12,5.7),gridspec_kw={'width_ratios':[1.65,1]});fig.subplots_adjust(left=.07,right=.98,bottom=.3,top=.73,wspace=.25)
for ax in axs:
 ax.plot(ps,cc,'o-',color='#0072b2',ms=4,label='CPU prefill（CC）')
 ax.plot(ps,hc,'s--',color='#d55e00',ms=4,label='HTP prefill＋迁移（HC）')
 ax.plot(ps,best,'D:',color='#7b3896',ms=5,lw=2,label='实测最小值 min(CC, HC)',zorder=4)
 ax.set_xlabel('返回后待处理 token 数 P');ax.grid(axis='y',color='#e5e5e5',lw=.6);ax.set_axisbelow(True)
axs[0].set_title('a  全部 24 个真实边界',loc='left',pad=15);axs[0].set_ylabel('Prefill＋迁移耗时（秒）');axs[0].set_xticks([23,32,40,56,67,83,97]);axs[0].set_ylim(bottom=0)
axs[1].set_title('b  P=29–45 附近',loc='left',pad=15);axs[1].set_xlim(28,46);axs[1].set_xticks([29,30,32,34,35,38,40,44,45]);axs[1].set_ylim(bottom=0)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.90),ncol=3,fontsize=10)
fig.suptitle('补齐后的实测查找表：CPU 与 HTP→CPU 的耗时下包络',x=.07,y=.98,ha='left',fontsize=12)
fig.text(.07,.19,'每个边界先取 3 轮中位数；同 P 的不同边界再等权平均。连线仅辅助阅读，H 未固定。',fontsize=10)
fig.text(.07,.13,'P≤39 使用本次同轮配对补测，P≥40 使用旧轮次；两批采集时段不同，不能推定统一阈值。',fontsize=10)
fig.text(.07,.07,'仅比较返回后的 prefill＋迁移，不含 decode、提前准备、初始化或夹具恢复；此包络为事后选择。',fontsize=10)
fig.canvas.draw();texts=[(t,t.get_window_extent()) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()];issues=[(t.get_text(),u.get_text()) for i,(t,a) in enumerate(texts) for u,b in texts[i+1:] if a.overlaps(b)]
print('overlaps',issues)
for ext in ['png','pdf']:fig.savefig(out/f'completed-envelope.{ext}',dpi=300)
(out/'completed-envelope-data.json').write_text(json.dumps(pts,ensure_ascii=False,indent=2))
