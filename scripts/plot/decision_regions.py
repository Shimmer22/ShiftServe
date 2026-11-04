#!/usr/bin/env python3
"""离线查看已有四路径曲线与离散决策点，不外推未测P或长输出。"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.font_manager as fm
import matplotlib.text as mtext
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from a2o.scheduling.cost_model import path_cost
from a2o.validation.contracts import check_response


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--history',type=int,default=4096);a=parser.parse_args()
    cfg=json.loads((a.run/'manifest.json').read_text())['config']
    rows=[json.loads(l) for f in a.run.glob('H*.jsonl') for l in f.read_text().splitlines()]
    rows=[r for r in rows if r['history_tokens']==a.history]
    for r in rows:
        r['included']=(check_response(r,{'check':{'type':'json_action','action':'wait'}})['passed']
                       and r['eog'] and r['numerical_passed'] and r['generated_ids_equal'])
    if not rows:raise ValueError('requested history not measured')
    spec=importlib.util.spec_from_file_location('figure_style',Path.home()/'.codex/skills/figure-style/kernel.py')
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
    font='/System/Library/Fonts/Supplemental/Arial Unicode.ttf';fm.fontManager.addfont(font)
    style.apply_figure_style(font=fm.FontProperties(fname=font).get_name(),sizes=(14,12,11))
    plt.rcParams['axes.unicode_minus']=False
    paths=cfg['paths'];homes=cfg['homes'];ps=cfg['new_tokens']
    colors={'CC':'#287EA8','HH':'#8B69AC','HC':'#D88635','CH':'#687984'}
    markers={'CC':'o','HH':'^','HC':'s','CH':'D'}
    out=a.run/'decision-figures';out.mkdir(exist_ok=True);qa={};curve_data=[];map_data=[]
    def save(fig,name):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
        qa[name]={'overlaps':[[x.get_text(),y.get_text()] for n,(x,xb) in enumerate(texts) for y,yb in texts[n+1:] if xb.overlaps(yb)],
                  'outside':[t.get_text() for t,b in texts if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]}
        fig.savefig(out/(name+'.png'),dpi=300);fig.savefig(out/(name+'.pdf'))
        fig.savefig(out/(name+'.svg'));plt.close(fig)
    def select(home,p,path):return [r for r in rows if r['home']==home and r['new_tokens']==p and r['path']==path]
    for home in homes:
        fig,axes=plt.subplots(1,len(ps),figsize=(11,5.3),squeeze=False)
        fig.subplots_adjust(left=.09,right=.97,top=.72,bottom=.29,wspace=.22)
        fig.text(.07,.95,f'四路径成本随 Decode 计算量变化 · 初始状态在 {home.upper()}',fontsize=14)
        fig.text(.07,.885,f'H={a.history}，Gap=0；P取实测点，未外推更长输出',fontsize=12)
        for ax,p in zip(axes.flat,ps):
            for path in paths:
                trials=select(home,p,path);valid=[r for r in trials if r['included']]
                if len(valid)>=cfg['repetitions']:
                    count=min(len(r['decode_prefix_ms']) for r in valid);d=np.arange(1,count+1)
                    vals=np.array([[path_cost(r,0,int(i))['total_ms']/1000 for i in d] for r in valid])
                    ax.fill_between(d,vals.min(axis=0),vals.max(axis=0),color=colors[path],alpha=.12)
                    ax.plot(d,np.median(vals,axis=0),color=colors[path],marker=markers[path],markersize=4)
                    for i,mid,lo,hi in zip(d,np.median(vals,axis=0),vals.min(axis=0),vals.max(axis=0)):
                        curve_data.append({'home':home,'P':p,'path':path,'decode_evaluations':int(i),
                                           'median_s':float(mid),'min_s':float(lo),'max_s':float(hi),'n':len(valid)})
                # Failed rows are individual dashed/open traces, never part of the summary above.
                for r in trials:
                    if r['included']:continue
                    d=np.arange(1,len(r['decode_prefix_ms'])+1)
                    vals=[path_cost(r,0,int(i))['total_ms']/1000 for i in d]
                    ax.plot(d,vals,color=colors[path],ls='--',lw=.8,marker=markers[path],mfc='none',ms=4,alpha=.55)
            ax.set_title(f'P={p}',loc='left',fontsize=14)
            ax.set_xticks(range(1,7));ax.set_xlabel('Decode 前向评估次数 d');ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.15);ax.margins(x=.045)
        axes[0,0].set_ylabel('累计路径成本（秒）')
        handles=[]
        for path in paths:
            all_failed=not any(r['included'] for r in rows if r['path']==path and r['home']==home)
            handles.append(Line2D([],[],color=colors[path],marker=markers[path],ls='--' if all_failed else '-',
                                  mfc='none' if all_failed else colors[path],
                                  label=f'{path} '+{'CC':'CPU/CPU','HH':'HTP/HTP','HC':'HTP/CPU','CH':'CPU/HTP'}[path]))
        fig.legend(handles=handles,ncol=4,loc='upper left',bbox_to_anchor=(.07,.845),fontsize=11,frameon=False)
        fig.text(.07,.16,'实线：合格试验的中位数与范围（每点n=3）；虚线空心：未过数值校验的单次记录。',fontsize=11)
        fig.text(.07,.09,'d=1–5为同一输出轨迹的计算前缀；d=6含EOG提交。不是六种实际输出长度。',fontsize=11)
        fig.text(.07,.025,'文本校准，包含迁移，不含GPU ViT/初始化/核验；HH失败轨迹不参与决策。',fontsize=11)
        save(fig,f'cost-vs-decode-{home}')

    if ps!=[16,64]:raise ValueError('discrete map layout currently expects measured P=16,64')
    fig,axes=plt.subplots(1,len(homes),figsize=(10.8,5.5),sharey=True,squeeze=False)
    fig.subplots_adjust(left=.10,right=.97,top=.71,bottom=.28,wspace=.22)
    fig.text(.07,.95,'Tool Slack 下的已测决策点 · 未测 P 区域保持空白',fontsize=14)
    fig.text(.07,.885,f'H={a.history}，完整短动作（6次前向，含EOG）；只在合格候选中比较',fontsize=12)
    cmap=ListedColormap([colors[p] for p in paths]);norm=BoundaryNorm(np.arange(5)-.5,4)
    edges=np.arange(0,201);qualified={}
    for ax,home in zip(axes.flat,homes):
        values=np.full((200,3),np.nan);qualified[home]={}
        for col,p in [(0,16),(2,64)]:
            eligible={path:[r for r in select(home,p,path) if r['included']] for path in paths}
            eligible={path:rr for path,rr in eligible.items() if len(rr)>=cfg['repetitions']}
            qualified[home][p]=list(eligible)
            for i in range(200):
                g=i+.5
                costs={path:float(np.median([path_cost(r,g)['total_ms'] for r in rr])) for path,rr in eligible.items()}
                if not costs:continue
                best=min(costs,key=costs.get);values[i,col]=paths.index(best)
                map_data.append({'home':home,'P':p,'gap_ms':g,'best_qualified':best,'cost_ms':costs[best],
                                 'qualified_paths':','.join(eligible),'all_four_qualified':len(eligible)==4})
        ax.pcolormesh([0,1,3,4],edges,values,cmap=cmap,norm=norm,shading='flat',alpha=.22)
        ax.add_patch(Rectangle((1,0),2,200,facecolor='#FAFAFA',edgecolor='#CDCDCD',hatch='///',lw=.6))
        ax.text(2,100,'17≤P≤63\n未测，不插值',ha='center',va='center',fontsize=12,color='#666666')
        for col,p,x in [(0,16,.5),(2,64,3.5)]:
            unique=set(values[:,col][np.isfinite(values[:,col])])
            label=paths[int(next(iter(unique)))]+'*' if len(unique)==1 else '无合格数据' if not unique else '随G变化'
            ax.text(x,100,label,ha='center',va='center',fontsize=14)
        ax.set_title('初始状态：'+home.upper(),loc='left',fontsize=14)
        ax.set_xticks([.5,2,3.5],['P=16','未测区域','P=64']);ax.set_yticks([0,50,100,150,200]);ax.set_ylim(0,200)
        ax.set_xlabel('新增输入 P（离散采样）')
    axes[0,0].set_ylabel('Tool Slack G（ms，离线模拟）')
    fig.legend(handles=[Patch(color=colors[p],alpha=.3,label=p) for p in ['CC','HC']],loc='upper left',bbox_to_anchor=(.07,.845),ncol=2,frameon=False,fontsize=11)
    fig.text(.07,.16,'* 仅为CC/HC/CH合格候选中的最小成本；HH在这组未过数值校验，不宣称四路径全局最优。',fontsize=11)
    fig.text(.07,.09,'两侧色条只代表P=16、64两个采样点；其宽度不表示已验证的P区间。',fontsize=11)
    fig.text(.07,.025,'G按实测准备耗时做理想重叠模拟；不是实际Tool并发结果，两个采样列未发生选路翻转。',fontsize=11)
    save(fig,'sampled-decision-map')
    for name,data in [('curve-data',curve_data),('decision-data',map_data)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    (out/'figure-plan.json').write_text(json.dumps({
        'curve_claim':'H4K measured decode prefixes show different CPU/HTP path slopes; failed traces are diagnostic only',
        'map_claim':'Among qualified candidates, measured P16 selects CC and P64 selects HC for G0–200; intermediate P is unknown',
        'source':[str(f) for f in a.run.glob('H*.jsonl')],'qualified_paths':qualified,
        'roles':{'curves':'measured cost and failure boundary','map':'qualified discrete choices; unmeasured region explicitly blank'},
        'no_interpolation_or_extrapolation':True},ensure_ascii=False,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
