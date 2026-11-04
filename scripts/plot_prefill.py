#!/usr/bin/env python3
"""离线画 B 阶段成本查表与状态准备开销，不访问设备。"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.font_manager as fm
import matplotlib.text as mtext
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.cost_model import route_times


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    data = json.loads((a.run / 'analysis.json').read_text())
    config = json.loads((a.run / 'manifest.json').read_text())['config']
    spec = importlib.util.spec_from_file_location('figure_style', Path.home() / '.codex/skills/figure-style/kernel.py')
    style = importlib.util.module_from_spec(spec); spec.loader.exec_module(style)
    font = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
    fm.fontManager.addfont(font)
    style.apply_figure_style(font=fm.FontProperties(fname=font).get_name(), sizes=(14, 12, 11))
    plt.rcParams['axes.unicode_minus'] = False
    out = a.run / 'figures'; out.mkdir(exist_ok=True)
    qa = {}
    def save(fig, name):
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
        qa[name] = {'text_overlaps': [[x.get_text(),y.get_text()] for i,(x,xb) in enumerate(texts)
                                     for y,yb in texts[i+1:] if xb.overlaps(yb)],
                    'outside': [t.get_text() for t,b in texts if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]}
        fig.savefig(out / (name+'.png'), dpi=300)
        fig.savefig(out / (name+'.pdf'))
        plt.close(fig)
    hs, ps = config['history_tokens'], config['new_tokens']
    lookup = {(r['history_tokens'],r['new_tokens'],r['gap_ms']):r for r in data['router_table']}
    status = {(r['history_tokens'], r['new_tokens']):r['status'] for r in data['cell_status']}
    fig, axes = plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.82,bottom=.17,top=.84,hspace=.3,wspace=.12)
    fig.text(.08,.95,'CPU/HTP Prefill 的实测路径查表',fontsize=14)
    fig.text(.08,.90,'文本后缀微基准 · 双上下文驻留 · 已测点各3次配对测量的中位数',fontsize=12)
    norm = TwoSlopeNorm(vmin=.5,vcenter=1,vmax=3)
    cmap = plt.get_cmap('PuOr_r')
    for ax,gap in zip(axes.flat,config['gap_ms']):
        values = np.full((len(hs),len(ps)),np.nan)
        for i,h in enumerate(hs):
            for j,pv in enumerate(ps):
                r=lookup.get((h,pv,gap))
                if r and r['choice']!='insufficient_samples':
                    values[i,j]=r['cpu_ms']/r['htp_ms']
        im=ax.imshow(values,cmap=cmap,norm=norm,aspect='auto')
        ax.set_title(f'Gap = {gap} ms',loc='left',fontsize=14,pad=9)
        ax.set_xticks(range(len(ps)),list(map(str,ps)))
        ax.set_yticks(range(len(hs)),[str(h) if h<1024 else f'{h//1024}K' for h in hs])
        for i in range(len(hs)):
            for j in range(len(ps)):
                v=values[i,j]
                if np.isnan(v):
                    label = '未通过' if status.get((hs[i],ps[j])) == 'validation_failed' else 'n.d.'
                    ax.text(j,i,label,ha='center',va='center',fontsize=12)
                else:
                    rgb=np.array(cmap(norm(v))[:3]); linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                    lum=float(np.dot(linear,[.2126,.7152,.0722]))
                    color='black' if (lum+.05)/.05>=1.05/(lum+.05) else 'white'
                    ax.text(j,i,('HTP' if v>1 else 'CPU')+f'\n{v:.2g}×',ha='center',va='center',fontsize=12,color=color)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.45,.095,'新增输入 P（token）',ha='center',fontsize=14)
    fig.text(.025,.50,'历史状态 H（token）',rotation=90,va='center',fontsize=14)
    cax=fig.add_axes([.86,.24,.025,.53]); cb=fig.colorbar(im,cax=cax,ticks=[.5,1,1.5,2,3])
    cb.set_label('CPU 耗时 / HTP 耗时；1 为持平',fontsize=12)
    fig.text(.08,.025,'HTP 成本含准备及返回；Gap 为理想重叠模拟。未通过的点不参与选路；n.d.表示样本不足。',fontsize=11)
    save(fig,'01_route_map')

    excluded={(r['history_tokens'],r['new_tokens'],r['repetition']) for r in data['rejected_trials']}
    trials=[json.loads(l) for f in a.run.glob('H*.jsonl') for l in f.read_text().splitlines()]
    trials=[r for r in trials if (r['history_tokens'],r['new_tokens'],r['repetition']) not in excluded and r['new_tokens']==560]
    fig,ax=plt.subplots(figsize=(8.8,5.2))
    fig.subplots_adjust(left=.14,right=.94,bottom=.23,top=.80)
    fig.text(.10,.92,'历史状态的准备与返回开销',fontsize=14)
    fig.text(.10,.855,'P = 560 文本 token · 淡点为每次测量，实点与连线为中位数',fontsize=12)
    for key,label,color,marker in [('prepare_ms','CPU→HTP 准备','#287EA8','o'),('return_ms','HTP→CPU 返回','#737B89','^')]:
        med=[]
        for i,h in enumerate(hs):
            vals=[route_times(t,0)[key] for t in trials if t['history_tokens']==h]
            med.append(np.median(vals) if vals else np.nan)
            ax.scatter(i+np.linspace(-.055,.055,len(vals)),vals,color=color,alpha=.3,s=26,marker=marker,zorder=2)
        ax.plot(range(len(hs)),med,marker=marker,color=color,label=label,zorder=3)
    ax.set_xticks(range(len(hs)),[str(h) if h<1024 else f'{h//1024}K' for h in hs])
    for i,h in enumerate(hs):
        if not any(t['history_tokens']==h for t in trials):
            label = '未通过' if status.get((h,560)) == 'validation_failed' else 'n.d.'
            ax.text(i,.025,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=11)
    ax.set_xlabel('历史状态 H（token）'); ax.set_ylabel('耗时（ms）')
    ax.set_ylim(bottom=0); ax.set_xlim(-.25,len(hs)-.75)
    ax.legend(frameon=False,loc='upper left')
    fig.text(.10,.035,'准备含 Export/Clear/Import；返回含反向三步。\n无独立 KV tiling 操作，未填造其耗时。',fontsize=11)
    save(fig,'02_state_cost')
    (out / 'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    (out / 'sources.json').write_text(json.dumps({'analysis':str(a.run/'analysis.json'),'raw':list(map(str,a.run.glob('H*.jsonl')))},indent=2)+'\n')
    print(out)


if __name__=='__main__': main()
