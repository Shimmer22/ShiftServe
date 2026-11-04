#!/usr/bin/env python3
"""四路径完整短动作成本；只画通过校验且重复数充足的测量。"""
import argparse
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.text as mtext
import numpy as np


def main():
    parser = argparse.ArgumentParser();parser.add_argument('run',type=Path);a=parser.parse_args()
    cfg=json.loads((a.run/'manifest.json').read_text())['config']
    report=json.loads((a.run/'path-analysis.json').read_text())
    spec=importlib.util.spec_from_file_location('figure_style',Path.home()/'.codex/skills/figure-style/kernel.py')
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
    font='/System/Library/Fonts/Supplemental/Arial Unicode.ttf';fm.fontManager.addfont(font)
    style.apply_figure_style(font=fm.FontProperties(fname=font).get_name(),sizes=(14,12,11))
    plt.rcParams['axes.unicode_minus']=False
    out=a.run/'figures';out.mkdir(exist_ok=True);qa={}
    lookup={(r['H'],r['P'],r['home'],r['G_ms'],r['path']):r for r in report['table'] if r['D']=='complete'}
    gaps=[0,max(cfg['gap_ms'])];paths=cfg['paths']
    for home in cfg['homes']:
        fig,axes=plt.subplots(len(cfg['history_tokens']),len(cfg['new_tokens']),squeeze=False,figsize=(10,7.4))
        fig.subplots_adjust(left=.10,right=.96,top=.78,bottom=.20,hspace=.5,wspace=.3)
        fig.text(.08,.95,f'四路径完整短动作成本 · 初始状态在 {home.upper()}',fontsize=14)
        fig.text(.08,.895,'结构化文本历史 · 每点3次 · 含结束符提交，非流式响应延迟',fontsize=12)
        for i,h in enumerate(cfg['history_tokens']):
            for j,p in enumerate(cfg['new_tokens']):
                ax=axes[i,j];x=np.arange(len(paths));width=.34
                for k,g in enumerate(gaps):
                    vals=[];low=[];high=[]
                    for path in paths:
                        r=lookup.get((h,p,home,g,path));ok=r is not None and r['n']>=cfg['repetitions']
                        vals.append(r['total_ms']/1000 if ok else np.nan)
                        low.append((r['total_ms']-r['min_ms'])/1000 if ok else 0)
                        high.append((r['max_ms']-r['total_ms'])/1000 if ok else 0)
                    ax.bar(x+(k-.5)*width,vals,width,label=f'Gap={g}ms（理想重叠）',color=['#287EA8','#D88635'][k],
                           yerr=[low,high],capsize=2,error_kw={'elinewidth':.8})
                for n,path in enumerate(paths):
                    r=lookup.get((h,p,home,0,path))
                    if r is None or r['n']<cfg['repetitions']:
                        ax.text(n,.15,'未通过\n或缺样本',transform=ax.get_xaxis_transform(),ha='center',fontsize=11)
                ax.set_xticks(x,paths);ax.set_title(f'H={h}, P={p}',loc='left',fontsize=12)
                ax.set_ylim(bottom=0);ax.set_ylabel('路径成本（秒）')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.08,.865),ncol=2,frameon=False,fontsize=11)
        fig.text(.08,.07,'仅含通过JSON、逐步数值及token一致性检查的点；误差线为实测范围。',fontsize=11)
        fig.text(.08,.025,'Gap离线模拟，仅隐藏初始历史准备；未验证真实Tool并发或在线调度。',fontsize=11)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
        qa[home]={'overlaps':[[x.get_text(),y.get_text()] for n,(x,xb) in enumerate(texts) for y,yb in texts[n+1:] if xb.overlaps(yb)],
                  'outside':[t.get_text() for t,b in texts if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]}
        fig.savefig(out/f'paths-{home}.png',dpi=300);fig.savefig(out/f'paths-{home}.pdf');plt.close(fig)
    (out/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    (out/'sources.json').write_text(json.dumps({'source':str(a.run/'path-analysis.json')},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
