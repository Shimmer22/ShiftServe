#!/usr/bin/env python3
"""用共同测点绘制共享权重和独立权重下的 CPU/HTP 配对耗时。"""

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
from statistics import median
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get('A2O_FIGURE_STYLE_DIR', str(Path.home() / '.codex/skills/figure-style')))
from kernel import apply_figure_style

MAIN = ROOT / 'results/mainline-eval'
POINTS = (16, 24, 32)
COLORS = {'cpu': '#2175A5', 'htp': '#B96420'}
MARKERS = {'cpu': 's', 'htp': 'o'}


def load_pairs(directory, shared):
    trials = defaultdict(dict)
    with (directory / 'stage-times.csv').open() as stream:
        for row in csv.DictReader(stream):
            p = int(row['P'])
            if p not in POINTS or (shared and row['cpu_prefill_threads'] != '4'):
                continue
            backend = row['backend'] if shared else ('cpu' if row['arm'] == 'independent' else 'htp')
            key = (0 if shared else int(row['pair']), int(row['round']), p)
            elapsed = float(row['prefill_ms']) / 1000
            if not math.isfinite(elapsed) or elapsed <= 0 or backend in trials[key]:
                raise ValueError(f'invalid or duplicated sample: {directory.name} {key} {backend}')
            trials[key][backend] = elapsed
    summary = json.loads((directory / 'summary.json').read_text())
    groups = {p: [v for k,v in sorted(trials.items()) if k[2] == p] for p in POINTS}
    expected = 4 if shared else 6
    for p, samples in groups.items():
        if len(samples) != expected or any(set(trial) != set(COLORS) for trial in samples):
            raise ValueError(f'incomplete pairs: {directory.name}, P={p}')
        reported = next(row for row in summary['thread_points' if shared else 'points']
                        if row['P'] == p and (not shared or row['cpu_prefill_threads'] == 4))
        if (not math.isclose(median(x['cpu'] for x in samples), reported['cpu_s' if shared else 'independent_s'])
                or not math.isclose(median(x['htp'] for x in samples), reported['htp_s'])):
            raise ValueError(f'median differs from accepted result: {directory.name}, P={p}')
    return groups, summary['snapshot_checksum']


def draw(groups, shared, limits, output):
    title = '共享权重下的 Prefill' if shared else '独立权重下的 Prefill'
    cpu_label = '共享权重 CPU' if shared else '独立权重 CPU'
    repeats = len(groups[POINTS[0]])
    fig, ax = plt.subplots(figsize=(7.25, 5.25))
    fig.subplots_adjust(left=.13, right=.95, bottom=.22, top=.72)
    fig.text(.13,.95,title,fontsize=22,va='top')
    fig.text(.13,.855,'Qwen3-8B Q4_0 · H=4096 · CPU 和 HTP 主机均 4 线程',
             color='#435061',fontsize=10.5,va='top')
    handles = [Line2D([],[],color=COLORS[b],marker=MARKERS[b],label=label,lw=2,ms=6)
               for b,label in (('cpu',cpu_label),('htp','HTP offload'))]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.125,.798),ncol=2,
               columnspacing=1.5,handletextpad=.65,borderaxespad=0)
    deltas = [median(x['cpu'] for x in groups[p]) - median(x['htp'] for x in groups[p]) for p in POINTS]
    for p,q,d1,d2 in zip(POINTS, POINTS[1:], deltas, deltas[1:]):
        if d1*d2 < 0:
            ax.axvspan(p,q,color='#EBEEF2',zorder=0)
            ax.text((p+q)/2,.945,f'交叉区间  {p}–{q}',transform=ax.get_xaxis_transform(),
                    color='#435061',ha='center',va='top',fontsize=10)
    for backend,color in COLORS.items():
        direction = -1 if backend == 'cpu' else 1
        for p,trials in groups.items():
            jitter = [p+.35*direction+.07*(j-(repeats-1)/2) for j in range(repeats)]
            ax.scatter(jitter,[x[backend] for x in trials],color=color,s=31,alpha=.46,
                       marker=MARKERS[backend],linewidths=0,zorder=2)
        ax.plot(POINTS,[median(x[backend] for x in groups[p]) for p in POINTS],
                color=color,marker=MARKERS[backend],ms=6,lw=2,zorder=3)
    ax.set(xlim=(14.3,33.7),ylim=limits,xticks=POINTS,yticks=[2,3,4,5])
    ax.grid(axis='y',color='#E2E6EA',linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel('新增 token 数 P',labelpad=9)
    ax.set_ylabel('纯 Prefill 耗时（秒）',labelpad=9)
    wins = '、'.join(f'{sum(x["cpu"]<x["htp"] for x in groups[p])}/{repeats}' for p in POINTS)
    fig.text(.13,.095,f'每点 {repeats} 对，散点为单次耗时，连线为中位数。',color='#435061',fontsize=10)
    fig.text(.13,.049,f'CPU 配对胜场（从左到右）为 {wins}。',color='#435061',fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(t.get_text(),t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
              if t.get_visible() and t.get_text().strip()]
    outside = [t for t,b in labels if not all(fig.bbox.contains(*p) for p in b.get_points())]
    overlap = [(t,u) for i,(t,b) in enumerate(labels) for u,c in labels[i+1:] if b.overlaps(c)]
    if outside or overlap:
        raise RuntimeError(f'{title}: {outside=} {overlap=}')
    stem = f'prefill-{"shared" if shared else "independent"}-weights'
    for ext in ('png','pdf'):
        fig.savefig(output/f'{stem}.{ext}',dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared',type=Path,default=MAIN/'prefill-fixed-history-thread-audit')
    parser.add_argument('--independent',type=Path,default=MAIN/'prefill-crossing-confirmation-h4096-8b')
    parser.add_argument('--output',type=Path,default=MAIN/'prefill-crossing-confirmation-h4096-8b/figures')
    args = parser.parse_args()
    shared, sh = load_pairs(args.shared,True)
    independent, ih = load_pairs(args.independent,False)
    if sh != ih:
        raise ValueError('the comparisons do not share the same KV snapshot')
    values = [v for groups in (shared,independent) for trials in groups.values() for t in trials for v in t.values()]
    limits = (math.floor((min(values)-.1)*2)/2,math.ceil((max(values)+.1)*2)/2)
    font = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(font)
    apply_figure_style(font='Arial Unicode MS' if font.exists() else 'Noto Sans CJK SC',sizes=(12,10,10))
    plt.rcParams.update({'savefig.bbox':None,'axes.unicode_minus':False})
    args.output.mkdir(parents=True,exist_ok=True)
    draw(shared,True,limits,args.output)
    draw(independent,False,limits,args.output)


if __name__ == '__main__':
    main()
