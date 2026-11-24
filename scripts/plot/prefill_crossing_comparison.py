#!/usr/bin/env python3
"""绘制旧密集扫描与 8B 直接交错的 CPU/HTP Prefill 交叉。"""

import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get('A2O_FIGURE_STYLE_DIR', str(Path.home() / '.codex/skills/figure-style')))
from kernel import apply_figure_style, panel_letter

HISTORICAL = ROOT / 'results/20261111-prefill-p24-128-geometric-cpu6-t10/figures-cleaned'
DIRECT = ROOT / 'results/mainline-eval/prefill-crossing-confirmation-h4096-8b'
COLORS = {'cpu': '#2175A5', 'htp': '#B96420'}
MARKERS = {'cpu': 's', 'htp': 'o'}


def historical_samples(directory):
    groups = defaultdict(list)
    with (directory / 'cleaned-stage-times.csv').open() as stream:
        for row in csv.DictReader(stream):
            p = int(row['P'])
            if 24 <= p <= 64:
                assert (row['H'], row['stage'], row['D']) == ('4096', 'prefill', '0')
                assert int(row['n_threads_batch']) == (6 if row['backend'] == 'cpu' else 4)
                groups[p, row['backend']].append(float(row['elapsed_ms']) / 1000)
    with (directory / 'excluded-stage-times.csv').open() as stream:
        excluded = sum(24 <= int(row['P']) <= 64 for row in csv.DictReader(stream))
    assert set(groups) == {(p, b) for p in range(24, 65) for b in COLORS}
    assert all(9 <= len(vals) <= 10 for vals in groups.values())
    assert sum(map(len, groups.values())) + excluded == 820
    return groups, excluded


def direct_samples(directory):
    groups = defaultdict(list)
    paired = defaultdict(dict)
    arms = ('independent', 'htp', 'htp', 'independent')
    with (directory / 'stage-times.csv').open() as stream:
        for row in csv.DictReader(stream):
            block = int(row['block'])
            assert row['arm'] == arms[block] and row['swap_kib'] == '0'
            p = int(row['P'])
            b = 'cpu' if row['arm'] == 'independent' else 'htp'
            value = float(row['prefill_ms']) / 1000
            groups[p, b].append(value)
            trial = paired[int(row['pair']), int(row['round']), p]
            assert b not in trial
            trial[b] = value
    assert set(groups) == {(p, b) for p in (16, 24, 32) for b in COLORS}
    assert all(len(vals) == 6 for vals in groups.values())
    assert set(paired) == {(block, rep, p) for block in range(2) for rep in range(3) for p in (16, 24, 32)}
    counts = {p: sum(arm['cpu'] < arm['htp'] for key, arm in paired.items() if key[2] == p)
              for p in (16, 24, 32)}
    return groups, counts


def panel(ax, groups, estimator, *, offset, markersize, alpha, scatter_size):
    points = sorted({p for p, _ in groups})
    for backend, color in COLORS.items():
        direction = -1 if backend == 'cpu' else 1
        for p in points:
            vals = groups[p, backend]
            assert all(math.isfinite(v) and v > 0 for v in vals)
            ax.scatter([p + offset * direction] * len(vals), vals, color=color,
                       s=scatter_size, alpha=alpha, linewidths=0, zorder=2)
        ax.plot(points, [estimator(groups[p, backend]) for p in points],
                color=color, marker=MARKERS[backend], markersize=markersize, linewidth=1.7,
                solid_capstyle='round', zorder=3)
    ax.grid(axis='y', color='#E4E7EB', linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('新增 token 数 P', labelpad=8)
    ax.set_ylabel('纯 Prefill 耗时（秒）', labelpad=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', type=Path, default=HISTORICAL)
    parser.add_argument('--direct', type=Path, default=DIRECT)
    parser.add_argument('--output', type=Path, default=DIRECT / 'figures')
    args = parser.parse_args()
    old, excluded = historical_samples(args.historical)
    new, cpu_wins = direct_samples(args.direct)
    delta_old = {p: statistics.mean(old[p,'cpu']) - statistics.mean(old[p,'htp']) for p in range(24,65)}
    flips = sum(delta_old[p] * delta_old[p+1] < 0 for p in range(24,64))
    delta_new = {p: statistics.median(new[p,'cpu']) - statistics.median(new[p,'htp']) for p in (16,24,32)}
    assert flips == 8
    assert delta_new[16] < 0 and delta_new[24] < 0 < delta_new[32]

    font_path = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
    apply_figure_style(font='Arial Unicode MS' if font_path.exists() else 'Noto Sans CJK SC',
                       sizes=(12, 10, 9))
    plt.rcParams.update({'svg.fonttype':'none', 'savefig.bbox':None, 'axes.unicode_minus':False})

    fig, (a,b) = plt.subplots(1, 2, figsize=(12.4, 5.9), gridspec_kw={'width_ratios':(1.35, 1)})
    fig.subplots_adjust(left=.083,right=.968,bottom=.217,top=.743,wspace=.25)
    fig.text(.083,.923,'CPU 与 HTP 的 Prefill 交叉', fontsize=22, va='center')
    fig.text(.083,.870,'小米 14 Pro  ·  固定历史 H=4096  ·  batch 128  ·  F16 KV',
             fontsize=11, color='#435061', va='center')
    handles = [Line2D([], [], color=COLORS[backend], marker=MARKERS[backend],
                      linewidth=1.7, markersize=5, label=label)
               for backend,label in (('cpu','独立 CPU'),('htp','HTP offload'))]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.971,.928),
               ncol=2, columnspacing=1.05, handletextpad=.55, borderaxespad=0)

    panel(a,old,statistics.mean,offset=.12,markersize=3.4,alpha=.19,scatter_size=10)
    panel(b,new,statistics.median,offset=.28,markersize=5.7,alpha=.29,scatter_size=21)
    for ax,letter,title,subtitle in (
        (a,'a','旧扫描中的多次交叉','Qwen3.5-9B Q4_0  ·  CPU 6 线程  ·  HTP 主机 4 线程'),
        (b,'b','同一份 KV 上的直接交叉','Qwen3-8B Q4_0  ·  CPU 和 HTP 主机均为 4 线程')):
        panel_letter(ax,letter,dx=-.051,dy=1.105,fontsize=11)
        ax.text(0,1.105,title,transform=ax.transAxes,fontsize=13,va='bottom')
        ax.text(0,1.038,subtitle,transform=ax.transAxes,fontsize=9,color='#435061',va='bottom')

    a.set(xlim=(23,65),ylim=(1.22,6.62),xticks=[24,32,40,48,56,64],yticks=[2,3,4,5,6])
    a.text(24.1,6.38,f'相邻测点排序翻转 {flips} 次',color='#435061',fontsize=9,va='top')
    b.set(xlim=(14.2,34.0),ylim=(1.27,4.17),xticks=[16,24,32],yticks=[1.5,2,2.5,3,3.5,4])
    b.axvspan(24,32,color='#E6E9EF',alpha=.72,zorder=0)
    b.text(28,4.035,'交叉区间',color='#435061',fontsize=9,ha='center',va='top')
    for p,wins in cpu_wins.items():
        b.text(p,.035,f'{wins}/6',transform=b.get_xaxis_transform(),fontsize=9,
               color=COLORS['cpu'],ha='center')

    fig.text(.083,.111,f'a  散点为保留样本，连线为均值。每点每后端保留 9–10 次，按上尾规则剔除 {excluded}/820 个样本。',
             fontsize=9,color='#435061')
    fig.text(.083,.070,'b  散点为全部样本，连线为中位数。每点 6 对，蓝色分数为 CPU 配对胜场。阴影标出实测夹住的区间。',
             fontsize=9,color='#435061')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    out_of_bounds = [t.get_text() for t,box in texts
                     if not all(fig.bbox.contains(*p) for p in box.get_points())]
    overlaps = [(t.get_text(),u.get_text()) for i,(t,box) in enumerate(texts)
                for u,other in texts[i+1:] if box.overlaps(other)]
    if out_of_bounds or overlaps:
        raise RuntimeError(f'figure layout check failed: {out_of_bounds=} {overlaps=}')
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(args.output / f'prefill-crossing-comparison.{ext}', dpi=300)
    print(f'Checked {flips} historical flips, direct paired CPU wins {cpu_wins}, and figure layout.')


if __name__ == '__main__':
    main()
