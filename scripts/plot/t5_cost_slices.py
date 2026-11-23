#!/usr/bin/env python3
"""绘制 T5 已验收的三路返回后耗时切片。"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / 'results/mainline-eval/t5-8b-routing'
sys.path.insert(0, str(Path.home() / '.codex/skills/figure-style'))
from kernel import apply_figure_style, panel_letter

ROUTES = {
    'CC': ('#176B9A', 's', '-'),
    'HC-Full': ('#B66B22', '^', '--'),
    'HC-Compact': ('#78519D', 'D', '-.'),
}
SLICES = {
    1792: 'short-h-supplement/h1792',
    2048: 'retry-p5/h2048',
    4096: 'retry-p5/h4096',
}


def load_data(root):
    with (root / 'acceptance/fig7a-truth.csv').open(newline='') as f:
        accepted = {(int(row['H']), int(row['P'])): row for row in csv.DictReader(f)}
    groups = defaultdict(dict)
    for h, slice_name in SLICES.items():
        for path in sorted((root / slice_name / 'raw/shared').glob('*.json')):
            sample = json.loads(path.read_text())
            shape = (sample['H'], sample['P'])
            row = accepted[shape]
            if shape[0] != h or sample['D'] != int(row['D']) or sample['rep'] in groups[shape]:
                raise ValueError(f'样本与验收切片不符: {path}')
            groups[shape][sample['rep']] = sample['row']['paired_truth']
    if groups.keys() != accepted.keys():
        raise ValueError('原始点与验收汇总的覆盖范围不同')
    for shape, samples in groups.items():
        row = accepted[shape]
        if len(samples) != int(row['pairs']):
            raise ValueError(f'{shape} 的重复数不符')
        for route in ROUTES:
            median = statistics.median(s[route]['post_return_ms'] for s in samples.values())
            field = route.lower().replace('-', '_') + '_post_return_ms'
            if not math.isclose(median, float(row[field])):
                raise ValueError(f'{shape} {route} 的中位数与验收表不同')
    return groups, accepted


def plot(groups, accepted, output):
    font_path = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        font = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font = 'Noto Sans CJK SC'
    apply_figure_style(font=font, sizes=(9, 8, 7), grid=False)
    matplotlib.rcParams['svg.fonttype'] = 'none'
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 88 / 25.4),
                             gridspec_kw={'width_ratios': [0.83, 1, 1]})
    fig.subplots_adjust(left=0.098, right=0.98, top=0.68, bottom=0.25, wspace=0.25)
    fig.text(0.098, 0.94, 'T5 三条执行路径在不同历史长度下的耗时', weight='bold')
    fig.text(0.098, 0.865, '小米 14 Pro · Qwen3-8B Q4_0 · D=32 · Gap=1,000 ms · K=1,536',
             fontsize=8, color='#414B58')
    handles = []
    for index, (h, ax) in enumerate(zip(SLICES, axes)):
        points = sorted(p for history, p in groups if history == h)
        for route, (color, marker, style) in ROUTES.items():
            key = route.lower().replace('-', '_') + '_post_return_ms'
            medians = [float(accepted[h, p][key]) / 1000 for p in points]
            line, = ax.plot(points, medians, style, color=color, marker=marker,
                            markeredgecolor='white', markeredgewidth=0.4, markersize=4.6,
                            linewidth=1.35, label=route, zorder=3)
            if index == 0:
                handles.append(line)
            for p in points:
                times = [s[route]['post_return_ms'] / 1000 for s in groups[h, p].values()]
                ax.scatter([p] * len(times), times, s=15, color=color, alpha=0.30,
                           linewidths=0, zorder=2)
        panel_letter(ax, 'abc'[index], dx=0, dy=1.045, fontsize=10)
        ax.text(0.085, 1.045, f'历史 H={h}', transform=ax.transAxes, va='bottom')
        ax.set_xticks(points)
        ax.margins(x=0.08)
        ax.set_ylim((6, 12.5) if h == 1792 else (7, 34))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2 if h == 1792 else 5))
        ax.yaxis.grid(color='#DFE4E9', linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('工具返回后总耗时（秒）')
    fig.supxlabel('新增 token 数 P', x=0.54, y=0.135, fontsize=9)
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.092, 0.790),
               ncol=3, columnspacing=1.6, handlelength=2.2)
    fig.text(0.098, 0.058, '浅色圆点为单次实测，带标记折线为中位数（每点 3 次配对）。',
             fontsize=8, color='#414B58')
    fig.text(0.098, 0.022, '左图纵轴单独缩放。折线仅连接已测点。',
             fontsize=8, color='#414B58')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [text for text in fig.findobj(matplotlib.text.Text)
              if text.get_visible() and text.get_text().strip()]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in labels]
    for i, (label, box) in enumerate(boxes):
        if not fig.bbox.contains(*box.min) or not fig.bbox.contains(*box.max):
            raise ValueError(f'文字超出图边界: {label}')
        if any(box.overlaps(other) for _, other in boxes[i + 1:]):
            raise ValueError(f'图中文字重叠: {label}')
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f't5-cost-slices.{ext}', dpi=300, bbox_inches=fig.bbox_inches)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    groups, accepted = load_data(args.results)
    plot(groups, accepted, args.output or args.results / 'figures')


if __name__ == '__main__':
    main()
