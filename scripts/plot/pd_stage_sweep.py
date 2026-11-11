#!/usr/bin/env python3
"""从固定H阶段原始CSV画四条曲线，数值标记点不进入汇总。"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / '.codex/skills/figure-style'))
from kernel import apply_figure_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/20261111-pd-stage-sweep-h4096/stage-times.csv')
    args = parser.parse_args()
    out = args.input.parent / 'figures'
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(args.input.open()))
    assert len(rows) == 174 and {r['H'] for r in rows} == {'4096'}
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    apply_figure_style(font='Arial Unicode MS', sizes=(12, 10, 9))
    plt.rcParams['savefig.bbox'] = None
    plt.rcParams['axes.unicode_minus'] = False
    colors = {'cpu': '#2379AE', 'htp': '#C47524'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1.1, 1, 1.1]})
    fig.subplots_adjust(left=.06, right=.985, bottom=.28, top=.79, wspace=.32)
    aggregate = []
    for panel, (ax, stage, zoom) in enumerate(zip(axes, ['prefill', 'prefill', 'decode'], [False, True, False])):
        for backend in ['cpu', 'htp']:
            subset = [r for r in rows if r['stage'] == stage and r['backend'] == backend
                      and (not zoom or 26 <= int(r['tokens']) <= 38)]
            xs = sorted({int(r['tokens']) for r in subset})
            means = []
            for x in xs:
                group = [r for r in subset if int(r['tokens']) == x]
                assert len(group) == 3
                included = [float(r['elapsed_ms'])/1000 for r in group if r['numerical_match'] == 'True']
                flagged = [float(r['elapsed_ms'])/1000 for r in group if r['numerical_match'] != 'True']
                ax.scatter([x]*len(included), included, s=22, color=colors[backend], alpha=.32,
                           marker='o', linewidths=0, zorder=4)
                ax.scatter([x]*len(flagged), flagged, s=40, edgecolors=colors[backend],
                           facecolors='none', marker='D', linewidths=1, zorder=5)
                mean = statistics.mean(included) if included else np.nan
                means.append(mean)
                if not zoom:
                    aggregate.append(dict(stage=stage, backend=backend, tokens=x,
                        mean_s=None if not included else mean, n=len(included), flagged=len(flagged)))
            ax.plot(xs, means, '-s', ms=4, lw=1.7, color=colors[backend], label=backend.upper(), zorder=3)
        ax.grid(axis='y', color='#E5E7EB', lw=.6)
        ax.set_axisbelow(True)
        ax.margins(x=.05, y=.08)
        ax.set_ylabel('累计 Decode 耗时（秒）' if stage == 'decode' else 'Prefill 耗时（秒）')
        ax.set_xlabel('输出 token 数 D' if stage == 'decode' else '新增 token 数 P')
        if zoom:
            ax.set_xticks([26, 28, 30, 32, 34, 36, 38])
            ax.set_title('b  短输入区域：每 2 token 取点', pad=15)
        elif stage == 'prefill':
            ax.set_xticks([16, 64, 128, 192, 256])
            ax.set_ylim(bottom=0)
            ax.set_title('a  Prefill：新增输入长度扫描', pad=15)
            ax.legend(loc='upper left')
        else:
            ax.set_xticks([1, 64, 128, 192, 256, 320])
            ax.set_ylim(bottom=0)
            ax.set_title('c  Decode：连续逐 token 执行', pad=15)
    fig.text(.06, .94, 'H=4096：CPU / HTP 的 Prefill 与 Decode 耗时', ha='left', fontsize=12)
    fig.text(.06, .875, 'Qwen3.5-9B-Q4_0 · batch 128 · 四线程 · F16 状态；Decode 固定 P=40。', fontsize=10)
    fig.text(.06, .16, '方点连线：三轮均值；浅色圆点：各轮原始值。D 曲线来自每轮同一次连续运行的累计检查点。', fontsize=10)
    fig.text(.06, .105, '空心菱形：P=28 的数值一致性标记点（3/3 轮），保留原值但不进入均值，连线在此断开。', fontsize=10)
    fig.text(.06, .05, '仅测固定 token 的阶段耗时；不包含初始化、历史准备、状态恢复或迁移，不代表任务质量或真实 Tool Gap 收益。', fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text()) for i, (a, box) in enumerate(texts)
                for b, other in texts[i+1:] if box.overlaps(other)]
    outside = [t.get_text() for t, box in texts
               if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
    save = dict(overlaps=overlaps, outside=outside)
    (out / 'layout-check.json').write_text(json.dumps(save, ensure_ascii=False, indent=2))
    (out / 'curve-data.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / ('pd-stage-curves.' + ext), dpi=300, bbox_inches=None)
    # 导出每个面板供逐面板视觉检查。
    from PIL import Image
    rendered = Image.open(out / 'pd-stage-curves.png')
    scale = 300 / fig.dpi
    for name, ax in zip(['prefill', 'prefill-short', 'decode'], axes):
        box = ax.get_tightbbox(renderer)
        crop = (max(0, int(box.x0*scale)-10), max(0, int(rendered.height-box.y1*scale)-10),
                min(rendered.width, int(box.x1*scale)+10), min(rendered.height, int(rendered.height-box.y0*scale)+10))
        rendered.crop(crop).save(out / (name + '.png'))
    print(json.dumps(save, ensure_ascii=False))
    print(out / 'pd-stage-curves.png')


if __name__ == '__main__':
    main()
