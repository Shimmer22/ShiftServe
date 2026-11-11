#!/usr/bin/env python3
"""密集区独立大图，按用户要求统一统计所有有效耗时。"""
import csv
import argparse
import json
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / '.codex/skills/figure-style'))
from kernel import apply_figure_style

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=ROOT / 'results/20261111-pd-prefill-dense-h4096/stage-times.csv')
parser.add_argument('--clean-upper-tail', action='store_true', help='各P/后端独立按Q3+1.5IQR剔除异常慢点')
args = parser.parse_args()
config = json.loads((args.input.parent / 'plan.json').read_text())['config']
repetitions = config['rounds']
cpu_threads = config.get('cpu_threads', 4)
out = args.input.parent / ('figures-cleaned' if args.clean_upper_tail else 'figures')
out.mkdir(exist_ok=True)
rows = list(csv.DictReader(args.input.open()))
assert len(rows) == len(config['P']) * repetitions * 2 and {r['stage'] for r in rows} == {'prefill'}
assert {r['H'] for r in rows} == {'4096'}
for row in rows:
    if 'n_threads' in row:
        assert int(row['n_threads']) == int(row['n_threads_batch']) == (cpu_threads if row['backend'] == 'cpu' else 4)
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
apply_figure_style(font='Arial Unicode MS', sizes=(14, 12, 11))
plt.rcParams['savefig.bbox'] = None
plt.rcParams['axes.unicode_minus'] = False
wide = len(config['P']) > 15
if wide:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    fig.subplots_adjust(left=.065, right=.98, bottom=.32, top=.77, wspace=.25)
    left = .065
else:
    fig, ax = plt.subplots(figsize=(10, 7))
    axes = [ax]
    fig.subplots_adjust(left=.12, right=.97, bottom=.25, top=.8)
    left = .12
colors = {'cpu': '#2379AE', 'htp': '#C47524'}
data = []
excluded = []
thresholds = {}
for backend in ['cpu', 'htp']:
    xs = sorted({int(r['P']) for r in rows})
    means = []
    for x in xs:
        vals = [float(r['elapsed_ms'])/1000 for r in rows if r['backend'] == backend and int(r['P']) == x]
        assert len(vals) == repetitions
        all_vals = list(vals)
        if args.clean_upper_tail:
            ordered = sorted(vals)
            def quantile(q):
                pos = (len(ordered)-1)*q
                i = int(pos)
                return ordered[i] + (ordered[min(i+1, len(ordered)-1)]-ordered[i])*(pos-i)
            q1, q3 = quantile(.25), quantile(.75)
            threshold = q3 + 1.5*(q3-q1)
            thresholds[(backend, x)] = threshold
            vals = [v for v in vals if v <= threshold]
            assert len(vals) >= 2
            excluded.extend(dict(row, threshold_s=threshold, exclusion_rule='elapsed_s > Q3 + 1.5*IQR')
                            for row in rows if row['backend'] == backend and int(row['P']) == x
                            and float(row['elapsed_ms'])/1000 > threshold)
        means.append(statistics.mean(vals))
        data.append(dict(backend=backend, P=x, n=len(vals), original_n=repetitions,
                         excluded_n=len(all_vals)-len(vals), mean_s=means[-1], median_s=statistics.median(vals),
                         stdev_s=statistics.stdev(vals), raw_s=vals, original_raw_s=all_vals))
    label = f'CPU（{cpu_threads} 线程）' if backend == 'cpu' else 'HTP（主机 4 线程）'
    for index, ax in enumerate(axes):
        selected = [r for r in data if r['backend'] == backend and (index == 0 or r['P'] <= 64)]
        for point in selected:
            ax.scatter([point['P']]*point['n'], point['raw_s'], color=colors[backend],
                       s=18 if wide else 40, alpha=.25 if wide else .35, linewidths=0, zorder=4)
            if wide and index == 1:
                for value in point['raw_s']:
                    if value > 7:
                        ax.scatter([point['P']], [6.8], marker='^', color=colors[backend], s=32, zorder=5)
                        ax.text(point['P'], 6.96, f'{value:.1f}s', color=colors[backend], ha='center', fontsize=12)
        ax.plot([r['P'] for r in selected], [r['mean_s'] for r in selected], '-s', color=colors[backend],
                label=label, lw=1.7 if wide else 2.2, ms=3.5 if wide else 7, zorder=3)
for index, ax in enumerate(axes):
    ax.set_xticks(([24, 40, 64, 80, 96, 128] if index == 0 else list(range(24, 65, 4))) if wide else xs)
    ax.set_xlabel('新增 token 数 P', labelpad=12)
    ax.set_ylabel('Prefill 耗时（秒）', labelpad=12)
    ax.margins(x=.04, y=.14)
    ax.grid(axis='y', color='#E5E7EB', lw=.7)
    ax.set_axisbelow(True)
    if wide:
        ax.set_title('a  全范围：P=24–128' if index == 0 else 'b  逐整数区域：P=24–64', pad=16)
        if index == 1:
            ax.set_ylim(1, 7.4)
            ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    if index == 0:
        ax.legend(loc='best')
title = 'Prefill 长度扫描：清理异常慢点后' if args.clean_upper_tail else f'Prefill 长度扫描：每点每后端 {repetitions} 次'
fig.text(left, .94 if wide else .93, title, fontsize=14)
fig.text(left, .865 if wide else .86, 'H=4096 · Qwen3.5-9B-Q4_0 · batch 128 · F16 状态', fontsize=12)
if args.clean_upper_tail:
    counts = [point['n'] for point in data]
    fig.text(left, .21 if wide else .13, f'各 P / 后端独立以 Q3＋1.5×IQR 为上界，剔除 {len(excluded)}/{len(rows)} 个异常慢点；每组保留 {min(counts)}–{max(counts)} 次。', fontsize=12)
    fig.text(left, .16 if wide else .07, '方点连线：保留样本均值；浅色圆点：保留样本。原始数据与剔除清单另存。', fontsize=12)
else:
    fig.text(left, .21 if wide else .13, f'每个 P、每个后端各 {repetitions} 次；方点连线为均值，浅色圆点为每次原始值。', fontsize=12)
    fig.text(left, .16 if wide else .07, '所有有效耗时统一纳入统计；不包含初始化、历史准备、状态恢复和迁移。', fontsize=12)
if wide:
    fig.text(left, .11, '不包含初始化、历史准备、状态恢复和迁移。' if args.clean_upper_tail else '右图放大常见耗时区域；超过 7 秒的原始点以向上三角及实际秒数标注，完整位置见左图。', fontsize=12)
    fig.text(left, .06, '66、68、72、80、96 为原扫描后的追加批次；两批次分别乱序执行，未在同一时段交错测量。', fontsize=12)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
         if t.get_visible() and t.get_text().strip()]
overlaps = [(a.get_text(), b.get_text()) for i, (a, box) in enumerate(texts)
            for b, other in texts[i+1:] if box.overlaps(other)]
outside = [t.get_text() for t, box in texts if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
(out / 'layout-check.json').write_text(json.dumps(dict(overlaps=overlaps, outside=outside), ensure_ascii=False, indent=2))
(out / 'curve-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
if args.clean_upper_tail:
    kept_rows = [r for r in rows if float(r['elapsed_ms'])/1000 <= thresholds[(r['backend'], int(r['P']))]]
    assert len(kept_rows) + len(excluded) == len(rows)
    for name, records in [('cleaned-stage-times.csv', kept_rows), ('excluded-stage-times.csv', excluded)]:
        if records:
            with (out / name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
                writer.writeheader();writer.writerows(records)
    (out / 'cleaning-summary.json').write_text(json.dumps(dict(
        rule='每个P和后端单独计算线性插值Q1/Q3；仅剔除elapsed_s > Q3+1.5*IQR，保留低尾',
        input_rows=len(rows), excluded_rows=len(excluded), kept_rows=len(kept_rows),
        min_group_n=min(counts), max_group_n=max(counts)), ensure_ascii=False, indent=2))
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(out / ('prefill-dense.' + ext), dpi=300)
if wide:
    from PIL import Image
    image = Image.open(out / 'prefill-dense.png')
    scale = 300 / fig.dpi
    for name, ax in zip(['overview', 'integer-region'], axes):
        box = ax.get_tightbbox(renderer)
        crop = (max(0, int(box.x0*scale)-12), max(0, int(image.height-box.y1*scale)-12),
                min(image.width, int(box.x1*scale)+12), min(image.height, int(image.height-box.y0*scale)+12))
        image.crop(crop).save(out / (name + '.png'))
print(json.dumps(dict(overlaps=overlaps, outside=outside), ensure_ascii=False))
print(out / 'prefill-dense.png')
