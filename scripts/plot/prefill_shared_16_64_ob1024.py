#!/usr/bin/env python3
"""绘制 H=4096 同进程配对（10 轮交错，16–64 每 4 点，ob1024）共享 CPU vs HTP 纯 Prefill 曲线。"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/mainline-eval/prefill-shared-16-64-h4096-8b-ob1024'
sys.path.insert(0, os.environ.get('A2O_FIGURE_STYLE_DIR', str(Path.home() / '.codex/skills/figure-style')))
from kernel import apply_figure_style

COLORS = {
    'cpu': '#258565',
    'htp': '#C2671D',
    'htp_plus_m': '#78519D',
    'handoff_band': '#F3C69F',
}


def setup_font():
    font_path = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font = 'Noto Sans CJK SC'
    apply_figure_style(font=font, sizes=(10, 9, 8), grid=False)
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['axes.unicode_minus'] = False


def render(run_dir: Path):
    table = [{k: float(v) for k, v in r.items()} | {'P': int(r['P']), 'paired_cpu_wins': int(r['paired_cpu_wins'])}
             for r in csv.DictReader((run_dir / 'summary-table.csv').open())]
    cleaned = [{'backend': r['backend'], 'P': int(r['P']), 'elapsed_s': float(r['elapsed_ms']) / 1000.0}
               for r in csv.DictReader((run_dir / 'cleaned-stage-times.csv').open())]
    summary = json.loads((run_dir / 'summary-cleaned.json').read_text())
    setup_font()

    m_h_s = summary['M_H_ms'] / 1000.0
    ps = [r['P'] for r in table]
    y_max = max(
        max(r['cpu_median_ms'] / 1000.0 for r in table),
        max(r['htp_plus_M_H_ms'] / 1000.0 for r in table),
        max(c['elapsed_s'] for c in cleaned),
    )
    ylim_top = max(5.6, y_max * 1.15)

    fig, ax = plt.subplots(figsize=(9.6, 5.25))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.735, bottom=0.19)
    fig.text(0.085, 0.94, 'H=4096 同进程配对交错（ob1024）：共享 CPU（6 线程）vs HTP 纯 Prefill（P=16–64）',
             fontsize=12.5, fontweight='bold')
    fig.text(0.085, 0.885,
             f'Xiaomi 14 Pro · Qwen3-8B Q4_0 (t6-final-ob1024) · 10 轮逐点配对（CPU先/HTP先各5次） · 中位CV {summary["median_cv_pct"]:.2f}% · Q3+1.5×IQR 剔除异常',
             fontsize=8.4, color='#414B58')

    # Find pure-prefill crossing pair dynamically
    cross_lo, cross_hi = None, None
    for a, b in zip(table, table[1:]):
        if a['cpu_median_ms'] <= a['htp_median_ms'] and b['htp_median_ms'] < b['cpu_median_ms']:
            cross_lo, cross_hi = a['P'], b['P']
            break
    if cross_lo is not None:
        ax.axvspan(cross_lo, cross_hi, color='#E4EAF1', alpha=0.80, zorder=0)
        ax.text((cross_lo + cross_hi) / 2.0, ylim_top * 0.96, f'纯Prefill交点\nP={cross_lo}–{cross_hi}',
                ha='center', va='top', fontsize=7.8, color='#334155')

    ax.axhspan(0, m_h_s, color=COLORS['handoff_band'], alpha=0.65, zorder=0)
    ax.text((ps[0] + ps[-1]) / 2.0 - 6.0, m_h_s + 0.08, f'底部等高色带：同 build 实测 M(4096) ≈ {m_h_s:.2f} s',
            ha='center', va='bottom', fontsize=7.8, color='#7A410E', fontweight='bold')

    for b, offset in (('cpu', -0.30), ('htp', 0.30)):
        pts = [c for c in cleaned if c['backend'] == b]
        ax.scatter([c['P'] + offset for c in pts], [c['elapsed_s'] for c in pts],
                   s=16, color=COLORS[b], alpha=0.26, linewidths=0, zorder=2)

    ax.plot(ps, [r['cpu_median_ms'] / 1000.0 for r in table], '-^',
            color=COLORS['cpu'], ms=4.8, lw=1.9, zorder=4)
    ax.plot(ps, [r['htp_median_ms'] / 1000.0 for r in table], '-o',
            color=COLORS['htp'], ms=4.8, lw=1.9, zorder=4)
    ax.plot(ps, [r['htp_plus_M_H_ms'] / 1000.0 for r in table], '-.D',
            color=COLORS['htp_plus_m'], ms=3.8, lw=1.6, zorder=4)

    # HMX step callout if P=28 and P=32 exist
    by_p = {r['P']: r for r in table}
    if 28 in by_p and 32 in by_p:
        p28, p32 = by_p[28], by_p[32]
        ax.annotate(f'P=28→32 HMX 切档: {p28["htp_median_ms"]/1000:.2f}→{p32["htp_median_ms"]/1000:.2f}s',
                    xy=(32, p32['htp_median_ms'] / 1000.0), xytext=(17.5, ylim_top * 0.87),
                    fontsize=8.0, color='#7A410E',
                    arrowprops=dict(arrowstyle='->', color='#9A5416', lw=0.95))

    for r in table:
        w = r['paired_cpu_wins']
        col = COLORS['cpu'] if w >= 6 else (COLORS['htp'] if w <= 4 else '#475569')
        ax.text(r['P'], 0.76, f'{w}/10', ha='center', va='bottom', fontsize=7.0, color=col)
    ax.text(ps[0] - 1.0, 1.04, 'CPU胜场/10:', ha='left', va='bottom', fontsize=7.1, color='#475569')

    ax.set_xlim(ps[0] - 1.8, ps[-1] + 1.8)
    ax.set_ylim(0, ylim_top)
    ax.set_xticks(ps)
    ax.set_xlabel('新增 token 数 P（16–64 每 4 点）', labelpad=7)
    ax.set_ylabel('耗时（秒）')
    ax.grid(axis='y', color='#DFE4E9', lw=0.6)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=COLORS['cpu'], marker='^', lw=1.9, ms=5.0, label='共享 CPU（6 线程，纯 Prefill）'),
        Line2D([0], [0], color=COLORS['htp'], marker='o', lw=1.9, ms=5.0, label='HTP offload（主机 4 线程，纯 Prefill）'),
        Line2D([0], [0], color=COLORS['htp_plus_m'], marker='D', ls='-.', lw=1.6, ms=4.2, label='HTP + M(4096) 参考线'),
        Patch(facecolor=COLORS['handoff_band'], alpha=0.65, label='底部等高 M(4096) 色带'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.080, 0.842),
               ncol=2, columnspacing=1.8, frameon=False, fontsize=8.5)
    fig.text(0.085, 0.030,
             f'共 {summary["raw_rows"]} 条后端样本（{summary["raw_rows"]//2} 对），按各 P×后端上尾规则剔除 {summary["excluded_rows"]}/{summary["raw_rows"]} 个慢点；下方数字为 10 轮配对中共享 CPU 获胜轮数。',
             fontsize=8.0, color='#475569')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in labels]
    for i, (txt, box) in enumerate(boxes):
        if not fig.bbox.contains(*box.min) or not fig.bbox.contains(*box.max):
            raise ValueError(f'Text outside figure: {txt}')
        for otxt, obox in boxes[i + 1:]:
            if box.overlaps(obox):
                raise ValueError(f'Text overlap: {txt!r} vs {otxt!r}')

    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out_dir / f'prefill-shared-16-64-ob1024.{ext}', dpi=300)
    plt.close(fig)
    print('Saved figures to', out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    render(args.dir.resolve())


if __name__ == '__main__':
    main()
