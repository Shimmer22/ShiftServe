#!/usr/bin/env python3
"""绘制 1k/2k/4k 三路 Prefill（独立CPU / 共享CPU / 独立HTP）与 M(H) 交叠图及放大图。"""

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
DEFAULT_DIR = ROOT / 'results/mainline-eval/prefill-3way-h1k2k4k-8b'
sys.path.insert(0, os.environ.get('A2O_FIGURE_STYLE_DIR', str(Path.home() / '.codex/skills/figure-style')))
from kernel import apply_figure_style, panel_letter

COLORS = {
    'independent_cpu': '#176B9A',
    'shared_cpu': '#258565',
    'independent_htp': '#C2671D',
    'htp_plus_m': '#78519D',
    'handoff_band': '#F3C69F',
}


def load_data(run_dir: Path):
    table = []
    for r in csv.DictReader((run_dir / 'summary-table.csv').open()):
        parsed = {k: float(v) for k, v in r.items()}
        parsed['H'] = int(r['H'])
        parsed['P'] = int(r['P'])
        table.append(parsed)
    cleaned = []
    for r in csv.DictReader((run_dir / 'cleaned-stage-times.csv').open()):
        cleaned.append({
            'arm': r['arm'],
            'H': int(r['H']),
            'P': int(r['P']),
            'prefill_s': float(r['prefill_ms']) / 1000.0,
        })
    summary = json.loads((run_dir / 'summary.json').read_text())
    status_path = run_dir / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    return table, cleaned, summary, status


def setup_font():
    font_path = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font = 'Noto Sans CJK SC'
    apply_figure_style(font=font, sizes=(9.5, 8.5, 7.5), grid=False)
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['axes.unicode_minus'] = False


def render_figure(table, cleaned, summary, status, out_dir: Path, *, p_max: int, stem: str, title: str, xticks: list[int], ylim_max: float):
    ind_t = int(status.get('ind_threads', 4))
    shr_t = int(status.get('shr_threads', ind_t))
    cpu_desc = f'CPU {ind_t} 线程（最优）' if ind_t == shr_t else f'独CPU {ind_t} / 共CPU {shr_t} 线程'
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.85), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.685, bottom=0.185, wspace=0.14)
    fig.text(0.065, 0.948, title, fontsize=12, fontweight='bold')
    fig.text(0.065, 0.892, f'Xiaomi 14 Pro · Qwen3-8B Q4_0 · {cpu_desc} · HTP 主机 4 线程 · Q3+1.5×IQR 剔除异常后中位数',
             fontsize=8.5, color='#414B58')

    histories = sorted({r['H'] for r in table})
    for idx, (h, ax) in enumerate(zip(histories, axes)):
        sub = [r for r in table if r['H'] == h and r['P'] <= p_max]
        ps = [r['P'] for r in sub]
        m_h = summary['handoff'][str(h)]['exact_H_median_ms'] / 1000.0

        # Bottom Handoff M(H) reference band
        ax.axhspan(0, m_h, color=COLORS['handoff_band'], alpha=0.65, zorder=0)
        band_x = (ps[0] + ps[-1]) / 2
        ax.text(band_x, m_h + (0.22 if p_max > 40 else 0.16), f'底部色带：M(H) ≈ {m_h:.2f} s',
                ha='center', va='bottom', fontsize=7.2, color='#7A410E', fontweight='bold')

        # Scatter cleaned raw points subtly
        for arm, mk in (('independent_cpu', 's'), ('shared_cpu', '^'), ('independent_htp', 'o')):
            pts = [c for c in cleaned if c['arm'] == arm and c['H'] == h and c['P'] <= p_max]
            ax.scatter([c['P'] for c in pts], [c['prefill_s'] for c in pts],
                       s=11, color=COLORS[arm], alpha=0.22, linewidths=0, zorder=1)

        # Median lines
        ax.plot(ps, [r['independent_cpu_median_ms'] / 1000.0 for r in sub], '-s',
                color=COLORS['independent_cpu'], ms=3.8, lw=1.6, zorder=3)
        ax.plot(ps, [r['shared_cpu_median_ms'] / 1000.0 for r in sub], '--^',
                color=COLORS['shared_cpu'], ms=3.8, lw=1.5, zorder=3)
        ax.plot(ps, [r['independent_htp_median_ms'] / 1000.0 for r in sub], '-o',
                color=COLORS['independent_htp'], ms=3.8, lw=1.6, zorder=3)
        ax.plot(ps, [r['htp_plus_M_HP_ms'] / 1000.0 for r in sub], '-.D',
                color=COLORS['htp_plus_m'], ms=3.4, lw=1.5, zorder=4)

        if h == 4096:
            p32 = next((r for r in sub if r['P'] == 32), None)
            p31 = next((r for r in sub if r['P'] == 31), None)
            if p31 and p32:
                y31 = p31['independent_htp_median_ms'] / 1000.0
                y32 = p32['independent_htp_median_ms'] / 1000.0
                if y31 - y32 > 0.25:
                    ax.annotate(f'P=32 HMX 切档: {y31:.2f}→{y32:.2f}s',
                                xy=(32, y32), xytext=(33.5 if p_max > 40 else 20.5, 1.35 if p_max > 40 else 1.05),
                                fontsize=7.2, color='#7A410E',
                                arrowprops=dict(arrowstyle='->', color='#9A5416', lw=0.85))

        panel_letter(ax, 'abc'[idx], dx=0.0, dy=1.04, fontsize=10)
        ax.text(0.08, 1.04, f'历史 H={h}', transform=ax.transAxes, va='bottom', fontsize=9.5)
        ax.set_xticks(xticks)
        ax.set_xlim(min(ps) - 1.5, max(ps) + 1.5)
        ax.set_ylim(0, ylim_max)
        ax.grid(axis='y', color='#DFE4E9', lw=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('耗时（秒）')
    fig.supxlabel('新增 token 数 P', x=0.525, y=0.085, fontsize=9.5)

    handles = [
        Line2D([0], [0], color=COLORS['independent_cpu'], marker='s', lw=1.6, ms=4.5, label='独立 CPU（纯 Prefill）'),
        Line2D([0], [0], color=COLORS['shared_cpu'], marker='^', ls='--', lw=1.5, ms=4.5, label='共享 CPU（纯 Prefill）'),
        Line2D([0], [0], color=COLORS['independent_htp'], marker='o', lw=1.6, ms=4.5, label='独立 HTP（纯 Prefill）'),
        Line2D([0], [0], color=COLORS['htp_plus_m'], marker='D', ls='-.', lw=1.5, ms=4.0, label='独立 HTP + 实测 M(H+P)'),
        Patch(facecolor=COLORS['handoff_band'], alpha=0.55, label='底部等高 M(H) Handoff 色带'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.06, 0.858),
               ncol=3, columnspacing=1.3, frameon=False, fontsize=8.2)
    fig.text(0.065, 0.025, f'全量 504 次 Prefill + 48 次 Handoff（n=4 轮平衡交错）；异常剔除 {summary["excluded_prefill_rows"]}/504 条 Prefill、{summary["excluded_handoff_rows"]}/48 条 Handoff。',
             fontsize=7.8, color='#475569')

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

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=300)
    plt.close(fig)


def auto_ylim(table, p_max: int) -> float:
    sub = [r for r in table if r['P'] <= p_max]
    peak = max(
        max(r['independent_cpu_median_ms'], r['shared_cpu_median_ms'], r['htp_plus_M_HP_ms']) / 1000.0
        for r in sub
    )
    return round(peak * 1.15, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    run_dir = args.dir.resolve()
    table, cleaned, summary, status = load_data(run_dir)
    setup_font()
    out_dir = run_dir / 'figures'
    render_figure(
        table, cleaned, summary, status, out_dir,
        p_max=64, stem='prefill-3way-h1k2k4k',
        title='1k / 2k / 4k 三路 Prefill 与 M(H) 交接对比（P=16–64）',
        xticks=[16, 24, 32, 40, 48, 56, 64], ylim_max=auto_ylim(table, 64),
    )
    render_figure(
        table, cleaned, summary, status, out_dir,
        p_max=36, stem='prefill-3way-zoom-16-36',
        title='1k / 2k / 4k 交叉放大图（P=16–36，含底部 M(H) 色带）',
        xticks=[16, 20, 24, 28, 32, 36], ylim_max=auto_ylim(table, 36),
    )
    print('Saved figures to', out_dir)


if __name__ == '__main__':
    main()
