#!/usr/bin/env python3
"""绘制 H=4096 扩展三路 Prefill（P=16–2048：32–64 每 2 点 + 后续倍增）与 M(H) 对比图。"""

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
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/mainline-eval/prefill-3way-h4096-8b-extended'
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


def plot_curves(ax, sub, cleaned_sub, m_h: float, *, scatter_size: float = 10, lw_scale: float = 1.0):
    for arm in ('independent_cpu', 'shared_cpu', 'independent_htp'):
        pts = [c for c in cleaned_sub if c['arm'] == arm]
        ax.scatter([c['P'] for c in pts], [c['prefill_s'] for c in pts],
                   s=scatter_size, color=COLORS[arm], alpha=0.20, linewidths=0, zorder=1)
    ps = [r['P'] for r in sub]
    ax.plot(ps, [r['independent_cpu_median_ms'] / 1000.0 for r in sub], '-s',
            color=COLORS['independent_cpu'], ms=3.3 * lw_scale, lw=1.55 * lw_scale, zorder=3)
    ax.plot(ps, [r['shared_cpu_median_ms'] / 1000.0 for r in sub], '--^',
            color=COLORS['shared_cpu'], ms=3.3 * lw_scale, lw=1.45 * lw_scale, zorder=3)
    ax.plot(ps, [r['independent_htp_median_ms'] / 1000.0 for r in sub], '-o',
            color=COLORS['independent_htp'], ms=3.3 * lw_scale, lw=1.55 * lw_scale, zorder=3)
    ax.plot(ps, [r['htp_plus_M_HP_ms'] / 1000.0 for r in sub], '-.D',
            color=COLORS['htp_plus_m'], ms=2.9 * lw_scale, lw=1.45 * lw_scale, zorder=4)


def render(run_dir: Path):
    table, cleaned, summary, status = load_data(run_dir)
    setup_font()
    ind_t = int(status.get('ind_threads', 6))
    shr_t = int(status.get('shr_threads', ind_t))
    cpu_desc = f'CPU {ind_t} 线程（最优）' if ind_t == shr_t else f'独CPU {ind_t} / 共CPU {shr_t} 线程'
    m_h = summary['handoff']['4096']['exact_H_median_ms'] / 1000.0

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.2, 5.15), gridspec_kw={'width_ratios': (1.05, 1.25)})
    fig.subplots_adjust(left=0.068, right=0.982, top=0.71, bottom=0.175, wspace=0.19)
    fig.text(0.068, 0.945, 'H=4096 三路 Prefill 扩展扫描（32–64 每 2 点 + 66–2048 倍增）', fontsize=12.5, fontweight='bold')
    fig.text(0.068, 0.890, f'Xiaomi 14 Pro · Qwen3-8B Q4_0 · {cpu_desc} · HTP 主机 4 线程 · context 8192 · Q3+1.5×IQR 剔除异常后中位数',
             fontsize=8.5, color='#414B58')

    # Panel a: Full range P=16-2048 (log-log so both 16-64 and 128-2048 stay clear)
    plot_curves(ax_a, table, cleaned, m_h, scatter_size=8, lw_scale=0.95)
    ax_a.set_xscale('log', base=2)
    ax_a.set_yscale('log')
    ax_a.xaxis.set_major_locator(FixedLocator([16, 32, 64, 128, 256, 512, 1024, 2048]))
    ax_a.xaxis.set_major_formatter(FixedFormatter(['16', '32', '64', '128', '256', '512', '1024', '2048']))
    ax_a.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 250]))
    ax_a.yaxis.set_major_formatter(FixedFormatter(['1', '2', '5', '10', '20', '50', '100', '250']))
    ax_a.minorticks_off()
    ax_a.set_xlim(14, 2350)
    ax_a.set_ylim(1.1, 300)
    ax_a.axvspan(16, 64, color='#E8EEF4', alpha=0.55, zorder=0)
    ax_a.text(32, 230, '右侧放大区\nP=16–64', ha='center', va='top', fontsize=7.5, color='#334155')
    p2048 = table[-1]
    ax_a.text(2048, 22,
              f'P=2048\nHTP {p2048["independent_htp_median_ms"]/1000:.1f}s (+M {p2048["htp_plus_M_HP_ms"]/1000:.1f}s)\n独CPU {p2048["independent_cpu_median_ms"]/1000:.1f}s\n共CPU {p2048["shared_cpu_median_ms"]/1000:.1f}s',
              ha='right', va='top', fontsize=7.3, color='#334155',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CBD5E1', alpha=0.95))
    panel_letter(ax_a, 'a', dx=0.0, dy=1.04, fontsize=10)
    ax_a.text(0.06, 1.04, '全范围（对数坐标）：P=16–2048', transform=ax_a.transAxes, va='bottom', fontsize=9.5)
    ax_a.set_ylabel('耗时（秒）')
    ax_a.set_xlabel('新增 token 数 P（倍增扩展至 2048）', labelpad=6)
    ax_a.grid(axis='y', color='#DFE4E9', lw=0.6)
    ax_a.set_axisbelow(True)

    # Panel b: Dense crossing zone P=16-64 (linear coordinates with M(H) band)
    sub_b = [r for r in table if r['P'] <= 64]
    cleaned_b = [c for c in cleaned if c['P'] <= 64]
    ax_b.axhspan(0, m_h, color=COLORS['handoff_band'], alpha=0.65, zorder=0)
    ax_b.text(40, m_h + 0.14, f'底部等高色带：M(4096) ≈ {m_h:.2f} s',
              ha='center', va='bottom', fontsize=7.5, color='#7A410E', fontweight='bold')
    plot_curves(ax_b, sub_b, cleaned_b, m_h, scatter_size=12, lw_scale=1.0)
    ax_b.set_xlim(14.5, 65.5)
    ax_b.set_ylim(0, 8.8)
    ax_b.set_xticks([16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
    p31 = next(r for r in sub_b if r['P'] == 31)
    p32 = next(r for r in sub_b if r['P'] == 32)
    p40 = next(r for r in sub_b if r['P'] == 40)
    ax_b.annotate(f'P=31→32 HMX 切档: {p31["independent_htp_median_ms"]/1000:.2f}→{p32["independent_htp_median_ms"]/1000:.2f}s',
                  xy=(32, p32['independent_htp_median_ms'] / 1000.0),
                  xytext=(24.5, 1.35), fontsize=7.4, color='#7A410E',
                  arrowprops=dict(arrowstyle='->', color='#9A5416', lw=0.85))
    ax_b.annotate(f'P=40 波谷仍穿回: 独CPU {p40["independent_cpu_median_ms"]/1000:.2f}s < HTP+M {p40["htp_plus_M_HP_ms"]/1000:.2f}s',
                  xy=(40, p40['independent_cpu_median_ms'] / 1000.0),
                  xytext=(34.5, 7.65), fontsize=7.3, color='#176B9A',
                  arrowprops=dict(arrowstyle='->', color='#176B9A', lw=0.85))
    panel_letter(ax_b, 'b', dx=0.0, dy=1.04, fontsize=10)
    ax_b.text(0.055, 1.04, '密集区（线性坐标）：P=16–64（32–64 每 2 点）', transform=ax_b.transAxes, va='bottom', fontsize=9.5)
    ax_b.set_ylabel('耗时（秒）')
    ax_b.set_xlabel('新增 token 数 P', labelpad=6)
    ax_b.grid(axis='y', color='#DFE4E9', lw=0.6)
    ax_b.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=COLORS['independent_cpu'], marker='s', lw=1.6, ms=4.5, label='独立 CPU（纯 Prefill）'),
        Line2D([0], [0], color=COLORS['shared_cpu'], marker='^', ls='--', lw=1.5, ms=4.5, label='共享 CPU（纯 Prefill）'),
        Line2D([0], [0], color=COLORS['independent_htp'], marker='o', lw=1.6, ms=4.5, label='独立 HTP（纯 Prefill）'),
        Line2D([0], [0], color=COLORS['htp_plus_m'], marker='D', ls='-.', lw=1.5, ms=4.0, label='独立 HTP + 实测 M(H+P)'),
        Patch(facecolor=COLORS['handoff_band'], alpha=0.55, label='底部等高 M(4096) Handoff 色带'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.062, 0.858),
               ncol=3, columnspacing=1.3, frameon=False, fontsize=8.2)
    fig.text(0.068, 0.025,
             f'全量 {summary["raw_prefill_rows"]} 次 Prefill + {summary["raw_handoff_rows"]} 次 Handoff（n=4 轮 Latin 平衡交错，block 间 30s 冷却）；'
             f'异常剔除 {summary["excluded_prefill_rows"]}/{summary["raw_prefill_rows"]} Prefill、{summary["excluded_handoff_rows"]}/{summary["raw_handoff_rows"]} Handoff；top_token 一致率 100%。',
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

    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out_dir / f'prefill-3way-h4096-extended.{ext}', dpi=300)
    plt.close(fig)
    print('Saved figures to', out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    render(args.dir.resolve())


if __name__ == '__main__':
    main()
