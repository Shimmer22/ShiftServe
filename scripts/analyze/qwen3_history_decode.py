#!/usr/bin/env python3
"""核验 Qwen3 历史长度 sweep，导出原始重复、中文报告与图。"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
from statistics import median
from decode_attribution import read_case

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='仅纳入已完成且校验通过的条件')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'configs/qwen3-history-decode.json').read_text())
    out = ROOT / 'results' / plan['name']
    complete = (out / 'completion.json').exists() and json.loads((out / 'completion.json').read_text())['completed']
    assert complete or args.partial, '全量实验尚未完成；阶段验收请用 --partial'
    target = out if complete else out / 'partial'
    target.mkdir(exist_ok=True)
    pending = []
    rows = []
    for case in plan['cases']:
        folder = out / case['name']
        if not (folder / 'exit.json').exists():
            assert args.partial
            pending.append(case['name'])
            continue
        d, s, _ = read_case(folder)
        assert d['actual_H'] == case['history_tokens'] and d['actual_P'] == 0
        assert d['warmup_elapsed_ms'] >= 40000
        assert d['checkpoint_replay']['final_state_bitwise_equal'] and d['json_memory']['passed']
        assert d['config']['context_size'] == 8704 and d['ffn_compaction']['keep_ratio'] == 1
        formal = [r for r in d['rows'] if not r['warmup']]
        rows.append(dict(backend=case['backend'], history_tokens=d['actual_H'],
                         ms_per_token=s['ms_per_token'], repetitions_ms_per_token=s['repetitions_ms_per_token'],
                         token_ms=[r['token_ms'] for r in formal],
                         initialization_ms=d['model_context_init_ms'], history_build_ms=d['history_build_ms'],
                         input_prefill_ms=d['full_prefill_ms'], warmup_ms=d['warmup_elapsed_ms'],
                         major_faults=[r['resources_after']['major_faults'] - r['resources_before']['major_faults'] for r in formal]))
    assert rows, '没有通过验收的测量'
    summary = dict(complete=complete, pending=pending, model=plan['model_id'], protocol=plan['protocol'], rows=rows)
    with (target / 'measurements.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['backend', 'history_tokens', 'repetition', 'decode_ms_per_token'])
        for row in rows:
            for i, value in enumerate(row['repetitions_ms_per_token']):
                w.writerow([row['backend'], row['history_tokens'], i, value])
    import numpy as np
    fits = {}
    for backend in ['cpu', 'htp']:
        series = [r for r in rows if r['backend'] == backend]
        h = np.array([r['history_tokens'] for r in series], dtype=float)
        y = np.array([r['ms_per_token'] for r in series])
        fits[backend] = {}
        for name, x in [('linear_H', h), ('log2_H', np.log2(h))]:
            slope, intercept = np.polyfit(x, y, 1)
            residual = y - (intercept + slope * x)
            fits[backend][name] = dict(intercept=float(intercept), slope=float(slope),
                rmse_ms_per_token=float(np.sqrt(np.mean(residual**2))),
                r_squared=float(1 - np.sum(residual**2) / np.sum((y-y.mean())**2)))
    summary['descriptive_fits'] = fits
    (target / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location('figure_style', Path.home() / '.codex/skills/figure-style/kernel.py')
    style = importlib.util.module_from_spec(spec); spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(11, 10, 9))
    for scale, stem in [('log', 'history-decode'), ('linear', 'history-decode-linear')]:
        fig, ax = plt.subplots(figsize=(7.8, 4.7))
        for backend, label, color in [('cpu', 'CPU (6 threads)', '#0072B2'), ('htp', 'HTP offload (4 host threads)', '#D55E00')]:
            series = [r for r in rows if r['backend'] == backend]
            x = [r['history_tokens'] for r in series]
            for r in series:
                ax.scatter([r['history_tokens']] * 3, r['repetitions_ms_per_token'], color=color, alpha=.35, s=18, zorder=3)
            ax.plot(x, [r['ms_per_token'] for r in series], marker='D', markersize=5, color=color, label=label)
        if scale == 'log':
            ax.set_xscale('log', base=2)
        hs = [128, 256, 512, 1024, 2048, 4096, 8192]
        if scale == 'log':
            ax.set_xticks(hs, [str(h) for h in hs])
            ax.set_xlim(105, 10000)
        else:
            ax.set_xticks([0, 2048, 4096, 6144, 8192])
            ax.set_xlim(-180, 8500)
        ax.set_xlabel('History before decode (tokens)')
        ax.set_ylabel('Decode latency (ms/token)')
        ax.set_title('Qwen3-8B Q4_0: decode latency versus history' + ('' if complete else ' (partial)'), loc='left')
        if not complete:
            ax.text(.98, .04, 'Missing points: pending validation', transform=ax.transAxes, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=.2); ax.margins(x=.06, y=.12); ax.legend(loc='best')
        fig.text(.12, .035, 'Diamonds: median; dots: 3 repeats × 16 tokens. Context: 8704. KV offload / flash attention: off.', fontsize=9)
        fig.subplots_adjust(bottom=.19, top=.88, left=.12, right=.97)
        fig.canvas.draw()
        texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
        renderer = fig.canvas.get_renderer()
        collisions = [(a.get_text(), b.get_text()) for i, a in enumerate(texts) for b in texts[i+1:]
                      if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
        outside = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
        (target / (stem + '-qa.json')).write_text(json.dumps(dict(text_overlaps=collisions, outside=outside), indent=2))
        assert not collisions and not outside, 'figure text layout failed'
        for ext in ['png', 'svg', 'pdf']:
            fig.savefig(target / (stem + '.' + ext), dpi=300)
        plt.close(fig)
    lines = ['# Qwen3 历史长度与 Decode 时间', '', '验收状态：' + ('全部完成' if complete else '阶段验收；待完成：' + '、'.join(pending)), '', plan['protocol'], '',
             '历史长度指第一个计时 decode 前的 KV token 数；随后每轮推进16 token，再恢复同一状态。固定token回放用于性能测量，不代表自由生成质量。正常OK响应、独立两轮JSON记忆与检查点续跑另行验证。每点先累计至少40秒 decode 预热，初始化、历史建立、状态恢复及验证均不计入 decode；计时包含逐token调用、同步与logits计算。', '',
             '| 历史token | CPU ms/token | HTP offload ms/token |', '|---:|---:|---:|']
    for h in [128, 256, 512, 1024, 2048, 4096, 8192]:
        pair = {r['backend']: r['ms_per_token'] for r in rows if r['history_tokens'] == h}
        values = [f'{pair[b]:.2f}' if b in pair else '待完成' for b in ['cpu', 'htp']]
        lines.append(f'| {h} | {values[0]} | {values[1]} |')
    lines += ['', '表中为3轮均值的中位数。图中浅色点为每轮实测，菱形为中位数。单台手机串行按H递增、每H先CPU后HTP；热状态与时间顺序可能影响曲线，3轮是同进程重复，不是独立设备重复。8192历史使用已有4096 token序列重复扩展，所有H取其相同前缀，属于合成成本测量。', '',
              '![历史长度与decode时间](history-decode.png)', '',
              '原始输出在各条件的response.json、runtime.log和command.json；summary.json保留初始化/prefill、逐token数据与缺页数，measurements.csv保留全部正式重复。复现：python3 scripts/run/decode_attribution.py --config configs/qwen3-history-decode.json，然后运行python3 scripts/analyze/qwen3_history_decode.py。']
    lines += ['', '## 与理论的关系', '',
              '标准全注意力逐token decode 的KV读取和attention工作量随H近似线性；总耗时可粗略写成固定权重/运行时成本加cH。对数横轴上的直线对应a+b·log₂H，不能直接当作标准attention的复杂度。以下仅对各H中位数作等权、两参数描述性拟合，不作复杂度或因果检验。', '',
              '| 后端 | a+bH RMSE (ms/token) | a+bH R² | a+b·log₂H RMSE (ms/token) | a+b·log₂H R² |', '|---|---:|---:|---:|---:|']
    for backend, results in fits.items():
        a, b = results['linear_H'], results['log2_H']
        lines.append(f"| {backend} | {a['rmse_ms_per_token']:.2f} | {a['r_squared']:.3f} | {b['rmse_ms_per_token']:.2f} | {b['r_squared']:.3f} |")
    lines += ['', '![线性横轴对照](history-decode-linear.png)', '',
              '恢复记录：8192 CPU历史建立约767秒，首次驱动在600秒超时，手机进程继续并正常完成；保留driver-timeout.log/status-timeout.json，恢复时只取回已有结果，随后启动8192 HTP，长历史超时配置为2400秒。', '',
              '有限区间的拟合优劣不能证明渐近复杂度。本轮按H递增，频率/热状态与长度混杂；缓存、内核及固定成本也会影响实际墙钟。确认机制需要随机或逆序长度复测与attention分项计时，本次没有据此宣称log复杂度。']
    (target / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
