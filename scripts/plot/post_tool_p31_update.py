#!/usr/bin/env python3
"""更新原策略图注释并单列配对复测，禁止混合新旧样本统计。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from observations import audit_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--retest', type=Path, required=True)
    args = parser.parse_args()
    out = args.campaign / 'figures'
    source = out / 'plot-data.json'
    rows = json.loads(source.read_text())['rows']
    summary = json.loads((args.retest / 'summary.json').read_text())
    assert {r['H'] for r in summary} == {2048, 4096}
    assert all(r['pairs'] == 20 and r['routes'] == {'CC': 20} for r in summary)
    spec = importlib.util.spec_from_file_location('style', Path.home() / '.codex/skills/figure-style/kernel.py')
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(10, 9, 8))
    qa = {}

    def save(fig, name):
        qa[name] = audit_geometry(fig)
        assert not qa[name]['text_collisions'] and not qa[name]['outside_canvas'], qa[name]
        for suffix in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{suffix}', dpi=300)
        plt.close(fig)

    policies = [('after_return', 'CC'), ('after_return', 'HC'), ('after_return', 'router'),
                ('always', 'router'), ('cost_aware', 'router')]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.5), layout='constrained')
    for row, h in enumerate((2048, 4096)):
        for ax, p in zip(axs[row], (24, 31, 32)):
            for i, (strategy, route) in enumerate(policies):
                group = [r for r in rows if r['phase'] == 'measurement' and r['H'] == h
                         and r['P'] == p and r['gap_requested_ms'] == 200
                         and r['strategy'] == strategy and r['requested_route'] == route]
                assert len(group) == 3 and all(r['passed'] for r in group)
                values = [r['tool_to_response_ms'] / 1000 for r in group]
                color = '#0072B2' if strategy == 'cost_aware' else '#888888'
                ax.scatter(np.linspace(i - .09, i + .09, 3), values, s=19, color=color, alpha=.55)
                ax.hlines(np.median(values), i - .22, i + .22, color=color, lw=2)
            ax.set_title(f'H={h}, P={p}' + ('\nP31 gain not reproduced; see paired retest' if p == 31 else ''))
            ax.set_xticks(range(5), ['CPU/CPU', 'HTP/CPU', 'Post-tool\nrouting', 'Always\nprepare', 'Cost-aware'])
            ax.margins(x=.08, y=.12)
            if ax is axs[row, 0]:
                ax.set_ylabel('Tool issue to full response (s)')
    fig.supxlabel('Original runs only: n=3 per condition; Tool target 200 ms. Paired retest is a separate experiment.', fontsize=9)
    save(fig, '04_end_to_end_policies')

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 6.6), layout='constrained', width_ratios=[1, 1.4])
    colors = ['#888888', '#0072B2']
    for row, run in enumerate(sorted(summary, key=lambda r: r['H'])):
        measured = [json.loads(line) for line in (Path(run['directory']) / 'checked-rows.jsonl').read_text().splitlines()
                    if json.loads(line)['phase'] == 'measurement']
        assert len(measured) == 40
        ax = axs[row, 0]
        for block in range(20):
            pair = {r['requested_route']: r for r in measured if r['comparison_block'] == block}
            values = [pair[r]['tool_to_response_ms'] / 1000 for r in ('CC', 'router')]
            ax.plot([0, 1], values, color='#CCCCCC', alpha=.65, lw=.6, zorder=1)
            ax.scatter([0, 1], values, c=colors, s=14, alpha=.65, zorder=2)
        for i, req in enumerate(('CC', 'router')):
            ax.hlines(np.median([r['tool_to_response_ms'] / 1000 for r in measured if r['requested_route'] == req]),
                      i - .14, i + .14, colors=colors[i], lw=2, zorder=3)
        ax.set_xticks([0, 1], ['CPU/CPU', 'Post-tool routing'])
        ax.set_ylabel('Tool issue to full response (s)')
        ax.set_title(f'H={run["H"]}, P=31: 20 adjacent pairs')
        ax.margins(x=.22, y=.12)
        ax = axs[row, 1]
        for i, key in enumerate(('tool_to_response_ms', 'response_after_return_ms', 'prefill_ms')):
            stats = run['statistics'][key]
            center = stats['paired_delta_median']
            lo, hi = stats['paired_delta_bootstrap_95_ci']
            ax.errorbar(center, i, xerr=[[center - lo], [hi - center]], fmt='D',
                        color='#0072B2', markersize=5, capsize=3)
        ax.axvline(0, color='#888888', ls='--', lw=.8)
        ax.set_yticks([0, 1, 2], ['Full response', 'After Tool return', 'CPU prefill'])
        ax.invert_yaxis()
        ax.set_xlabel('Paired difference: Post-tool minus CPU (ms)')
        ax.set_title('Median paired difference and 95% bootstrap interval')
        ax.margins(x=.15, y=.3)
    fig.supxlabel('Independent retest: AB/BA balanced; 4 CPU threads; all Post-tool routes = CPU. Negative difference = faster Post-tool.', fontsize=9)
    save(fig, '06_p31_paired_retest')
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        retest_sha256=hashlib.sha256((args.retest / 'summary.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), geometry=qa,
        policy='原04图原样保留762行来源，只增加复测注释；06图使用独立92行复测来源，不混合批次',
        outputs={f'{name}.{suffix}': hashlib.sha256((out / f'{name}.{suffix}').read_bytes()).hexdigest()
                 for name in qa for suffix in ('png', 'pdf')})
    (out / 'p31-update-qa.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
