#!/usr/bin/env python3
"""按用户认可的探索性口径接纳 HC；保留旧数值门槛及原始实验，不重写在线决策。"""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(directory):
    raw = [json.loads(line) for line in (directory / 'checked-rows.jsonl').read_text().splitlines()]
    if len(raw) != 228 or not all(r['passed'] and r['response_check']['passed'] for r in raw):
        raise ValueError('要求完整矩阵及全部响应/状态通过')
    if len({(r['output'], tuple(r['generated_ids']), r['D']) for r in raw}) != 1:
        raise ValueError('完整输出和 D 不一致')
    rows = [r for r in raw if r['phase'] == 'measurement']
    conditions = []
    for p in (16, 64):
        for gap in (0, 50, 200, 500):
            paths = []
            for strategy in ('after_return', 'always'):
                for route in ('CC', 'HC'):
                    sample = [r for r in rows if (r['P'], r['gap_requested_ms'], r['strategy'], r['requested_route']) == (p, gap, strategy, route)]
                    if sorted(r['repetition'] for r in sample) != [0, 1, 2]:
                        raise ValueError('不得选择成功子集')
                    paths.append({'strategy': strategy, 'route': route, 'trial_ids': [r['id'] for r in sample],
                        'ready_count': sum(r['ready_at_return'] for r in sample),
                        'after_return_ms': median(r['response_after_return_ms'] for r in sample),
                        'total_ms': median(r['tool_to_response_ms'] for r in sample),
                        'actual_gap_ms': median(r['actual_gap_ms'] for r in sample)})
            cc, late_hc, early_cc, early_hc = paths
            # 选择完整路径实测中位数；没有按阶段相加或模拟重叠。
            joint = min(paths, key=lambda r: r['total_ms'])
            conditions.append({'P': p, 'G_target_ms': gap, 'paths': paths,
                'joint_oracle_allow_explicit_preparation_or_wait': joint,
                'CC_vs_best_HC_after_return_reduction_pct': 100 * (1 - min(late_hc['after_return_ms'], early_hc['after_return_ms']) / cc['after_return_ms']),
                'early_vs_late_HC_total_gain_ms': late_hc['total_ms'] - early_hc['total_ms'],
                'online_relaxed_router_measured': False})
    result = {'acceptance': '用户于 2026-11-04 认可探索阶段当前 HC 数值可用；响应、完整生成 ID、EOG 与状态核验仍须全部通过',
        'original_numeric_flags_preserved': True, 'D_including_eog': raw[0]['D'], 'conditions': conditions,
        'limitations': ['原在线 Router 因旧门槛禁用 HC，cost_aware 分支未提前准备且全部选 CC；不能把离线改资格当成在线放宽门槛后的实测',
            '此处 Oracle 是完整实测路径重放的经验最小值，允许显式返回后准备或等待；不同于旧 Router 仅接纳 ready 的选路 Oracle',
            '三次重复、无线 HTTP 实际 Gap 波动；几十毫秒差异不足以证明提前准备的稳定收益']}
    (directory / 'exploratory-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 探索性口径重分析', '', '接纳当前 HC 数值作为探索性可用候选；原数值结果与严格报告均保留。H4096、完整相同输出、D=6。', '',
        '**主要发现：P16 选 CC，P64 选 HC；G 未改变最优执行路径。提前准备相对返回后准备的额外收益尚未体现稳定优势。**', '',
        '以下为返回后到完整响应的中位数，单位 ms；G 为服务器延迟目标。', '',
        '| P | G | 不提前准备 CC | 返回后准备 HC | 提前准备 HC | 提前 HC ready/3 |', '|---|---|---|---|---|---|']
    for c in conditions:
        cc, late, _, early = c['paths']
        lines.append(f'| {c["P"]} | {c["G_target_ms"]} | {cc["after_return_ms"]:.1f} | {late["after_return_ms"]:.1f} | {early["after_return_ms"]:.1f} | {early["ready_count"]} |')
    lines += ['', 'P64 提前 HC 相对返回后 HC 的 Tool 发起到完整响应收益（正数更快）：']
    for c in conditions:
        if c['P'] == 64:
            lines.append(f'- G={c["G_target_ms"]}：{c["early_vs_late_HC_total_gain_ms"]:+.1f}ms。')
    lines += ['', '联合 Oracle（允许显式准备/等待）在 P16 全部选择 CC；P64 在 G=0/50/200 选择返回后准备 HC，G=500 选择提前准备 HC。这是三次完整重放中位数的经验最小值，不能把小差异作为确定策略边界。', '',
        '当前在线 Router 的差距仍需要放宽准入后重跑：原 cost_aware 运行因为数值门槛从未接纳 HC，也未提前准备。现有强制路径证明可选路径的潜力，不能替代该策略本身的在线测量。', '',
        '解释边界：'] + ['- ' + x for x in result['limitations']]
    (directory / 'exploratory-report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(directory / 'exploratory-report.md'), 'conditions': len(conditions)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
