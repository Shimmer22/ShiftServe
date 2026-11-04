#!/usr/bin/env python3
"""从真实重放结果取反事实 Oracle；不使用 max(G,Tprep) 等离线重叠公式。"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def numeric_accepted(row, limits):
    """按本轮 manifest 判定；null NRMSE 表示只诊断，不是零误差。"""
    return (row['boundary_min_cosine'] >= limits['min_cosine']
            and (limits['max_nrmse'] is None or row['boundary_max_nrmse'] <= limits['max_nrmse']))


def summarize(directory):
    rows = [json.loads(line) for line in (directory / 'checked-rows.jsonl').read_text().splitlines()]
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = manifest['config']
    expected = 6 * len(cfg['new_tokens']) + len(cfg['new_tokens']) * len(cfg['gap_ms']) * len(cfg['strategies']) * (len(cfg['routes']) + 1) * cfg['repetitions']
    if len(rows) != expected:
        raise ValueError('不完整或存在失败条件，拒绝选择成功子集生成 Oracle')
    limits = cfg.get('numeric_acceptance', {'min_cosine': 0.999, 'max_nrmse': 0.05})
    qualified = {}
    for p in cfg['new_tokens']:
        calibration = [r for r in rows if r['phase'] == 'calibration' and r.get('P') == p]
        if sorted((r['route'], r['repetition']) for r in calibration) != [(route, rep) for route in ('CC', 'HC') for rep in range(3)]:
            raise ValueError('独立校准重复组不完整')
        if any(bool(r['numerical_passed']) != numeric_accepted(r, limits) for r in calibration):
            raise ValueError('native 数值准入标记与 manifest 阈值不一致')
        if not all(r['numerical_passed'] for r in calibration if r['route'] == 'CC'):
            raise ValueError('CPU 参考自身数值不一致')
        cpu = [r for r in calibration if r['route'] == 'CC']
        if not all(r['passed'] and r['response_check']['passed'] for r in cpu):
            raise ValueError('CPU 参考输出不合格')
        qualified[p] = all(r['numerical_passed'] and r['passed'] and r['response_check']['passed'] for r in calibration)
    accepted = [r for r in rows if r['route'] == 'CC' or qualified[r['P']]]
    if not all(r['passed'] and r['response_check']['passed'] for r in accepted):
        raise ValueError('准入路径响应/状态失败')
    if len({(r['output'], tuple(r['generated_ids']), r['D']) for r in accepted}) != 1:
        raise ValueError('准入路径 D 或完整输出未固定')
    diagnostic_failures = sum(not r['passed'] or not r['response_check']['passed'] for r in rows)
    measurements = [r for r in rows if r['phase'] == 'measurement']
    keys = [(r['P'], r['gap_requested_ms'], r['strategy'], r['requested_route'], r['repetition']) for r in measurements]
    if len(set(keys)) != len(keys):
        raise ValueError('重复条件')
    lifecycle_checked = False
    if any('response_compute_completed_ms' in r for r in rows):
        events = [json.loads(line) for line in (directory / 'native-events.jsonl').read_text().splitlines()]
        published = {e['turn']: e for e in events if e['event'] == 'response_published'}
        completed = {e['turn']: e for e in events if e['event'] == 'trial_completed'}
        for r in rows:
            pub, end = published[r['id']], completed[r['id']]
            if not (r['http_issued_ms'] <= r['http_returned_ms'] <= r['response_compute_completed_ms']
                    <= pub['monotonic_ms'] <= r['response_completed_ms'] <= end['monotonic_ms']):
                raise ValueError('响应发布/回收时间顺序不成立')
            if pub['output'] != r['output'] or (not pub['eog'] and (r['route'] == 'CC' or qualified[r['P']])):
                raise ValueError('实际发布的完整输出不一致')
        lifecycle_checked = True
    med = statistics.median
    def group(p, g, s, route):
        result = [r for r in measurements if (r['P'], r['gap_requested_ms'], r['strategy'], r['requested_route']) == (p, g, s, route)]
        if sorted(r['repetition'] for r in result) != list(range(cfg['repetitions'])):
            raise ValueError('缺少完整重复组')
        return result
    conditions = []
    for p in cfg['new_tokens']:
        for g in cfg['gap_ms']:
            policies = []
            for s in cfg['strategies']:
                cc, hc, router = (group(p, g, s, r) for r in ('CC', 'HC', 'router'))
                # 严格 Oracle 使用同重复 Router 返回时 availability，且 HC 重放也必须 ready。
                selected = []
                for actual in sorted(router, key=lambda r: r['repetition']):
                    rep = actual['repetition']; c = next(r for r in cc if r['repetition'] == rep); h = next(r for r in hc if r['repetition'] == rep)
                    # 已选路径必须使用这轮的实际结果；Oracle 不能比自身候选更慢。
                    candidates = [actual]
                    available_hc = qualified[p] and actual['ready_at_return']
                    complete = True
                    if actual['route'] == 'HC':
                        if c['ready_at_return'] == actual['ready_at_return'] and c.get('prepared_early') == actual.get('prepared_early'):
                            candidates.append(c)
                        else:
                            complete = False
                    elif available_hc:
                        if h['ready_at_return'] and h.get('prepared_early') == actual.get('prepared_early'):
                            candidates.append(h)
                        else:
                            complete = False
                    best = min(candidates, key=lambda r: r['response_after_return_ms'])
                    selected.append({'repetition': rep, 'route': best['route'], 'measured_trial_id': best['id'],
                        'response_after_return_ms': best['response_after_return_ms'], 'tool_to_response_ms': best['tool_to_response_ms'],
                        'HC_admissible': bool(available_hc), 'counterfactual_state_matched': complete,
                        'source_is_actual_router': best['id'] == actual['id']})
                comparisons_complete = all(r['counterfactual_state_matched'] for r in selected)
                policies.append({'strategy': s, 'router_routes': [r['route'] for r in router],
                    'router_ready_count': sum(r['ready_at_return'] for r in router),
                    'router_after_return_ms': med(r['response_after_return_ms'] for r in router),
                    'router_tool_to_response_ms': med(r['tool_to_response_ms'] for r in router),
                    'routing_oracle_comparisons_complete': comparisons_complete,
                    'routing_oracle_after_return_ms': med(r['response_after_return_ms'] for r in selected) if comparisons_complete else None,
                    'routing_oracle_tool_to_response_ms': med(r['tool_to_response_ms'] for r in selected) if comparisons_complete else None,
                    'routing_oracle_replays': selected,
                    'CC_after_return_ms': med(r['response_after_return_ms'] for r in cc),
                    'HC_after_return_ms': med(r['response_after_return_ms'] for r in hc),
                    'HC_ready_count': sum(r['ready_at_return'] for r in hc),
                    'prepare_ms': med(r['prepare_ms'] for r in hc),
                    'actual_gap_ms': med(r['actual_gap_ms'] for r in router),
                    'CC_drain_ms': med(r['drain_after_response_ms'] for r in cc)})
            # 联合 Oracle 在独立三次重放的聚合成本中选择准备/路径，不逐 token 或用离线阶段相加模拟。
            candidates = []
            for s in ('after_return', 'always'):
                for route in ('CC', 'HC'):
                    sample = group(p, g, s, route)
                    candidates.append({'strategy': s, 'route': route,
                        'qualified': route == 'CC' or qualified[p],
                        'strict_available': route == 'CC' or (qualified[p] and all(r['ready_at_return'] for r in sample)),
                        'tool_to_response_ms': med(r['tool_to_response_ms'] for r in sample),
                        'after_return_ms': med(r['response_after_return_ms'] for r in sample),
                        'trial_ids': [r['id'] for r in sample]})
            joint = min((x for x in candidates if x['strict_available']), key=lambda r: r['tool_to_response_ms'])
            extended = min((x for x in candidates if x['qualified']), key=lambda r: r['tool_to_response_ms'])
            current = next(x for x in policies if x['strategy'] == 'cost_aware')
            conditions.append({'P': p, 'G_ms': g, 'D': rows[0]['D'], 'policies': policies,
                'inference_status': 'two_qualified_paths' if qualified[p] else 'HC_not_qualified_single_path_only', 'joint_candidates': candidates,
                'joint_oracle': joint, 'joint_oracle_allow_wait': extended,
                'router_vs_routing_oracle_ms': current['router_after_return_ms'] - current['routing_oracle_after_return_ms'] if qualified[p] and current['routing_oracle_comparisons_complete'] else None,
                'router_vs_joint_oracle_ms': current['router_tool_to_response_ms'] - joint['tool_to_response_ms'] if qualified[p] else None,
                'routing_vs_joint_oracle_ms': current['routing_oracle_tool_to_response_ms'] - joint['tool_to_response_ms'] if qualified[p] and current['routing_oracle_comparisons_complete'] else None})
    summary = {'passed': True, 'trials': len(rows), 'D_including_eog': rows[0]['D'], 'output': rows[0]['output'],
        'H': cfg.get('history_tokens', rows[0].get('H')), 'numeric_acceptance': cfg.get('numeric_acceptance', {'min_cosine': 0.999, 'max_nrmse': 0.05}),
        'conditions': conditions, 'HC_numerically_qualified': qualified, 'response_publication_lifecycle_checked': lifecycle_checked,
        'response_and_state_passed': diagnostic_failures == 0, 'accepted_response_and_state_passed': True, 'diagnostic_failures': diagnostic_failures, 'oracle_gain_identifiable': all(qualified.values()),
        'HC_calibration_metrics': {p: {'samples': 3, 'qualified': qualified[p],
            'min_boundary_cosine': min(r['boundary_min_cosine'] for r in rows if r['phase'] == 'calibration' and r['P'] == p and r['route'] == 'HC'),
            'max_boundary_nrmse': max(r['boundary_max_nrmse'] for r in rows if r['phase'] == 'calibration' and r['P'] == p and r['route'] == 'HC')}
            for p in cfg['new_tokens']},
        'analysis_script_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['三次串行反事实重放，不能获得同一轮未选路径的真实耗时；最小值存在选择偏差，差值可为负',
            'HC 未准入时只有 CC，选路差距与联合收益记为 null；不能把退化的单路径 Oracle 解释为 Router 已达最优',
            'HC 仅在同一 P 的三次完整校准全部过 manifest 中的数值门槛后准入；失败组原样保留，强制 HC 测量仍可作诊断',
            '严格选路 Oracle 总包含实际已选路径；反事实重放的准备/ready 状态不匹配则该条件差距记 null，不将其伪装成可比较 Oracle；联合 Oracle 要求 HC 三次均 ready；另列允许等待的扩展值',
            '联合 Oracle 选用三次中位数，不将不同阶段的最小值拼接；没有离线 overlap 模拟',
            'G 是服务器延迟目标；实际 Gap 含 adb reverse 网络、线程调度及 HTTP 开销',
            '固定文本微基准；无图像、无真实 GUI 操作、无能耗结论；每轮回收等待在响应计时之外并单独记录',
            '返回后准备策略的 Router 遵循现有不等待 pending 的规则，因此选 CC；HC 强制重放单独测按需准备成本',
            '共同 snapshot 构建/导出和 CPU 恢复在 Tool 前；提前准备包括 clear/import/字节核验，未压缩 KV',
            '边界 logits 校验仅在独立校准执行；测量阶段检查真实 HTTP P、完整生成 ID、EOG 与状态迁移字节']}
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (directory / 'summary.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['P', 'G_ms', 'strategy', 'router_routes', 'ready_count', 'actual_gap_ms', 'router_response_ms', 'routing_oracle_response_ms', 'joint_oracle_total_ms', 'joint_strategy', 'joint_route'])
        for c in conditions:
            for s in c['policies']:
                writer.writerow([c['P'], c['G_ms'], s['strategy'], '/'.join(s['router_routes']), s['router_ready_count'], s['actual_gap_ms'], s['router_after_return_ms'], s['routing_oracle_after_return_ms'], c['joint_oracle']['tool_to_response_ms'], c['joint_oracle']['strategy'], c['joint_oracle']['route']])
    lines = ['# 真实 HTTP Gap Oracle 首批实测', '', f'完整输出 `{rows[0]["output"]}`；D={rows[0]["D"]}（含 EOG 评估）。H={cfg.get("history_tokens", rows[0].get("H"))}，三次重复。共 {len(rows)} 条记录，准入路径响应/状态检查通过，诊断失败 {diagnostic_failures} 条；HC 准入：{qualified}。', '',
        ('**核心结果：HC 未通过既定数值门槛。本批只能得到单路径 CC 的退化 Oracle，无法据此判断 Router 的选路差距或联合策略的有效收益。**' if not all(qualified.values()) else '两个 P 的 CC/HC 均通过数值准入。'), '',
        '时间单位 ms；Router 列为成本感知策略返回后到完整响应的中位数；选路 Oracle 同该准备策略。联合 Oracle 列为从 Tool 发起到完整响应的中位数。', '',
        '| P | G 目标 | 实际 G | Router 路径 | ready/3 | Router 返回后 | 选路 Oracle 返回后 | 联合 Oracle 总时长 | 联合策略/路径 |',
        '|---|---|---|---|---|---|---|---|---|']
    def ms(value):
        return '状态未匹配' if value is None else f'{value:.1f}'
    for c in conditions:
        s = next(x for x in c['policies'] if x['strategy'] == 'cost_aware'); j = c['joint_oracle']
        lines.append(f'| {c["P"]} | {c["G_ms"]} | {s["actual_gap_ms"]:.1f} | {"/".join(s["router_routes"])} | {s["router_ready_count"]} | {s["router_after_return_ms"]:.1f} | {ms(s["routing_oracle_after_return_ms"])} | {j["tool_to_response_ms"]:.1f} | {j["strategy"]}/{j["route"]} |')
    if not all(qualified.values()):
        lines += ['', '**HC 数值未准入的 P 只有 CC 候选，因此选路差距与联合收益无法辨识（JSON 中为 null）；表中的单路径时间不构成最优性或收益证据。**']
    lines += ['', '## 强制路径诊断（不合格 HC 不进入 Oracle）', '',
        '| P | G 目标 | 策略 | CC 返回后 ms | HC 返回后 ms | HC ready/3 | HC 准入 |', '|---|---|---|---|---|---|---|']
    for c in conditions:
        for s in c['policies']:
            lines.append(f'| {c["P"]} | {c["G_ms"]} | {s["strategy"]} | {s["CC_after_return_ms"]:.1f} | {s["HC_after_return_ms"]:.1f} | {s["HC_ready_count"]} | {qualified[c["P"]]} |')
    lines += ['', '## 解释边界', ''] + ['- ' + x for x in summary['limitations']]
    lines += ['', '原始数据：`checked-rows.jsonl`、`device/trials.jsonl`、`http-server.jsonl`、`protocol.log`、`runtime.log`。`summary.csv` 列出三个准备策略，`summary.json` 保留每次严格候选及扩展允许等待 Oracle。', '']
    (directory / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'run': str(directory), 'passed': True, 'trials': len(rows), 'D': rows[0]['D']}, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
