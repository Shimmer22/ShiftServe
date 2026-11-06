"""有限等待配对验收与触发率，分开统计校准、强制路径及真实Router决策。"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import median

from .termux_campaign import audit_raw


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def trigger_rates(rows):
    eligible = [r for r in rows if r['phase'] == 'measurement' and r['requested_route'] == 'router']
    def count(group):
        n = len(group)
        prepared = sum(r['prepared_early'] for r in group)
        pending = sum(r['prepared_early'] and not r['ready_at_return'] for r in group)
        attempted = sum(r.get('pending_wait_attempted', False) for r in group)
        success = sum(r.get('pending_wait_succeeded', False) for r in group)
        timeout = sum(r.get('pending_wait_timed_out', False) for r in group)
        return dict(router_decisions=n, prepare=fraction(prepared, n),
            select_htp=fraction(sum(r['route'] == 'HC' for r in group), n),
            prepared_but_cpu=fraction(sum(r['prepared_early'] and r['route'] == 'CC' for r in group), n),
            pending_at_return=fraction(pending, n), wait_attempt=fraction(attempted, n),
            wait_given_pending=fraction(attempted, pending),
            wait_success=fraction(success, attempted), wait_timeout=fraction(timeout, attempted))
    by_strategy = defaultdict(list)
    by_condition = defaultdict(list)
    for r in eligible:
        policy = r.get('wait_policy', 'ready_only')
        by_strategy[(r['strategy'], policy)].append(r)
        by_condition[(r['H'], r['P'], r['gap_requested_ms'], r['strategy'], policy)].append(r)
    return dict(denominator='measurement rows with requested_route=router; excludes calibration and forced CC/HC',
        by_strategy=[dict(strategy=s, wait_policy=p, **count(g)) for (s, p), g in sorted(by_strategy.items())],
        by_condition=[dict(H=h, P=p, gap_target_ms=g, strategy=s, wait_policy=w, **count(rs))
                      for (h, p, g, s, w), rs in sorted(by_condition.items())])


def bootstrap_interval(values):
    rng = random.Random(20261106)
    sample = sorted(median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [sample[249], sample[9749]]


def analyze_pending_run(run):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    cfg = manifest['config']
    count = 6 * len(cfg['new_tokens']) + 2 * len(cfg['pending_wait_cases']) * cfg['repetitions']
    rows = audit_raw(run, count)
    cal = [r for r in rows if r['phase'] == 'calibration']
    measured = [r for r in rows if r['phase'] == 'measurement']
    groups = defaultdict(list)
    for r in measured:
        groups[(r['P'], r['gap_requested_ms'], r['strategy'], r['comparison_block'])].append(r)
        if r['pending_wait_succeeded'] and (r['route'] != 'HC' or not r['pending_wait_attempted']):
            raise ValueError('等待成功与实际选路不一致')
        if r['pending_wait_timed_out'] and (r['route'] != 'CC' or not r['pending_wait_attempted']):
            raise ValueError('等待超时后未选CPU')
        if r['pending_wait_attempted']:
            if r['ready_at_return'] or not r['prepared_early'] or r['wait_policy'] != 'bounded':
                raise ValueError('非pending候选发生等待')
            if not 0 < r['pending_wait_budget_ms'] <= cfg['pending_wait_cap_ms']:
                raise ValueError('等待预算超出显式上限')
        if r['wait_policy'] == 'bounded':
            p = r['P']
            upper = max(c['prepare_ms'] for c in cal if c['P'] == p and c['route'] == 'HC') * cfg['pending_wait_prepare_factor']
            costs = [median(c['prefill_ms'] + c['handoff_checked_ms'] + c['decode_ms']
                       for c in cal if c['P'] == p and c['route'] == route) for route in ('CC', 'HC')]
            if abs(upper - r['calibrated_prepare_upper_ms']) > .001 or any(
                    abs(c - r[field]) > .001 for c, field in zip(costs, ['calibrated_cc_ms', 'calibrated_hc_ms'])):
                raise ValueError('等待成本不来自事前独立校准')
            allowed = costs[1] + upper + cfg['pending_wait_margin_ms'] < costs[0]
            expected_wait = allowed and r['prepared_early'] and not r['ready_at_return']
            if bool(r['pending_wait_attempted']) != expected_wait:
                raise ValueError('经济准入与实际等待不一致')
    pairs = []
    expected_groups = {(p, g, s, b) for p, g, s in cfg['pending_wait_cases'] for b in range(cfg['repetitions'])}
    if set(groups) != expected_groups:
        raise ValueError('配对条件不完整')
    for (p, gap, strategy, block), group in groups.items():
        if len(group) != 2 or group[1]['id'] != group[0]['id'] + 1 or [r['comparison_order'] for r in group] != [0, 1]:
            raise ValueError('不是相邻配对')
        lookup = {r['wait_policy']: r for r in group}
        if set(lookup) != {'ready_only', 'bounded'}:
            raise ValueError('配对策略缺失')
        pairs.append(dict(P=p, gap_target_ms=gap, strategy=strategy, block=block,
            order='/'.join(r['wait_policy'] for r in group),
            ready_trial=lookup['ready_only']['id'], bounded_trial=lookup['bounded']['id'],
            **{k: lookup['bounded'][k] - lookup['ready_only'][k] for k in
               ['tool_to_response_ms', 'response_after_return_ms', 'actual_gap_ms', 'prefill_ms']}))
    stats = []
    for p, gap, strategy in cfg['pending_wait_cases']:
        pp = [x for x in pairs if (x['P'], x['gap_target_ms'], x['strategy']) == (p, gap, strategy)]
        orders = Counter(x['order'] for x in pp)
        if set(orders.values()) != {cfg['repetitions'] // 2}:
            raise ValueError('AB/BA未平衡')
        rr = [r for r in measured if (r['P'], r['gap_requested_ms'], r['strategy']) == (p, gap, strategy)]
        fields = {}
        for key in ['tool_to_response_ms', 'response_after_return_ms', 'actual_gap_ms', 'prefill_ms']:
            delta = [x[key] for x in pp]
            fields[key] = dict(ready_median=median(r[key] for r in rr if r['wait_policy'] == 'ready_only'),
                bounded_median=median(r[key] for r in rr if r['wait_policy'] == 'bounded'),
                paired_delta_median=median(delta), bootstrap_95_ci=bootstrap_interval(delta),
                bounded_faster_pairs=sum(x < 0 for x in delta))
        stats.append(dict(P=p, gap_target_ms=gap, strategy=strategy, pairs=len(pp), orders=dict(orders), statistics=fields))
    attempted = [r for r in measured if r['pending_wait_attempted']]
    return dict(passed=True, run=str(run.resolve()), H=cfg['history_tokens'], trials=count,
        calibration_trials=len(cal), measurement_trials=len(measured), pairs=len(pairs),
        trigger_rates=trigger_rates(rows), comparisons=stats, pair_details=pairs,
        max_wait_ms=max((r['pending_wait_ms'] for r in attempted), default=None),
        max_budget_overshoot_ms=max((max(0, r['pending_wait_ms'] - r['pending_wait_budget_ms']) for r in attempted), default=None),
        max_suspend_ms=max(r['suspend_during_tool_to_response_ms'] for r in rows),
        limitations='单H、单会话相邻配对；bootstrap不代表跨会话稳定性。100ms为wait_for请求预算，操作系统调度可超时；全部慢点保留。')
