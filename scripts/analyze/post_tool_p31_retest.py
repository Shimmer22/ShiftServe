#!/usr/bin/env python3
"""P31相邻AB/BA复测验收和配对统计；不筛掉慢点。"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import median


def paired_interval(values):
    rng = random.Random(20261105)
    samples = sorted(median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [samples[249], samples[9749]]


def analyze(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    completion = json.loads((directory / 'completion.json').read_text())
    rows = [json.loads(line) for line in (directory / 'checked-rows.jsonl').read_text().splitlines()]
    config = manifest['config']
    assert config['paired_post_tool'] and completion['passed']
    assert len(rows) == manifest['expected_trials'] == 6 + 2 * config['repetitions']
    assert all(r['passed'] and r['response_check']['passed'] and r['generated_ids_equal']
               and r['eog'] and r['P'] == 31 for r in rows)
    assert all(r['numerical_passed'] for r in rows if r['phase'] == 'calibration')
    assert all(abs(r['actual_gap_ms'] + r['response_after_return_ms'] - r['tool_to_response_ms']) < .01
               for r in rows)
    measured = [r for r in rows if r['phase'] == 'measurement']
    assert len(measured) == 2 * config['repetitions']
    pairs = []
    for block in range(config['repetitions']):
        group = [r for r in measured if r['comparison_block'] == block]
        assert len(group) == 2 and group[1]['id'] == group[0]['id'] + 1
        assert [r['comparison_order'] for r in group] == [0, 1]
        lookup = {r['requested_route']: r for r in group}
        assert set(lookup) == {'CC', 'router'}
        assert all(r['strategy'] == 'after_return' and not r['prepared_early']
                   and r['gap_requested_ms'] == 200 for r in group)
        pairs.append(dict(H=config['history_tokens'], block=block,
            order='/'.join(r['requested_route'] for r in group),
            cpu_trial=lookup['CC']['id'], post_tool_trial=lookup['router']['id'],
            post_tool_route=lookup['router']['route'],
            **{key: lookup['router'][key] - lookup['CC'][key] for key in
               ('tool_to_response_ms', 'response_after_return_ms', 'actual_gap_ms',
                'prefill_ms', 'decode_ms')}))
    orders = Counter(p['order'] for p in pairs)
    assert set(orders.values()) == {config['repetitions'] // 2}
    stats = {}
    for field in ('tool_to_response_ms', 'response_after_return_ms', 'actual_gap_ms', 'prefill_ms', 'decode_ms'):
        values = [p[field] for p in pairs]
        stats[field] = dict(cpu_median=median(r[field] for r in measured if r['requested_route'] == 'CC'),
            post_tool_median=median(r[field] for r in measured if r['requested_route'] == 'router'),
            paired_delta_median=median(values), paired_delta_bootstrap_95_ci=paired_interval(values),
            post_tool_faster_pairs=sum(v < 0 for v in values),
            delta_by_order={order: median(p[field] for p in pairs if p['order'] == order) for order in orders})
    original = json.loads((directory.parents[0] / ('20261105-165933-954653-http-oracle-grid'
        if config['history_tokens'] == 2048 else '20261105-173045-235354-http-oracle-grid') / 'manifest.json').read_text())
    if manifest.get('transport') == 'ssh' and manifest.get('device_identity_access_limited'):
        assert all(manifest['device_identity'][k] == original['device_identity'][k]
                   for k in original['device_identity'] if k != 'ro.serialno')
    else:
        assert manifest['device_identity'] == original['device_identity']
    assert manifest['device_files_sha256'] == original['device_files_sha256']
    return dict(directory=str(directory), H=config['history_tokens'], seed=config['seed'],
        trials=len(rows), pairs=len(pairs), orders=dict(orders),
        routes=dict(Counter(r['route'] for r in measured if r['requested_route'] == 'router')),
        binary_sha256=manifest['binary_sha256'], original_binary_sha256=original['binary_sha256'],
        changed_native_sources=[k for k, v in manifest['source_files_sha256'].items()
                                if original['source_files_sha256'].get(k) != v],
        statistics=stats, pair_details=pairs,
        raw_sha256=hashlib.sha256((directory / 'checked-rows.jsonl').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path, nargs=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [analyze(p.resolve()) for p in args.runs]
    assert {r['H'] for r in runs} == {2048, 4096}
    assert len({r['seed'] for r in runs}) == 2
    assert len({r['binary_sha256'] for r in runs}) == 1
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / 'summary.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2) + '\n')
    details = [p for r in runs for p in r['pair_details']]
    with (args.output / 'pairs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    no_gain = all(r['routes'] == {'CC': 20} and
        r['statistics']['tool_to_response_ms']['paired_delta_bootstrap_95_ci'][0] <= 0 <=
        r['statistics']['tool_to_response_ms']['paired_delta_bootstrap_95_ci'][1] for r in runs)
    lines = ['# P31 Post-tool配对复测', '',
        ('原图P31的374～442ms优势未在本轮复现；40次Post-tool全部走CPU，'
         '两组配对完整耗时差的95%区间均包含0。本轮不支持P31存在Post-tool专用稳定加速机制。'
         if no_gain else '统计与实际路径如下，需结合配对区间判断。'), '',
        '同一设备/模型/共享库/输入历史/ctx6144/batch128/F16 KV；CPU prefill与decode均4线程。'
        '每个H同一进程，先6次独立CPU/HTP数值校准，再20组CC/Post-tool相邻配对；AB/BA各10组，两个H不同seed。'
        '每次G目标200ms，无profile、KV整理、预搬移或手机设置修改。', '',
        '表中差值均为Post-tool减CPU，负数表示Post-tool更快。区间为20组配对差值中位数的10000次bootstrap百分位区间；'
        '同一进程中的相关性仍可能影响区间，不能把区间当作跨会话稳定性证明。', '',
        '| H | 指标 | CPU中位数ms | Post-tool中位数ms | 配对差中位数ms | 95%区间ms | Post-tool较快组数 |',
        '|---|---|---|---|---|---|---|']
    for run in runs:
        for field, label in [('tool_to_response_ms', '完整响应'), ('response_after_return_ms', '返回后响应'), ('prefill_ms', 'CPU prefill')]:
            s = run['statistics'][field]
            lo, hi = s['paired_delta_bootstrap_95_ci']
            lines.append(f'| {run["H"]} | {label} | {s["cpu_median"]:.1f} | {s["post_tool_median"]:.1f} | '
                         f'{s["paired_delta_median"]:.1f} | [{lo:.1f}, {hi:.1f}] | {s["post_tool_faster_pairs"]}/20 |')
    for run in runs:
        lines += ['', f'H{run["H"]}：Post-tool实际路径{run["routes"]}，seed={run["seed"]}。'
                  f'原始记录：{run["directory"]}。', '']
    lines += ['全部92条记录的JSON、生成ID、EOG和状态验收通过，独立数值校准通过；40个配对完整保留。'
              'HTTP正文与历史由执行器逐字节对照原实验；模型与共享运行时hash、设备身份由本分析器核对原manifest。', '',
              '本次使用当前原生源码构建，两H二进制一致；与原图二进制不同，源码差异见summary.json。'
              '因此策略效应首先依据本轮同二进制配对对照，不能把新旧绝对耗时变化直接归因于策略。', '',
              '本轮没有复刻原13个P与HTP混合前驱序列，也没有逐轮硬件调度/频率观测；'
              '因此能判定原策略优势未复现，尚不能定位原始异常的具体底层原因。执行协议见protocol.md。', '',
              '统计、执行顺序、阶段差值、来源与hash见summary.json和pairs.csv。']
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k not in ('pair_details',)} for r in runs], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
