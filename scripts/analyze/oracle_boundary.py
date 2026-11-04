#!/usr/bin/env python3
"""汇总 P/H 交界扫描：仅比较实际 HTTP 重放，不模拟 Gap 重叠。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import median
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.validation.oracle import verify_oracle_run


def summarize(directories, output):
    data = []; provenance = []; outputs = set(); constants = None
    for directory in directories:
        directory = Path(directory)
        audit = verify_oracle_run(directory)
        manifest = json.loads((directory / 'manifest.json').read_text()); cfg = manifest['config']
        rows = [json.loads(s) for s in (directory / 'checked-rows.jsonl').read_text().splitlines()]
        summary = json.loads((directory / 'summary.json').read_text())
        comparable = {k: cfg[k] for k in ('mode', 'context_size', 'batch_size', 'htp_queue', 'htp_opbatch', 'kv_policy', 'pre_move', 'cpu_repack', 'input_fixture', 'numeric_acceptance')}
        comparable.update(htp_profile=cfg.get('htp_profile', 0), htp_use_hmx=cfg.get('htp_use_hmx', 'default'))
        if constants is not None and constants != comparable:
            raise ValueError('比较 H/P 时其他配置发生变化')
        constants = comparable
        provenance.append({'directory': str(directory.resolve()), 'config': cfg, 'audit': audit})
        qualified = summary['HC_numerically_qualified']
        valid = [r for r in rows if r['route'] == 'CC' or qualified[str(r['P'])]]
        outputs.update((r['output'], tuple(r['generated_ids']), r['D']) for r in valid)
        measurements = [r for r in rows if r['phase'] == 'measurement']
        for p in cfg['new_tokens']:
            for g in cfg['gap_ms']:
                def group(strategy, route):
                    return [r for r in measurements if (r['P'], r['gap_requested_ms'], r['strategy'], r['requested_route']) == (p, g, strategy, route)]
                cc = group('after_return', 'CC'); late = group('after_return', 'HC'); early = group('always', 'HC'); router = group('cost_aware', 'router')
                future = lambda sample: median(r['response_after_return_ms'] for r in sample)
                q = qualified[str(p)]
                cc_ms, late_ms, early_ms = map(future, (cc, late, early))
                prep = median(r['prepare_ms'] for r in early)
                data.append({'H': cfg['history_tokens'], 'P': p, 'G_target_ms': g, 'HC_qualified': q,
                    'CC_after_return_ms': cc_ms, 'late_HC_after_return_ms': late_ms, 'early_HC_after_return_ms': early_ms,
                    'HC_ready_count': sum(r['ready_at_return'] for r in early), 'repetitions': cfg['repetitions'],
                    'prepare_ms': prep, 'handoff_ms': median(r['handoff_checked_ms'] for r in early),
                    'history_bytes': early[0]['state_bytes'], 'handoff_bytes': early[0]['handoff_bytes'],
                    'early_vs_late_total_gain_ms': median(r['tool_to_response_ms'] for r in late) - median(r['tool_to_response_ms'] for r in early) if q else None,
                    'router_routes': [r['route'] for r in router],
                    'ready_path_choice': ('HC' if early_ms < cc_ms else 'CC') if q else 'HC_unavailable',
                    'late_path_choice': ('HC' if late_ms < cc_ms else 'CC') if q else 'HC_unavailable',
                    'possible_preparation_flip': q and late_ms > cc_ms and early_ms < cc_ms,
                    'near_boundary': q and min(abs(early_ms-cc_ms), abs(late_ms-cc_ms)) < max(100, prep),
                    'source': str(directory.resolve())})
    if len(outputs) != 1:
        raise ValueError('不同 H/P 的完整输出或 D 不一致')
    # 同一 H/P/G 若另有细扫，按命令参数顺序用后提供的完整运行；不按耗时择优。
    all_data = data
    latest = {(r['H'], r['P'], r['G_target_ms']): r for r in data}
    data = sorted(latest.values(), key=lambda r: (r['H'], r['P'], r['G_target_ms']))
    output.mkdir(parents=True, exist_ok=True)
    result = {'config_constants': constants, 'runs': provenance, 'conditions': data, 'all_run_conditions': all_data, 'D': next(iter(outputs))[2],
        'scope': '小样本探索；翻转为三次实测中位数的候选现象，不冒称统计显著或模拟收益'}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# P/H 与真实 Tool Gap 交界扫描', '', f'固定完整输出，D={result["D"]}（含 EOG）。cosine≥0.99，NRMSE 仅记录；每条件三次。其余模型/上下文/后端配置相同。', '',
        '单位 ms；时间为实际 HTTP 返回后到完整响应的中位数。HC 未准入的耗时仅为诊断。', '',
        '| H | P | G 目标 | CC | 返回后准备 HC | 提前准备 HC | HC ready | HC 准入 | 成本感知实际路径 |',
        '|---|---|---|---|---|---|---|---|---|']
    for r in data:
        lines.append(f'| {r["H"]} | {r["P"]} | {r["G_target_ms"]} | {r["CC_after_return_ms"]:.1f} | {r["late_HC_after_return_ms"]:.1f} | {r["early_HC_after_return_ms"]:.1f} | {r["HC_ready_count"]}/{r["repetitions"]} | {r["HC_qualified"]} | {"/".join(r["router_routes"])} |')
    lines += ['', '仅从实测中位数筛查的准备翻转候选：']
    flips = [r for r in data if r['possible_preparation_flip']]
    lines += [f'- H={r["H"]}, P={r["P"]}, G目标={r["G_target_ms"]}：返回后准备选 CC，提前准备选 HC；需更多重复确认。' for r in flips] or ['- 本批未观察到。']
    lines += ['', '历史 KV 与返回后交接均保留字节数和耗时；改变 H 同时改变注意力计算，不能把总延迟变化全部归因于搬移。实际 Gap 包含 HTTP/adb reverse/调度开销。原始响应、失败条件及数值诊断不改写。', '', '同一 H/P/G 重复运行时，主表采用命令中后提供的完整细扫；所有粗扫与细扫仍保留在 JSON 中，不按性能择优。', '', '来源：']
    lines += ['- ' + p['directory'] for p in provenance]
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(output / 'README.md'), 'conditions': len(data), 'possible_flips': len(flips)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directories', type=Path, nargs='+'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'results' / (datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f') + '-oracle-boundary-comparison')
    summarize(args.directories, output)
