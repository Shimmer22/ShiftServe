#!/usr/bin/env python3
"""汇总运行时优化叠加固定FFN裁剪；只判断decode速度，不推断接受率。"""
import json
from pathlib import Path
from decode_attribution import read_case

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/20261113-npu-decode-stacked-sparse'


def main():
    plan = json.loads((ROOT / 'configs/npu-decode-stacked-sparse.json').read_text())
    cases = {}
    for case in plan['cases']:
        d, stats, _ = read_case(OUT / case['name'])
        assert d['warmup_elapsed_ms'] >= 40000
        assert d['checkpoint_replay']['final_state_bitwise_equal']
        stats['packing_ms'] = d['ffn_compaction']['packing_ms']
        stats['kept_ffn'] = sorted({v['kept_ffn'] for v in d['ffn_compaction']['layers']})
        stats['warmup_seconds'] = d['warmup_elapsed_ms'] / 1000
        stats['checkpoint_replay_passed'] = True
        stats['major_fault_deltas'] = [
            r['resources_after']['major_faults'] - r['resources_before']['major_faults']
            for r in d['rows'] if not r['warmup']]
        cases[case['name']] = stats
    cpu = [cases[n]['ms_per_token'] for n in ['cpu-before', 'cpu-after']]
    dense = [cases[n]['ms_per_token'] for n in ['optimized-dense-before', 'optimized-dense-after']]
    comparisons = {}
    for name in ['optimized-keep50', 'optimized-keep25']:
        ms = cases[name]['ms_per_token']
        comparisons[name] = dict(cpu_speedup_range=[v / ms for v in sorted(cpu)],
            dense_speedup_range=[v / ms for v in sorted(dense)],
            faster_than_both_cpu_medians=ms < min(cpu),
            faster_than_all_cpu_repetitions=max(cases[name]['repetitions_ms_per_token']) < min(
                v for n in ['cpu-before', 'cpu-after'] for v in cases[n]['repetitions_ms_per_token']))
    summary = dict(cases=cases, comparisons=comparisons,
        scope='Qwen3.5-9B Q4_0 H4096/P40；完整prefill后固定通道稀疏；不含packing/恢复；未测重要性、接受率与端到端投机')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 运行时优化叠加FFN稀疏', '', summary['scope'], '',
        '| 条件 | ms/token中位数 | 三次正式测量 | 打包ms |', '|---|---:|---|---:|']
    for name, s in cases.items():
        lines.append(f"| {name} | {s['ms_per_token']:.2f} | " + ', '.join(f'{v:.2f}' for v in s['repetitions_ms_per_token']) + f" | {s['packing_ms']:.2f} |")
    lines += ['', '每组至少40秒纯decode预热，然后3×16 token。CPU为6线程；HTP主机4线程，CPU融合AR DeltaNet、OPBATCH1024/OPQUEUE16、图复用开启。每条件独立进程，同一进程完成完整prefill及decode。', '',
        '所有条件正常响应和状态恢复通过，固定token检查点续跑的最终状态逐字节一致。正常响应在裁剪前验证；稀疏部分仅验证有限logits和状态续跑，不代表生成质量。稀疏保留前50%/25%通道，不是重要性选择。', '',
        'CPU与完整HTP各有前后对照；保留全部预热、正式重复与资源观测。差异范围见summary.json，不把不同时间的单次测量拼成端到端加速。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(cases={n: s['ms_per_token'] for n, s in cases.items()}, comparisons=comparisons), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
