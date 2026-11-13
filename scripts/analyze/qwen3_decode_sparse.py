#!/usr/bin/env python3
"""Qwen3稀疏decode同轮对照，仅判断速度，不推断草稿接受率。"""
import json
from pathlib import Path
from decode_attribution import read_case

ROOT = Path(__file__).resolve().parents[2]


def main():
    plan = json.loads((ROOT / 'configs/qwen3-decode-sparse.json').read_text())
    out = ROOT / 'results' / plan['name']
    assert json.loads((out / 'completion.json').read_text())['completed']
    cases = {}
    for case in plan['cases']:
        d, stats, _ = read_case(out / case['name'])
        assert d['actual_H'] == 4096 and d['actual_P'] == 40
        assert d['warmup_elapsed_ms'] >= 40000
        assert d['checkpoint_replay']['final_state_bitwise_equal']
        if case['keep_ratio'] == 1:
            assert d['json_memory']['passed']
        formal = [r for r in d['rows'] if not r['warmup']]
        stats.update(packing_ms=d['ffn_compaction']['packing_ms'],
                     major_fault_deltas=[r['resources_after']['major_faults'] - r['resources_before']['major_faults'] for r in formal],
                     resources_after=[r['resources_after'] for r in formal],
                     kept_ffn=sorted({v['kept_ffn'] for v in d['ffn_compaction']['layers']}))
        cases[case['name']] = stats
    cpu = [cases[n]['ms_per_token'] for n in ['cpu-before', 'cpu-after']]
    comparisons = {}
    for name in ['htp-keep50', 'htp-keep25', 'htp-no-ffn']:
        ms = cases[name]['ms_per_token']
        comparisons[name] = dict(cpu_speedup_range=[v / ms for v in sorted(cpu)], faster_than_both_cpu_medians=ms < min(cpu))
    summary = dict(model=plan['model_id'], cases=cases, comparisons=comparisons,
                   scope='Qwen3-8B Q4_0；H4096/P40、context4608；CPU6/HTP4；KV offload关闭/FA关闭、OPBATCH1024/OPQUEUE16；每条件独立进程、完整prefill后固定FFN通道裁剪；初始化/历史建立/打包/恢复单列；不评价重要性、接受率或端到端投机')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Qwen3稀疏decode对照', '', summary['scope'], '', '| 条件 | 中位数ms/token | 正式重复ms/token | 打包ms |', '|---|---:|---|---:|']
    for name, s in cases.items():
        repeats = ' / '.join(f'{v:.2f}' for v in s['repetitions_ms_per_token'])
        lines.append(f"| {name} | {s['ms_per_token']:.2f} | {repeats} | {s['packing_ms']:.2f} |")
    lines += ['', '每组40秒以上纯decode预热，3×16固定token。完整模型另验JSON/两轮记忆；所有条件验正常响应（裁剪前）、有限logits及检查点续跑逐字节一致。移除全部FFN仅为诊断下界，不能视作可用草稿模型。', '', 'CPU和完整HTP均有前后对照，资源及原始重复见summary.json。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
