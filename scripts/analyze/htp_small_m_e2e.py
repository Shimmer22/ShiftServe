#!/usr/bin/env python3
"""汇总固定H4096的完整模型P31+D16，对比全部logits、状态与greedy延续。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def compare(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    identical_bytes = a.tobytes() == b.tobytes()
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    delta = b - a
    return dict(elements=a.size, bitwise_equal=identical_bytes,
                max_abs=float(np.max(np.abs(delta))), nrmse=float(np.linalg.norm(delta) / np.linalg.norm(a)),
                cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                allclose_1e_5=bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)),
                top1_equal=bool(np.argmax(a) == np.argmax(b)), top10_overlap=len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    out = args.input
    assert json.loads((out / 'status.json').read_text())['phase'] == 'complete'
    names = ['baseline-a', 'guarded-a', 'guarded-b', 'baseline-b']
    records = {name: json.loads((out / (name + '.json')).read_text()) for name in names}
    arrays = {(name, stage): np.fromfile(out / (name + '.json.' + stage + '.f32'), dtype='<f4')
              for name in names for stage in ['prefill', 'forced']}
    for record in records.values():
        assert record['complete'] and (record['H'], record['P'], record['D']) == (4096, 31, 16)
        assert record['backend'] == 'HTP0' and record['htp_host_threads'] == 4 and len(record['trials']) == 10
        assert record['validation']['state_roundtrip_verified'] and record['validation']['continuation_bitwise_verified']
        assert all(t['final_logits_bitwise_verified'] for t in record['trials'])
        for trial in record['trials']:
            assert all(math.isfinite(trial[k]) and trial[k] > 0 for k in ['restore_work_ms', 'prefill_ms', 'decode_ms'])
            assert abs(trial['prefill_decode_ms']-trial['prefill_ms']-trial['decode_ms']) < 1e-5
            assert abs(trial['total_ms']-trial['restore_work_ms']-trial['prefill_decode_ms']) < 1e-5
    assert len({r['seed_checksum'] for r in records.values()}) == 1
    metrics = ['restore_work_ms', 'prefill_ms', 'decode_ms', 'prefill_decode_ms', 'total_ms']
    timing = {}
    for condition in ['baseline', 'guarded']:
        trials = [t for name, record in records.items() if name.startswith(condition) for t in record['trials']]
        timing[condition] = {metric: dict(mean=statistics.mean(t[metric] for t in trials),
            median=statistics.median(t[metric] for t in trials), min=min(t[metric] for t in trials), max=max(t[metric] for t in trials)) for metric in metrics}
    reduction = {metric: 1-timing['guarded'][metric]['mean']/timing['baseline'][metric]['mean'] for metric in metrics}
    blocks = {suffix: {metric: 1-statistics.mean(t[metric] for t in records['guarded-'+suffix]['trials']) /
        statistics.mean(t[metric] for t in records['baseline-'+suffix]['trials']) for metric in metrics} for suffix in ['a', 'b']}
    numerical = {stage: compare(arrays[('baseline-a', stage)], arrays[('guarded-a', stage)]) for stage in ['prefill', 'forced']}
    repeatability = {condition: {stage: compare(arrays[(condition+'-a', stage)], arrays[(condition+'-b', stage)]) for stage in ['prefill', 'forced']}
                     for condition in ['baseline', 'guarded']}
    states = {stage: dict(baseline_a=records['baseline-a']['validation'][stage+'_state_checksum'],
        guarded_a=records['guarded-a']['validation'][stage+'_state_checksum'],
        cross_variant_equal=records['baseline-a']['validation'][stage+'_state_checksum']==records['guarded-a']['validation'][stage+'_state_checksum'],
        within_variant_repeat_equal=all(records[c+'-a']['validation'][stage+'_state_checksum']==records[c+'-b']['validation'][stage+'_state_checksum'] for c in ['baseline','guarded']))
        for stage in ['prefill', 'forced']}
    greedy = {name: {k: record['validation'][k] for k in ['greedy_ids', 'greedy_text', 'greedy_eog']} for name, record in records.items()}
    greedy_equal = all(row['greedy_ids'] == greedy['baseline-a']['greedy_ids'] for row in greedy.values())
    base_ids = greedy['baseline-a']['greedy_ids']
    candidate_ids = greedy['guarded-a']['greedy_ids']
    first_difference = next((i+1 for i, (a, b) in enumerate(zip(base_ids, candidate_ids)) if a != b),
                            min(len(base_ids), len(candidate_ids))+1 if len(base_ids) != len(candidate_ids) else None)
    summary = dict(H=4096, P=31, D=16, samples_per_variant=20, timing_ms=timing,
        time_reduction=reduction, block_time_reduction=blocks, numerical=numerical, repeatability=repeatability,
        state_checks=states, all_roundtrip_continuations_bitwise_passed=True, greedy_equal=greedy_equal,
        greedy_first_difference_1based=first_difference, greedy=greedy,
        logits_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.f32'))},
        scope='完整模型固定历史后的prefill与HTP forced decode；不包含模型初始化、工具、网络、视觉与验证开销；非任务质量验收')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 设备内补齐：完整模型P31+D16验证', '',
             '同一Qwen3.5-9B Q4_0、H4096权威CPU快照、P31真实输入、D16固定后续token。原版/guarded使用相同主机可执行文件与运行库，唯一干预是私有DSP。HTP主机4线程、batch128、F16状态、无KV整理/预搬移。', '',
             '| 阶段 | 原版均值ms | 补齐版均值ms | 时间减少 |', '|---|---:|---:|---:|']
    labels = {'restore_work_ms':'状态恢复（排除核验）','prefill_ms':'完整模型prefill','decode_ms':'HTP decode 16 token','prefill_decode_ms':'prefill＋decode','total_ms':'状态恢复＋prefill＋decode'}
    for metric in metrics:
        lines.append(f"| {labels[metric]} | {timing['baseline'][metric]['mean']:.3f} | {timing['guarded'][metric]['mean']:.3f} | {reduction[metric]:.2%} |")
    lines += ['', '| 进程对照块 | prefill时间减少 | 恢复＋prefill＋decode时间减少 |', '|---|---:|---:|']
    for suffix in ['a', 'b']:
        lines.append(f"| {suffix} | {blocks[suffix]['prefill_ms']:.2%} | {blocks[suffix]['total_ms']:.2%} |")
    mixed = blocks['a']['total_ms'] * blocks['b']['total_ms'] < 0
    lines += ['', '负数表示变慢。' + ('两块对照方向不一致，不能据此确认稳定的整体收益，也不能仅凭合并均值断言机制必然变慢。'
        if mixed else '两块对照方向一致，但只有两个进程对，不能据此宣称对其他设备条件普遍成立。')]
    lines += ['', '## 数值与状态', '',
              f"完整prefill logits：cosine={numerical['prefill']['cosine']:.9f}，相对L2差异={numerical['prefill']['nrmse']:.4%}，最大绝对差异={numerical['prefill']['max_abs']:.6f}；top1相同={numerical['prefill']['top1_equal']}。",
              f"D16之后logits：cosine={numerical['forced']['cosine']:.9f}，相对L2差异={numerical['forced']['nrmse']:.4%}，最大绝对差异={numerical['forced']['max_abs']:.6f}；top1相同={numerical['forced']['top1_equal']}。",
              f"原版/补齐版greedy延续token完全相同={greedy_equal}；首次分叉位置（从1开始）={first_difference}。该延续从受控长度截断的历史开始，不冒充原BFCL任务成功或JSON语义验收。",
              '每个进程的完整prefill状态恢复后，继续相同D16，最终全部logits及完整状态checksum与未恢复路径完全一致；全部40次计时最终logits也与各自验证参考一致。',
              f"原版/补齐版prefill状态checksum相同={states['prefill']['cross_variant_equal']}；D16后状态checksum相同={states['forced']['cross_variant_equal']}。状态可恢复不等于两个版本状态逐位相同。", '',
              '## 方法与限制', '',
              'ABBA四个独立进程，各验证、预热一次后测10次，全部保留，不去长尾。两个进程块的时间变化分别保存在block_time_reduction；只有两个进程块，不能把20次进程内重复当作20次独立设备条件。',
              'restore_work使用原有restore_session工作计时，完整导出/字节比较开销单列并排除；prefill与decode都在同一HTP上下文完成。模型初始化、状态checksum与导出、logits读取/比较、JSON写入、自由greedy验收均不计入性能段。',
              '这是固定历史之后的完整模型推理段，不是包含工具调用、网络和冷启动的完整Agent端到端。不同HVX/HMX路径存在数值差异，不要求严格allclose通过，原始结果全部保留。', '',
              '自由greedy延续（原版）：', '', '```text', greedy['baseline-a']['greedy_text'], '```', '',
              '自由greedy延续（补齐版）：', '', '```text', greedy['guarded-a']['greedy_text'], '```']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(time_reduction=reduction, block_time_reduction=blocks, numerical=numerical,
                          greedy_equal=greedy_equal, state_checks=states), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
