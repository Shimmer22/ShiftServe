#!/usr/bin/env python3
"""只分析NPU稀疏草稿→CPU完整验证的三个关键假设。"""
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/20261113-npu-draft-key-probe'
CPU = ROOT / 'results/20261113-cpu-verify-control/cpu.json'
OLD_CPU = ROOT / 'results/20261112-ffn-sparse-feasibility/cpu.json'


def main():
    cpu = json.loads(CPU.read_text())
    old_cpu = json.loads(OLD_CPU.read_text())
    npu = json.loads((OUT / 'htp.json').read_text())
    assert cpu['completed'] and npu['completed']
    assert cpu['normal_response']['passed'] and npu['normal_response']['passed']
    assert cpu['config']['prefix_ids'] == npu['config']['prefix_ids']
    assert cpu['config']['input_ids'] == npu['config']['input_ids']
    assert cpu['config']['context_size'] == npu['config']['context_size'] == 8192
    assert cpu['config']['generated_ids'][:8] == npu['config']['generated_ids'][:8]
    assert npu['history_source']['model_tokens_context_checksum_and_restore_verified']
    assert len(npu['rows']) == len(npu['config']['keep_ratios']) * (npu['config']['rounds'] + 1)
    assert all(r['finite_logits'] and math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0 for r in npu['rows'])
    def times(d, i):
        return [r['elapsed_ms'] / r['D'] for r in d['rows'] if r['stage'] == 'sparse_decode'
                and r['condition_index'] == i and not r['warmup']]
    controls = times(cpu, 0)
    baseline = median(controls)
    sparse = []
    for i, c in enumerate(npu['conditions']):
        assert len(times(npu, i)) == 3
        sparse.append(dict(condition_index=i, keep_ratio=c['keep_ratio'], ms_per_token=median(times(npu, i)),
            repetitions_ms_per_token=times(npu, i), packing_ms=c['packing_ms'], packed_bytes=c['packed_bytes'],
            inplace_compaction=c['inplace_compaction'],
            max_major_fault_delta=max(r['resources_after']['major_faults'] - r['resources_before']['major_faults']
                for r in npu['rows'] if r['condition_index'] == i and not r['warmup'])))
    verify = []
    for k in cpu['config']['verify_k']:
        trials = [r['elapsed_ms'] for r in cpu['rows'] if r['stage'] == 'verify' and r['k'] == k and r['batch']]
        assert len(trials) == 3
        numerical = next(v for v in cpu['verification'] if v['k'] == k)
        verify.append(dict(k=k, cpu_batch_ms=median(trials),
            argmax_matches=sum(v['argmax_match'] for v in numerical['per_position_logits']),
            continuation_argmax_match=numerical['continuation']['argmax_match'],
            max_affordable_draft_ms=((k + 1) * baseline - median(trials)) / k))
    bounds = []
    for s in sparse:
        if s['keep_ratio'] in (0, 1):
            continue
        for v in verify:
            cost = v['k'] * s['ms_per_token'] + v['cpu_batch_ms']
            bounds.append(dict(keep_ratio=s['keep_ratio'], k=v['k'],
                ideal_ms_per_output_token=cost / (v['k'] + 1),
                ideal_speedup=baseline * (v['k'] + 1) / cost,
                ideal_speedup_cpu_control_range=[c * (v['k'] + 1) / cost for c in sorted(controls)]))
    best = max(bounds, key=lambda r: r['ideal_speedup'])
    zero = next(s for s in sparse if s['keep_ratio'] == 0)
    zero_best = max((dict(k=v['k'], ideal_ms_per_output_token=(v['k'] * zero['ms_per_token'] + v['cpu_batch_ms']) / (v['k'] + 1),
        ideal_speedup=baseline * (v['k'] + 1) / (v['k'] * zero['ms_per_token'] + v['cpu_batch_ms'])) for v in verify), key=lambda r: r['ideal_speedup'])
    result = dict(scope='仅NPU稀疏草稿→CPU完整验证；不测反向分工、重要性迁移或真实接受率',
        cpu_dense_controls_ms_per_token=controls, cpu_reference_ms_per_token=baseline,
        npu_draft=sparse, cpu_verify=verify, optimistic_bounds=bounds, best_optimistic_bound=best,
        zero_ffn_diagnostic_best=zero_best, cpu_rollback=old_cpu['rollback'],
        bounds_assumptions='全部草稿接受且每块额外推进1个token；忽略打包、同步、状态追赶及回滚；不是实际投机加速',
        initial_attempts='按层、整块额外分配HTP权重均发生FastRPC映射失败；最终原地复用私有进程已有映射，模型文件未写入',
        cpu_raw=str(CPU.relative_to(ROOT)), npu_raw='htp.json')
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# NPU草稿、CPU验证：关键假设实机检查', '',
        '仅验证三个问题：缩窄FFN能否降低NPU草稿成本；CPU完整验证能否摊销；拒绝后能否恢复状态。暂不验证prefill重要性迁移、通道选择质量和草稿接受率。', '',
        'Qwen3.5-9B Q4_0，小米14 Pro；H4096/P40、context8192、batch128、F16状态；CPU6线程、HTP主机4线程。模型条件持锁串行，无KV整理、预搬移或手机设置修改。NPU之后补一个当前时段CPU短对照，只测完整模型decode及k=1/4验证，不测CPU稀疏草稿。', '',
        '## 1. NPU草稿成本', '',
        '完整prefill后替换各层FFN前部连续通道，gate/up保留对应行、down保留对应列，256通道对齐。原地复用进程已有HTP映射并重新打包，不写模型文件；各稀疏条件结束后从原GGUF重装完整FFN。紧凑FFN矩阵均通过HTP支持检查，权重仍使用HTP buffer；后端不可用时直接失败。', '',
        '每条件预热1次，正式3次、每次8个token；完整状态恢复和打包独立计时。所有条件关闭图复用，不能直接与旧生产路径拼接。历史快照核对模型身份、全部token、上下文、校验和及恢复后的完整状态字节；prefill/decode使用同一上下文。', '',
        '| FFN保留比例 | NPU ms/token | 打包 ms |', '|---|---:|---:|']
    for i, s in enumerate(sparse):
        label = f"{s['keep_ratio']:.0%}" + (' 回测' if i == len(sparse) - 1 else '')
        lines.append(f"| {label} | {s['ms_per_token']:.2f} | {s['packing_ms']:.1f} |")
    lines += ['', f"当前CPU完整decode为{baseline:.2f} ms/token，三次范围{min(controls):.2f}–{max(controls):.2f}。0%为去掉FFN矩阵、保留SiLU(norm(x))的性能诊断，不能生成有效草稿。固定通道未按重要性选择，有限logits不代表回答质量通过。", '',
        'NPU完整FFN前后对照从551.70降至254.09 ms/token，50%条件的三次也有明显波动。因此不能把首个完整条件与稀疏条件的全部差值归因于剪枝；报告保留全部重复，不筛选异常点。本结论只使用草稿与验证的实测成本判断当前组合是否有空间。', '',
        '## 2. 即使全接受，能否加速', '',
        '用乐观成本 `(k × NPU草稿单步 + CPU批量验证)/(k+1)`；额外1个target token、全接受、无同步/回滚/状态追赶/打包成本。', '',
        '| 草稿长度k | CPU验证 ms | 盈亏平衡要求：NPU单步低于 ms | CPU批量/逐token argmax一致数 |',
        '|---:|---:|---:|---:|']
    for v in verify:
        lines.append(f"| {v['k']} | {v['cpu_batch_ms']:.2f} | {v['max_affordable_draft_ms']:.2f} | {v['argmax_matches']}/{v['k']} |")
    lines += ['', f"最有利实测宽度点：保留{best['keep_ratio']:.0%}、k={best['k']}，乐观估算{best['ideal_ms_per_output_token']:.2f} ms/token，相对CPU为{best['ideal_speedup']:.3f}×；按CPU三次对照取值为{best['ideal_speedup_cpu_control_range'][0]:.3f}–{best['ideal_speedup_cpu_control_range'][-1]:.3f}×。小于1表示仍慢于CPU。不是实际投机加速。", '',
        f"去掉全部FFN矩阵的诊断点，最乐观估算仍为{zero_best['ideal_ms_per_output_token']:.2f} ms/token、{zero_best['ideal_speedup']:.3f}×。在本轮已测k=1/4下，FFN剪枝本身没有打开NPU draft→CPU verify的净收益空间。", '',
        'CPU部分块长存在批量与逐token argmax差异，因此严格保持原输出的验证路径尚未通过；不能只看验证吞吐。', '',
        '## 3. 拒绝后的状态恢复', '',
        f"前一批CPU探针中，recurrent部分尾部删除返回{old_cpu['rollback']['partial_seq_rm_returned']}；完整检查点恢复通过字节核验，耗时{old_cpu['rollback']['restore_ms']:.2f} ms。恢复后重放4个已接受token另耗{old_cpu['rollback']['replay_accepted_4_ms']:.2f} ms。它们属于先前时段，不拼接为本轮实测E2E；上述乐观估算忽略了这些额外成本。", '',
        '## 结果边界', '',
        '这是固定H、短decode、固定连续通道的性能探针，不是τ²任务E2E，也没有检验重要性先验与真实接受率。前两次额外HTP权重分配失败及原始日志保留在20261112-ffn-sparse-feasibility和20261112-npu-draft-key-probe。', '',
        '原始响应：htp.json；CPU当前短对照：../20261113-cpu-verify-control/cpu.json；旧CPU状态检查：../20261112-ffn-sparse-feasibility/cpu.json；分析：summary.json；配置：configs/npu-draft-key-probe.json与configs/cpu-verify-key-control.json；分析入口：scripts/analyze/npu_draft_key_probe.py。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(cpu_dense=controls, npu=sparse, best=best), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
