#!/usr/bin/env python3
"""汇总原版计时，私有profile只确认实际路由；保留不等价的负面结果。"""
import collections
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.validation.htp_layout import parse_layout_line
out=ROOT/'results/20261111-pad31-matmul'
data=json.loads((out/'unprofiled.json').read_text());profile=json.loads((out/'profiled.json').read_text());assert len(data['cases'])==len(profile['cases'])==3
records=[r for line in (out/'profiled.log').read_text().splitlines() if (r:=parse_layout_line(line))]
assert len(records)==66 and len(records)==sum('profile-op MUL_MAT:' in line for line in (out/'profiled.log').read_text().splitlines())
assert all(r['path']==('HVX' if r['M']==31 else 'HMX') for r in records)
assert {r['M'] for r in records}=={31,32}
summary=[]
for case in data['cases']:
    assert len(case['trials'])==10
    values=collections.defaultdict(list)
    orders=collections.Counter()
    for trial in case['trials']:
        orders[trial['results'][0]['condition']]+=1
        assert {r['condition'] for r in trial['results']}=={'plain31','pad32_discard_last'}
        for row in trial['results']:values[row['condition']].append(row)
    assert sorted(orders.values())==[5,5]
    timing={key:{metric:statistics.mean(r[metric] for r in rows) for metric in ['compute_ms','total_ms','setup_upload_ms','read_crop_ms']} for key,rows in values.items()}
    item=dict(tensor=case['tensor'],K=case['K'],N=case['N'],timing_ms=timing,
              total_time_reduction=1-timing['pad32_discard_last']['total_ms']/timing['plain31']['total_ms'],
              compute_time_reduction=1-timing['pad32_discard_last']['compute_ms']/timing['plain31']['compute_ms'],
              retained_output_comparison=case['retained_output_comparison'],
              plain31_vs_fp32_reference=case['plain31_vs_fp32_reference'],pad32_vs_fp32_reference=case['pad32_vs_fp32_reference'])
    summary.append(item)
result=dict(scope=data['scope'],repetitions=10,diagnostic_matrix_ops=len(records),route_verified=True,cases=summary,
            all_bitwise_equal=all(r['retained_output_comparison']['bitwise_equal'] for r in summary),
            all_strict_allclose=all(r['retained_output_comparison']['allclose_atol_1e_5_rtol_1e_5'] for r in summary))
(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
report=['# 31行补零到32行，再丢弃补行的最小验证','','## 结论','',
        '三种真实Q4_0权重矩阵中，补行均触发HVX→HMX。保留31行输出不逐位相同，也未通过atol=rtol=1e-5的严格近似检查。性能因矩阵形状而异，不能当作普遍等价加速。','',
        '| 矩阵 | K×N | 原31行总耗时ms | 补32行并裁剪ms | 总耗时减少 | 相对L2差异 | 最大绝对差异 |',
        '|---|---|---:|---:|---:|---:|---:|']
for r in summary:
    report.append(f"| {r['tensor']} | {r['K']}×{r['N']} | {r['timing_ms']['plain31']['total_ms']:.3f} | {r['timing_ms']['pad32_discard_last']['total_ms']:.3f} | {r['total_time_reduction']:.2%} | {r['retained_output_comparison']['nrmse']:.4%} | {r['retained_output_comparison']['max_abs']:.6f} |")
report += ['','## 方法与边界','',
           '复用手机Qwen3.5-9B-Q4_0 GGUF中的layer0 QKV、FFN Up、FFN Down原始权重，不复制模型文件。激活输入为固定seed20261119、均匀[-1,1]的合成F32矩阵；前31行完全相同，第32行为零。',
           '每矩阵每条件10次，5次原31行先、5次补32行先，固定种子打乱顺序；两条件各预热一次。原版库profile=0用于计时；私有v75 DSP探针profile=2另跑一次完整批次，只验证路径，不混入计时。66个诊断矩阵记录全部带直接路径标签。',
           '总耗时包含补零/拷贝、输入上传、计算同步、完整输出读回和裁剪。权重上传、重排、图分配与F32参考计算在计时外；因此这是热权重单算子微基准。',
           'F32参考使用相同Q4权重的解量化值，在CPU6线程计算。补32行结果在本次三种样例中均更接近该参考，但这不代表完整模型回答质量改善。',
           '这不是同一H下的完整31-token prefill测试：单个独立矩阵没有H或recurrent状态。没有在真实prompt追加dummy token，也没有裁剪Qwen3.5 recurrent/conv状态。若做完整模型方案，必须在算子内部正确处理padding，另验证最终logits、完整状态与后续decode。',
           '数学上独立矩阵行补零不改变前31行；但HVX/HMX实现与数值近似不同，因此实际浮点输出会变化。数值不等价的试次没有被筛掉，也没有计为数值通过。',
           '','## 文件','',
           '`unprofiled.json/.log` 为原版原始记录，`profiled.json/.log` 为诊断记录，`manifest.json` 为构建/模型/输入和探针身份，`build/` 为构建清单及日志。构建入口为 `scripts/build/pad31_matmul.py`，设备入口为 `scripts/run/pad31_matmul.py`，本汇总由 `scripts/analyze/pad31_matmul.py` 复现。']
(out/'README.md').write_text('\n'.join(report)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
