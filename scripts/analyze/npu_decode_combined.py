#!/usr/bin/env python3
"""组合优化的时延、充分预热交错对照、JSON/记忆/检查点与数值验收。"""
from array import array
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analyze'))
from decode_attribution import read_case, numerical
OUT=ROOT/'results/20261113-npu-decode-combined'
WARM=ROOT/'results/20261113-npu-decode-combined-warm'


def main():
    cases={};functional={};data={}
    for label in ['original','optimized','cpu']:
        d,s,_=read_case(OUT/label);cases[label]=s;data[label]=d
        cp=d['checkpoint_replay'];memory=d['json_memory']
        functional[label]=dict(normal=d['normal_response']['passed'],checkpoint_restore_bytes=cp['restored_bytes_verified'],
            replay_final_state_bytes=cp['final_state_bitwise_equal'],replay_argmax=cp['logits']['argmax_match'],
            json_and_memory=memory['passed'],turns=[v['response'] for v in memory['turns']])
        assert all(functional[label][k] for k in ['normal','checkpoint_restore_bytes','replay_final_state_bytes','replay_argmax','json_and_memory'])
    ref=array('f');ref.frombytes((OUT/'original/logits.f32').read_bytes())
    comparisons={label:numerical(OUT/label,ref,data[label]['numerical_replay']) for label in ['optimized','cpu']}
    warm={}
    for label in ['original-before','optimized','original-after']:
        d,s,_=read_case(WARM/label);assert d['warmup_elapsed_ms']>=40000
        s['warmup_seconds']=d['warmup_elapsed_ms']/1000;s['warmup_runs']=d['warmup_runs']
        s['observed_cpu_state']=[dict(before=r['cpu_state_before'],after=r['cpu_state_after']) for r in d['rows'] if not r['warmup']]
        warm[label]=s
    before=warm['original-before']['ms_per_token'];after=warm['original-after']['ms_per_token'];opt=warm['optimized']['ms_per_token'];control=median([before,after])
    stable=max(before,after)/min(before,after)<=1.10 and all(max(v['repetitions_ms_per_token'])/min(v['repetitions_ms_per_token'])<=1.10 for v in warm.values())
    stats=dict(reference_ms_per_token=control,optimized_ms_per_token=opt,speedup=control/opt,reduction=1-opt/control,
        reduction_vs_before=1-opt/before,reduction_vs_after=1-opt/after,
        all_optimized_faster_than_all_controls=max(warm['optimized']['repetitions_ms_per_token'])<min(warm['original-before']['repetitions_ms_per_token']+warm['original-after']['repetitions_ms_per_token']),
        before_after_controls_within_10_percent=stable,scope='单一H4096/P40，前后原配置与中间优化配置；每条件正式3次×16token；不把重复当独立任务')
    matched=sum(r['argmax_match'] for r in comparisons['optimized']);total=len(comparisons['optimized'])
    result=dict(cases=cases,functional=functional,numerical=comparisons,warm_cases=warm,warm_comparison=stats,
        optimized_argmax_matches=matched,positions=total,
        adoption='仅保留显式候选；跨版本逐位置greedy未完全一致，不切换默认运行时或更新路由成本表' if matched<total else '短验收通过，仍是有限范围候选，不代表任务E2E或所有H/P',
        runtime='CPU融合AR DeltaNet + HTP矩阵；OPBATCH1024 OPQUEUE16；F16状态仍在CPU，图复用开启；不改CH prefill融合开关')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# NPU decode组合优化候选', '',
        '把CPU融合AR DeltaNet与HTP批量提交组合，先验证完整模型，再考虑稀疏。使用Qwen3.5-9B Q4_0、H4096/P40、context8192、batch128、F16状态、CPU6线程/HTP主机4线程。只有CPU与HTP，全部Termux SSH持锁串行；无KV整理、预搬移或手机设置/频率修改。', '',
        '## 初次组合测量', '',
        '| 条件 | ms/token中位数 | 全部正式重复 |', '|---|---:|---|']
    for label in ['original','optimized','cpu']:
        s=cases[label];lines.append(f"| {label} | {s['ms_per_token']:.2f} | "+', '.join(f'{v:.2f}' for v in s['repetitions_ms_per_token'])+' |')
    lines += ['', '每条件预热一次，正式3次×16个token。原配置最后一次在同一重复内突然从约540降至约250 ms/token，故不能直接把两个中位数的全部差距称为优化收益。全部逐token耗时保留，未删除突变点。', '',
        '## 补充：充分预热与交错原配置对照', '',
        '按原配置→组合优化→原配置顺序串行执行。每条件至少40秒纯decode预热，再测3次×16token；全部预热单列，恢复、校验、初始化不混入decode。CPU频率与cgroup只读观察，不写入设置。', '',
        '| 条件 | 预热纯decode秒 | 正式ms/token | 全部正式重复 |', '|---|---:|---:|---|']
    for label in ['original-before','optimized','original-after']:
        s=warm[label];lines.append(f"| {label} | {s['warmup_seconds']:.2f} | {s['ms_per_token']:.2f} | "+', '.join(f'{v:.2f}' for v in s['repetitions_ms_per_token'])+' |')
    lines += ['', f"前后原配置中值{control:.2f} ms/token；组合配置{opt:.2f} ms/token，比值{control/opt:.3f}×，耗时变化{(1-opt/control):.1%}。前后基线及各组正式重复是否均处于10%范围：{stable}。它是当前一个状态点的对照，不是多任务统计结论。", '',
        f"组合配置的全部正式观测均快于两个原配置组的全部观测：{stats['all_optimized_faster_than_all_controls']}。按两侧组中位数分别减少{stats['reduction_vs_before']:.1%}、{stats['reduction_vs_after']:.1%}；控制组存在时段漂移，不能把单个25%左右的数值宣称为稳定总体收益。", '',
        '## 功能与状态', '',
        '- 原配置、组合配置、CPU均返回正常OK。',
        '- 连续生成16个固定token，与在第8个token保存/恢复检查点后继续生成相比，各条件内部最终状态逐字节相同，logits和argmax一致。',
        '- 两轮自由生成均返回有效JSON `{"code":314159}`；第二轮沿用第一轮同一上下文、未恢复或重建历史，成功记住数值。',
        f"- 原配置与组合配置跨版本的16个位置，argmax一致{matched}/{total}；最低cosine {min(v['cosine'] for v in comparisons['optimized']):.8f}。这是数值漂移，不能宣称输出严格无损。CPU对照另列在summary.json。", '',
        '## 使用边界与下一步', '',result['adoption']+'。', '',
        '候选开关：私有libllama.so设置 `A2O_DIAG_CPU_FUSED_GDN=1`，HTP设置 `GGML_HEXAGON_OPBATCH=1024`、`GGML_HEXAGON_OPQUEUE=16`，图复用开启，offload_kqv=false。AR融合放CPU，其他矩阵仍显式HTP。共享安装库与默认配置未改。未测稀疏、任务E2E或所有prefill形状。', '',
        '源补丁：src/a2o/native/backend/cpu-fused-gdn.patch；配置：configs/npu-decode-combined.json与configs/npu-decode-combined-warm.json；构建：scripts/build/decode_attribution.py --followup --tag <新具名构建>；采集：scripts/run/decode_attribution.py；分析：scripts/analyze/npu_decode_combined.py。原始数据分别在本目录与../20261113-npu-decode-combined-warm/。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(warm=stats,argmax=f'{matched}/{total}',functional=functional),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
