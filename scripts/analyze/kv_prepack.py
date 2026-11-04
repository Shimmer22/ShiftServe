#!/usr/bin/env python3
"""复核实际 HTTP + 原生 KV 预整理对照；不以离线公式替代并发。"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

def analyze(out):
    out=Path(out)
    read=lambda name:[json.loads(s) for s in (out/name).read_text().splitlines()]
    manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['config']
    rows=read('checked-rows.jsonl');raw=read('device/trials.jsonl');http=read('http-server.jsonl');events=read('native-events.jsonl')
    calibration=[r for r in rows if r['phase']=='calibration'];measured=[r for r in rows if r['phase']=='measurement']
    expected=6*len(cfg['new_tokens'])+4*len(cfg['new_tokens'])*len(cfg['gap_ms'])*cfg['repetitions']
    by_http={r['id']:r for r in http};published={e['turn']:e for e in events if e['event']=='response_published'}
    packed=[r for r in rows if r['route']=='HC' and r['kv_prepack_enabled']]
    qualified=json.loads((out/'candidate-qualification.json').read_text())
    checks={
        'completed':json.loads((out/'completion.json').read_text())['passed'],
        'counts':len(rows)==len(raw)==len(http)==len(published)==expected,
        'raw_preserved':len(rows)==len(raw) and all(all(c[k]==v for k,v in r.items()) for c,r in zip(rows,raw)),
        'unique_ids':{r['id'] for r in rows}==set(range(expected)) and len(by_http)==expected,
        'http_conditions_and_hashes':all(r['id'] in by_http and (by_http[r['id']]['P'],by_http[r['id']]['gap_requested_ms'])==(r['P'],r['gap_requested_ms']) and by_http[r['id']]['body_sha256']==hashlib.sha256((out/f"P{r['P']}.txt").read_bytes()).hexdigest() for r in rows),
        'json_ids_eog_state':all(r['passed'] and r['response_check']['passed'] and r['eog'] and (r['route']!='HC' or r['handoff_bytes']>0) for r in rows),
        'one_output':len({(r['output'],tuple(r['generated_ids']),r['D']) for r in rows})==1,
        'qualified':all(qualified.values()) and all(r['numerical_passed'] for r in calibration),
        'exact_htp_logits_and_final_state':all(sum(r['P']==p and r['route']=='HC' and r['kv_prepack_enabled'] and r['htp_baseline_compared'] and r['htp_baseline_state_compared'] and r['htp_baseline_nrmse']==0 for r in calibration)==2 for p in cfg['new_tokens']),
        'full_kv_reuse':bool(packed) and all(r['kv_prepack_entries']==72 and r['kv_prepack_cache_bytes']==cfg['history_tokens']*256*2*72 and r['kv_prepack_k_hits']==r['kv_prepack_v_hits']==72 and r['kv_prepack_k_bytes']==r['kv_prepack_cache_bytes']/2 and r['kv_prepack_v_bytes']==r['kv_prepack_cache_bytes']/2 for r in packed),
        'requested_release_applied':not cfg.get('htp_kv_release_before_decode',False) or all(r.get('kv_prepack_released_before_decode',False) for r in packed),
        'uncached_path_no_reuse':all(r['kv_prepack_k_hits']==r['kv_prepack_v_hits']==0 for r in rows if not r['kv_prepack_enabled']),
        'response_publication':all(r['id'] in published and r['response_compute_completed_ms']<=published[r['id']]['monotonic_ms']<=r['response_completed_ms'] for r in rows),
        'source_binary_recorded':hashlib.sha256((out/'device/agent').read_bytes()).hexdigest()==manifest['binary_sha256'],
    }
    releases=[e for e in events if e['event']=='calibration_buffers_released']
    checks['calibration_state_released_before_measurement']=not cfg.get('release_calibration_state',False) or (
        len(releases)==1 and releases[0]['state_bytes']>0 and releases[0]['monotonic_ms']<=min(r['http_issued_ms'] for r in measured)
        and all(r.get('calibration_state_retained_bytes',-1)==0 for r in measured))
    groups=[];comparisons=[]
    for p in cfg['new_tokens']:
        for g in cfg['gap_ms']:
            cells={}
            for strategy in ('after_return','always'):
                for pack in (False,True):
                    part=[r for r in measured if (r['P'],r['gap_requested_ms'],r['strategy'],r['kv_prepack_enabled'])==(p,g,strategy,pack)]
                    assert len(part)==cfg['repetitions'],(p,g,strategy,pack,len(part))
                    cells[strategy,pack]=part
                    values={k:median(r[k] for r in part) for k in ['response_after_return_ms','tool_to_response_ms','actual_gap_ms','prepare_ms','kv_prepack_ms','prefill_ms','handoff_checked_ms','decode_ms','wait_or_prepare_after_return_ms']}
                    values['kv_prepack_release_ms']=median(r.get('kv_prepack_release_ms',0) for r in part)
                    groups.append({'P':p,'G_target_ms':g,'strategy':strategy,'prepack':pack,'n':len(part),'ready_count':sum(r['ready_at_return'] for r in part),'median':values})
            med=lambda s,b,k='response_after_return_ms':median(r[k] for r in cells[s,b])
            paired={r['repetition']:r for r in cells['always',False]}
            diffs=[paired[r['repetition']]['response_after_return_ms']-r['response_after_return_ms'] for r in cells['always',True]]
            comparisons.append({'P':p,'G_target_ms':g,
                'incremental_prepack_gain_ms':med('always',False)-med('always',True),
                'combined_gain_ms':med('after_return',False)-med('always',True),
                'prepack_after_return_penalty_ms':med('after_return',True)-med('after_return',False),
                'prepack_gap_overlap_gain_ms':med('after_return',True)-med('always',True),
                'paired_gain_median_ms':median(diffs),'paired_positive_count':sum(d>0 for d in diffs),
                'paired_gain_range_ms':[min(diffs),max(diffs)],'n':len(diffs)})
    overlap=[r for r in measured if r['prepared_early'] and r['kv_prepack_enabled']]
    checks['actual_prepack_http_overlap']=len(overlap)==len(cfg['new_tokens'])*len(cfg['gap_ms'])*cfg['repetitions'] and all(min(r['prepare_completed_ms'],r['http_returned_ms'])>max(r['prepare_started_ms'],r['http_issued_ms']) for r in overlap)
    timing_keys=('prepare_ms','kv_prepack_ms','prefill_ms','handoff_checked_ms','decode_ms')
    stages=[{'prepack':pack,'n':sum(r['kv_prepack_enabled']==pack for r in measured),
             'median':{k:median(r[k] for r in measured if r['kv_prepack_enabled']==pack) for k in timing_keys}}
            for pack in (False,True)]
    response_range=[min(r['response_after_return_ms'] for r in measured),max(r['response_after_return_ms'] for r in measured)]
    result={'passed':all(checks.values()),'diagnostic_only':cfg.get('diagnostic_only',False),'timing_valid_for_formal_comparison':not cfg.get('diagnostic_only',False),'checks':checks,'trials':len(rows),'D':rows[0]['D'],
            'measurement_response_range_ms':response_range,'stage_medians':stages,
            'groups':groups,'comparisons':comparisons,'packed_early_trials':len(overlap),'packed_ready_trials':sum(r['ready_at_return'] for r in overlap),
            'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'prepack-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert result['passed'],checks
    lines=['# 历史 KV 的 HMX tiles 提前准备对照','',
        f"H={cfg['history_tokens']}，P={cfg['new_tokens']}，完整自由生成 D={result['D']}（含 EOG）；共 {len(rows)} 条记录。每条件 {cfg['repetitions']} 次，随机执行顺序。",
        '', '私有 host bridge + v79 DSP，明确 hybrid：HTP prefill、CPU decode；Q4_0 模型、F16 KV、无压缩、Flash Attention 关闭。默认行为不启用预整理。',
        '', '本轮固定 HC 路径，比较预整理开/关 × 返回后/提前准备；CC 只用于独立校准。这不是新的 CC/HC Router 或联合 Oracle 选路收益测量。',
        '', '## 准入与机制', '',
        '每个 P 先运行原 HC，再要求预整理 HC 的完整 prefill logits 数值完全相同、最终 CPU sequence 状态逐字节相同。独立 CPU/HC 数值门槛保持配置值，原始结果保留。所有输出均为同一完整 JSON、同一生成 ID 序列。',
        '', f"K/V 共 72 份历史缓存，额外空间 {packed[0]['kv_prepack_cache_bytes']/2**20:.0f}MiB；K、V 历史 tiles 均在真实 attention 算子中读取。新增输入与填充尾部仍在返回后整理。缓存在每个 trial 前显式失效，跨轮更新、滑动窗口、多序列不在原型范围。",
        '', f"提前预整理 {len(overlap)} 次全部与真实 HTTP 区间重叠，其中 {result['packed_ready_trials']} 次在 Tool 返回时已 ready。以下 G 为服务端目标，实际手机 Gap 见分组数据，不将 adb reverse 往返视为零。",
        '', '## 返回后到完整响应', '', '单位 ms；中位数。增量收益＝仅状态提前准备 − 状态与 tiles 均提前准备；正值表示 tiles 有额外收益。', '',
        '| P | G目标 | 普通/返回后准备 | 普通/提前准备 | tiles/返回后准备 | tiles/提前准备 | tiles增量收益 | 配对获益次数 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    if cfg.get('release_calibration_state'):
        lines+=['', '本轮在第一条测量开始前释放校准专用的完整状态与 logits 副本；生成 ID/响应签名、数值准入和独立校准原始记录保留。']
    if cfg.get('htp_kv_release_before_decode'):
        lines+=['', '本轮 tiles 在 HTP prefill 完成后、CPU 状态交接/decode 前读取计数并释放；释放 RPC 的实测耗时计入完整响应，单列 kv_prepack_release_ms。响应后再次核验缓存已清空。']
    if cfg.get('diagnostic_only'):
        lines[2:2]=['本轮为带资源计数的错误排查回归，计时仅作诊断，不并入正式性能结论。','']
    for c in comparisons:
        selected=[r for r in groups if r['P']==c['P'] and r['G_target_ms']==c['G_target_ms']]
        v={(r['strategy'],r['prepack']):r['median']['response_after_return_ms'] for r in selected}
        lines.append(f"| {c['P']} | {c['G_target_ms']} | {v['after_return',False]:.1f} | {v['always',False]:.1f} | {v['after_return',True]:.1f} | {v['always',True]:.1f} | {c['incremental_prepack_gain_ms']:.1f} | {c['paired_positive_count']}/{c['n']} |")
    lines+=['','## 阶段与结论','',
        '准备工作成功提前并不等于完整响应更快；是否有额外收益须结合分组完整响应、配对结果和长尾判断，不能只看 prefill 或准备耗时。',
        '', f"全部测量的返回后响应范围为 {response_range[0]/1000:.2f}–{response_range[1]/1000:.2f}s，慢样本全部保留。下表合并 P/G/准备时机，仅用于描述阶段分布，各列中位数不能相加解释总耗时或因果收益。",
        '', '| 预整理 | 准备 | tiles整理 | prefill | 状态交接与核验 | CPU decode |',
        '|---|---:|---:|---:|---:|---:|']
    for stage in stages:
        v=stage['median']
        lines.append('| '+('开' if stage['prepack'] else '关')+' | '+' | '.join(f'{v[k]:.1f}' for k in timing_keys)+' |')
    lines+=['','## 解释范围','',
        '这是当前可运行原型的实测收益，不能当成所有可能优化的理论上限。准备包含状态 clear/import/字节核验、DSP 缓存分配、RPC 与实际 tile 整理；共同历史快照、fixture 重置和旧缓存回收在请求前，未计入 Tool→响应。校准额外逐元素核验 tiles，校准时延不混入测量。缓存计数读取和最终校准状态比较均在响应发布之后。',
        '', '预整理在 DSP DDR 中保存历史 tiles，返回后仍需 DMA 到 VTCM；无法让整个模型的 KV 常驻有限 VTCM。Q/K/V 新增投影、attention、CPU decode 和返回 CPU 的状态交接仍需在 Tool 返回后执行。本轮为固定文本和短 JSON 输出，无图像、能耗或跨轮复用结论。',
        '', '原始文件：`device/trials.jsonl`、`checked-rows.jsonl`、`http-server.jsonl`、`native-events.jsonl`、`runtime.log`；完整性检查和各阶段中位数见 `prepack-summary.json`。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    r=analyze(a.directory);print(json.dumps({'passed':r['passed'],'trials':r['trials'],'comparisons':r['comparisons']},ensure_ascii=False))
