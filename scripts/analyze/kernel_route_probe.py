#!/usr/bin/env python3
"""验证DSP实际路由探针与响应、迁移、加载库身份，独立保存诊断结论。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.validation.kernel_probe import parse_route_line


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = manifest['config']
    if cfg.get('htp_profile') != 2 or not manifest.get('route_probe'):
        raise ValueError('不是私有route探针运行，不能将普通PMU数字解释成路由')
    rows = [json.loads(l) for l in (directory / 'checked-rows.jsonl').read_text().splitlines()]
    expected = 6 * len(cfg['new_tokens'])
    complete = len(rows) == expected and all(r['passed'] and r['response_check']['passed'] for r in rows)
    if not complete: raise ValueError('响应/状态验收或记录完整性失败')
    if len({tuple(r['generated_ids']) for r in rows}) != 1: raise ValueError('完整输出不一致')
    lines = (directory / 'runtime.log').read_text().splitlines()
    records = [record for line in lines if (record := parse_route_line(line))]
    matrix_lines = sum('profile-op MUL_MAT:' in line for line in lines)
    if not records or len(records) != matrix_lines:
        raise ValueError('探针未覆盖每个MUL_MAT，不能确认私有skel已实际生效')
    counts = Counter((r['M'], r['dsp_weight_type'], r['path'], r['reason']) for r in records)
    attention = [r for r in records if 'cache_k_' in r['names'] or 'cache_v_' in r['names']]
    if not attention: raise ValueError('没有attention实际路由证据')
    attention_counts = Counter((r['M'], r['path'], r['reason']) for r in attention)
    checks = {'all_rows_passed': complete, 'all_matrix_ops_have_direct_markers': True,
        'no_hmx_errors': all(r['reason'] != 'hmx_error' for r in records),
        'attention_M31_HVX': all(r['path'] == 'HVX' and r['reason'] == 'M_lt_32' for r in attention if r['M'] == 31),
        'attention_M32_M33_HMX': all(r['path'] == 'HMX' and r['reason'] == 'hmx_success' and r['hmx_return_code'] == 0 for r in attention if r['M'] in (32, 33)),
        'all_P_attention_present': all(any(r['M'] == p for r in attention) for p in cfg['new_tokens'])}
    if not all(checks.values()): raise ValueError(checks)
    summary = {'passed': True, 'checks': checks, 'diagnostic_only': True, 'hardware_pmu_values_valid': False,
        'matrix_ops': len(records), 'attention_ops': len(attention), 'D': rows[0]['D'],
        'route_counts': [{'M': k[0], 'dsp_weight_type': k[1], 'path': k[2], 'reason': k[3], 'count': n} for k,n in sorted(counts.items())],
        'attention_counts': [{'M': k[0], 'path': k[1], 'reason': k[2], 'count': n} for k,n in sorted(attention_counts.items())],
        'probe_sha256': manifest['route_probe']['sha256'], 'source_patch': 'src/a2o/native/profiling/htp-route-probe.patch'}
    (directory / 'direct-route-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (directory / 'direct-route-records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    report = ['# v79 DSP直接路由探针', '', f'共{len(rows)}条CPU/HC校准，完整输出一致，D={rows[0]["D"]}，响应与迁移通过。{len(records)}个MUL_MAT均带DSP实际分支标签。', '',
        '| Attention M | 实际路径 | 原因 | 记录数 |', '|---|---|---|---|']
    report += [f'| {r["M"]} | {r["path"]} | {r["reason"]} | {r["count"]} |' for r in summary['attention_counts']]
    report += ['', '标签由DSP原有分支处写入，经现有profile数据通道回传。未扩大RPC结构；本次pmu字段承载探针元数据，不能作为硬件PMU数值。该运行是诊断，不混入正式延迟结果。', '',
        '只加载本次临时目录的私有v79库，原安装库保留。补丁未改计算内核、门槛或原分支返回逻辑；探针库重新构建，构建与hash记录单列，不能把它冒充原发布库内部的直接日志。']
    (directory / 'README.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'run': str(directory), 'checks': checks, 'matrix_ops': len(records), 'attention_counts': summary['attention_counts']}, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
