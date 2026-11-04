#!/usr/bin/env python3
"""只汇总独立kernel诊断，不把profiling耗时混进正式Gap实验。"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
PARSER = ROOT.parent / 'shared/runtime/llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py'


def matrix_shape(op):
    if op['name'] != 'MUL_MAT':
        return None
    operands = op['dims'].split('->')[0].strip().split(' x ')
    if len(operands) != 2:
        return None
    a, b = [[int(x) for x in dims.strip().split(':')] for dims in operands]
    if len(a) < 2 or len(b) < 2 or b[1] not in (31, 32, 33):
        return None
    return {'M': b[1], 'K': a[0], 'N': a[1], 'weight_type': op['types'].split(' x ')[0].strip(),
            'batched': any(x > 1 for x in a[2:] + b[2:])}


def summarize(directories, output):
    spec = importlib.util.spec_from_file_location('shared_hexagon_profile', PARSER)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    runs = []
    for directory in directories:
        cfg = json.loads((directory / 'manifest.json').read_text())['config']
        if not cfg.get('calibration_only') or cfg.get('htp_profile') != 1:
            raise ValueError('只接受独立basic profile诊断')
        rows = [json.loads(s) for s in (directory / 'checked-rows.jsonl').read_text().splitlines()]
        if len(rows) != 6 * len(cfg['new_tokens']):
            raise ValueError('校准记录不完整')
        ops = parser.parse_log(str(directory / 'runtime.log'))
        if not ops:
            raise ValueError('未得到设备profile-op，不把空日志当验证')
        groups = defaultdict(list)
        for op in ops:
            shape = matrix_shape(op)
            if shape:
                groups[(shape['M'], shape['weight_type'], shape['K'], shape['N'], shape['batched'])].append(op['usec'])
        by_p = []
        for p in cfg['new_tokens']:
            sample = [r for r in rows if r['P'] == p and r['route'] == 'HC']
            if len(sample) != 3:
                raise ValueError('HC须三次校准，不挑样本')
            families = defaultdict(float)
            for (m, weight_type, _, _, _), values in groups.items():
                if m == p: families[weight_type] += sum(values) / 3 / 1000
            by_p.append({'P': p, 'prefill_ms': median(r['prefill_ms'] for r in sample),
                'D_values': sorted({r['D'] for r in sample}), 'responses_passed': all(r['passed'] and r['response_check']['passed'] for r in sample),
                'matrix_kernel_device_ms_per_call_by_weight_type': dict(families)})
        runs.append({'directory': str(directory.resolve()), 'config': cfg, 'profile_op_records': len(ops), 'by_P': by_p,
            'matrix_shapes': [{'M': key[0], 'type': key[1], 'K': key[2], 'N': key[3], 'batched': key[4],
                'op_records': len(values), 'median_usec': median(values), 'sum_usec': sum(values)} for key, values in sorted(groups.items())]})
    output.mkdir(parents=True, exist_ok=True)
    summary = {'runs': runs, 'shared_parser_path': str(PARSER), 'shared_parser_sha256': hashlib.sha256(PARSER.read_bytes()).hexdigest(),
        'scope': 'PROFILE=1与HMX开关是诊断；逐op日志是通用算子名，不是直接kernel分支打印；总wall时间含探针开销',
        'remote_compilation_performed': False}
    (output / 'kernel-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Hexagon kernel独立诊断', '', '正式H/P计时完成后单独运行PROFILE=1。H2048，P31/32/33，每条HC三次；默认HMX与显式HMX=0对照。没有编译或替换共享库。', '',
        '| HMX设置 | P | HTP prefill中位数 ms（含profile开销） | 矩阵乘device ms/次 | 响应通过 |', '|---|---|---|---|---|']
    for run in runs:
        for r in run['by_P']:
            lines.append(f'| {run["config"].get("htp_use_hmx", "默认")} | {r["P"]} | {r["prefill_ms"]:.1f} | {json.dumps(r["matrix_kernel_device_ms_per_call_by_weight_type"])} | {r["responses_passed"]} |')
    lines += ['', 'device时长为真实逐op计数按每P的三次HC调用归一；不是模拟重叠，也不等于端到端墙钟时间。详细矩阵shape、类型和op统计见JSON。', '',
        '若关闭HMX后P32/33的优势消失，支持HMX参与造成性能阶跃；不能由通用MUL_MAT名称直接宣称每一个算子的实际分支。直接分支打印仍需私有debug skel，远端工具链检查不等同于已经编译。']
    (output / 'kernel-report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(output / 'kernel-report.md'), 'runs': len(runs)}, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directories', type=Path, nargs='+'); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); summarize(args.directories, args.output)
