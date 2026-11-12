#!/usr/bin/env python3
"""校验设备内路由、哨兵、跨进程完整输出，并汇总热算子时间。"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.validation.htp_layout import parse_layout_line


def compare(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = b.astype('float64') - a
    return dict(bitwise_equal=a.tobytes() == b.tobytes(), max_abs=float(np.max(np.abs(delta))),
                nrmse=float(np.linalg.norm(delta) / np.linalg.norm(a)),
                allclose_1e_5=bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    args = p.parse_args()
    out = args.input
    assert json.loads((out / 'status.json').read_text())['phase'] == 'complete'
    labels = ['baseline-a', 'guarded-a', 'force', 'guarded-b', 'baseline-b', 'guarded-profile']
    data = {label: json.loads((out / (label + '.json')).read_text()) for label in labels}
    text = (out / 'guarded-profile.log').read_text()
    records = [r for line in text.splitlines() if (r := parse_layout_line(line))]
    assert len(records) == text.count('profile-op MUL_MAT:') == 66
    for record in records:
        admitted = record['M'] == 31 and record['K'] == 4096 and record['N'] in {8192, 12288}
        assert record['path'] == ('HMX' if admitted or record['M'] == 32 else 'HVX')
        if admitted:
            assert record['mc'] == 32
    cases = []
    for i, reference in enumerate(data['baseline-a']['cases']):
        tensor = reference['tensor']
        times = collections.defaultdict(lambda: collections.defaultdict(list))
        output = {}
        for label in labels:
            case = data[label]['cases'][i]
            assert case['tensor'] == tensor and case['real_rows'] == 31 and case['output_guard_passed']
            assert case['input_bytes_plain31'] == case['K'] * 31 * 4
            assert case['output_bytes_plain31'] == case['N'] * 31 * 4
            assert len(case['trials']) == 10
            order = collections.Counter()
            for trial in case['trials']:
                order[trial['results'][0]['condition']] += 1
                assert {r['condition'] for r in trial['results']} == {'plain31', 'pad32_discard_last'}
                for row in trial['results']:
                    if label != 'guarded-profile':
                        group = 'baseline' if label.startswith('baseline') else 'guarded' if label.startswith('guarded') else 'force'
                        times[group][row['condition']].append(row)
            assert sorted(order.values()) == [5, 5]
            for condition in ['plain31', 'pad32']:
                path = out / (label + '.json.' + tensor + '.' + condition + '.f32')
                array = np.fromfile(path, dtype='<f4')
                assert array.size == case['N'] * 31
                output[(label, condition)] = array
        # 同HMX路径的外部31行与外部32行结果应逐位相同。
        for label in labels:
            assert compare(output[('baseline-a', 'pad32')], output[(label, 'pad32')])['bitwise_equal']
        assert compare(output[('force', 'plain31')], output[('force', 'pad32')])['bitwise_equal']
        eligible = reference['K'] == 4096
        for label in ['guarded-a', 'guarded-b', 'guarded-profile']:
            expected = output[(label, 'pad32')] if eligible else output[('baseline-a', 'plain31')]
            assert compare(expected, output[(label, 'plain31')])['bitwise_equal']
        assert compare(output[('baseline-a', 'plain31')], output[('baseline-b', 'plain31')])['bitwise_equal']
        timing = {group: {condition: dict(samples=len(rows), **{
            metric: statistics.mean(r[metric] for r in rows)
            for metric in ['compute_ms', 'total_ms', 'setup_upload_ms', 'read_crop_ms']})
            for condition, rows in conditions.items()} for group, conditions in times.items()}
        native = timing['guarded']['plain31']; baseline = timing['baseline']['plain31']
        cases.append(dict(tensor=tensor, K=reference['K'], N=reference['N'], admitted=eligible, timing_ms=timing,
            output_guard_passed=True, expected_output_bitwise_passed=True,
            vs_original=compare(output[('baseline-a', 'plain31')], output[('guarded-a', 'plain31')]),
            total_reduction_vs_original=1-native['total_ms']/baseline['total_ms'],
            compute_reduction_vs_original=1-native['compute_ms']/baseline['compute_ms'],
            within_guarded_total_reduction_vs_hostpad=1-native['total_ms']/timing['guarded']['pad32_discard_last']['total_ms']))
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.f32'))}
    summary = dict(scope='仅31行、三种真实权重的热算子；非完整prefill/状态/回答验收',
                   diagnostic_ops=len(records), route_counts=dict(collections.Counter(r['path'] for r in records)),
                   all_output_guards_passed=True, all_expected_output_bitwise_passed=True,
                   outputs_sha256=checksums, cases=cases)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# HTP设备内小M补齐原型', '',
             '外部输入输出始终为31行。DSP复用原有VTCM打包补零与有效行写回；没有在主机增加补行、复制或裁剪。', '',
             '| 矩阵 | 路径 | 原版总ms | 内部策略总ms | 总时间减少 | 计算时间减少 |',
             '|---|---|---:|---:|---:|---:|']
    for case in cases:
        lines.append(f"| {case['tensor']} | {'HMX内部补齐' if case['admitted'] else '保留HVX'} | {case['timing_ms']['baseline']['plain31']['total_ms']:.3f} | {case['timing_ms']['guarded']['plain31']['total_ms']:.3f} | {case['total_reduction_vs_original']:.2%} | {case['compute_reduction_vs_original']:.2%} |")
    lines += ['', '## 验证与限制', '',
              '原版与guarded各两个进程，ABBA顺序；各进程每矩阵31/32行各10次，5/5交替先后，全部保留。强制版本各10次作为FFN Down负对照。诊断独立运行，不混入计时。缓冲区复用，权重上传/重排、图分配、CPU参考和哨兵检查均在计时外。',
              '每次计算后输出尾部128个F32哨兵均保持不变；诊断66个MUL_MAT全部覆盖，QKV/Up的M31走HMX且mc32，Down的M31走HVX。',
              '内部HMX的31行输出与主机补到32行的前31行逐位相同；未选中的Down与原版逐位相同。选中路径相对原HVX仍有浮点差异，见summary.json，不能称数值不变。',
              '微基准使用固定合成激活和layer0真实Q4权重，仅支持本设备这两种形状、M31、Q4、非batch。没有验证其他M、其他层激活、完整模型状态或后续decode，不宣称端到端加速。进程间均值还受温度/频率等波动影响，未改手机设置。',
              '私有DSP与原版共享主机ABI；原运行库不覆盖。新小M路径若HMX执行失败则显式失败，不递归回退。', '',
              '构建：scripts/build/htp_small_m.py；设备验证：scripts/run/htp_small_m.py；分析：scripts/analyze/htp_small_m.py。']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'outputs_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
