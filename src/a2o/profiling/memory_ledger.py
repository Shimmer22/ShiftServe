"""解析端侧阶段账本，保留缺测状态并计算同进程差值。"""

import json
import re


MODEL_BUFFER = re.compile(r'load_tensors:\s+(\S+) model buffer size\s*=\s*([\d.]+) MiB')
PROCESS_KEYS = ('VmRSS_kib', 'Pss_kib', 'VmSwap_kib', 'minor_faults', 'major_faults')


def parse_runtime_log(lines):
    samples, buffers = [], []
    metadata = {}
    for line in lines:
        if line.startswith('A2O_MEMORY '):
            samples.append(json.loads(line.split(' ', 1)[1]))
        elif line.startswith('A2O_META '):
            metadata = json.loads(line.split(' ', 1)[1])
        else:
            match = MODEL_BUFFER.search(line)
            if match:
                buffers.append(dict(buffer=match[1], allocated_mib=float(match[2])))
    # Tool preparation runs on a worker. Its log line may be emitted after a later sample started.
    samples.sort(key=lambda entry: entry['sample']['monotonic_ms'])
    return metadata, samples, buffers


def validate_turn(timing, trim):
    ledger = timing['memory_ledger']
    if ledger['version'] != 1 or ledger['kv_hwm_trim'] is not trim:
        raise ValueError('逐轮账本版本或 kv_hwm_trim 配置不一致')
    checkpoints = {s['stage']: s for s in ledger['samples']}
    required = ['before_prefill', 'after_prefill', 'before_decode', 'after_decode']
    if timing['handoff_bytes']:
        required.append('handoff_peak')
    for stage in required:
        sample = checkpoints[stage]
        if not {'VmRSS_kib', 'VmSwap_kib', 'major_faults', 'minor_faults'} <= sample['process'].keys():
            raise ValueError(f'{stage} 缺少必需的进程内存计数')
        if not {'MemAvailable_kib', 'Slab_kib', 'SUnreclaim_kib'} <= sample['meminfo'].keys():
            raise ValueError(f'{stage} 缺少系统内存回退计数')
        if not ('Pss_kib' in sample['process'] or 'error' in sample['smaps_rollup']):
            raise ValueError(f'{stage} 缺少 PSS 和读取错误')
    return checkpoints


def summarize(samples):
    """以加载前为基线，保留所有阶段和相邻采样之间的 Swap 增长区间。"""
    if not samples or samples[0]['stage'] != 'before_model_load':
        raise ValueError('缺少加载前基线')
    baseline = samples[0]
    result = []
    previous = baseline
    for sample in samples:
        point = dict(stage=sample['stage'], monotonic_ms=sample['monotonic_ms'])
        process = sample['process']
        point.update({key: process.get(key) for key in PROCESS_KEYS})
        point['memavailable_drop_kib'] = _delta(baseline, sample, 'meminfo', 'MemAvailable_kib')
        point['pss_plus_identified_dma_kib'] = sample.get('pss_plus_identified_dma_kib')
        point['pss_growth_kib'] = _delta(sample, baseline, 'process', 'Pss_kib')
        point['dma_buf_identified_growth_bytes'] = _delta(
            sample, baseline, 'driver_buffers', 'dma_buf_identified_bytes')
        point['kernel_growth_kib'] = {
            field: _delta(sample, baseline, 'meminfo', field)
            for field in ('Slab_kib', 'SUnreclaim_kib')
        }
        point['since_previous'] = {
            key: _delta(sample, previous, 'process', key)
            for key in ('VmSwap_kib', 'minor_faults', 'major_faults')
        }
        result.append(point)
        previous = sample
    return result


def _delta(later, earlier, section, key):
    a = later[section].get(key)
    b = earlier[section].get(key)
    return a - b if a is not None and b is not None else None
