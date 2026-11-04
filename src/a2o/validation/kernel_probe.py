"""解析私有DSP探针路由标签；该运行的pmu字段不是硬件PMU计数。"""
import re

MAGIC = 0xA20F0000
PATHS = {1: 'HVX', 2: 'HMX', 3: 'SKIP'}
REASONS = {0: 'hmx_success', 1: 'hmx_disabled', 2: 'N_not_aligned_32', 3: 'weight_type_unsupported',
           4: 'quant_K_not_aligned_256', 5: 'f16_K_not_aligned_32', 6: 'quant_batched',
           7: 'nonmonotonic_strides', 8: 'M_lt_32', 9: 'hmx_error', 10: 'hmx_not_compiled', 11: 'skip_compute'}
PATTERN = re.compile(r'profile-op\s+MUL_MAT:\s+(?P<names>.*?)\s+:\s+(?P<dims>.*?)\s+:\s+(?P<types>.*?)\s+:\s+.*?\s+usec\s+(?P<usec>\d+)\s+cycles\s+(?P<cycles>\d+)\s+pmu\s+\[(?P<pmu>[\d,\s]+)\]')


def decode_route(values):
    if len(values) != 8 or values[0] & 0xFFFF0000 != MAGIC:
        return None
    path, reason = values[0] & 0xFF, values[0] >> 8 & 0xFF
    if path not in PATHS or reason not in REASONS:
        raise ValueError('未知路由探针编码')
    ret = values[5] if values[5] < 2**31 else values[5] - 2**32
    return {'path': PATHS[path], 'reason': REASONS[reason], 'M': values[1], 'N': values[2], 'K': values[3],
            'dsp_weight_type': values[4], 'hmx_return_code': ret, 'source0_batches': values[6], 'source1_batches': values[7]}


def parse_route_line(line):
    match = PATTERN.search(line)
    if not match: return None
    values = [int(v.strip()) for v in match['pmu'].split(',')]
    route = decode_route(values)
    if route is None: return None
    return {**route, 'names': match['names'], 'host_dims': match['dims'], 'host_types': match['types'],
            'usec': int(match['usec']), 'cycles': int(match['cycles'])}
