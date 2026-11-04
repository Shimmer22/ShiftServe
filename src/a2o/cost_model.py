"""第一版查表决策：只隐藏输入到达前可完成的准备，不隐藏 prefill/return。"""
def route_times(row, gap_ms):
    if gap_ms < 0:
        raise ValueError('gap_ms must be nonnegative')
    h = row['htp']
    prepare = h['state_export_ms'] + h['state_clear_ms'] + h['state_import_ms']
    # There is no separate tiling operation in the current canonical F16 path.
    prepare += h.get('tiling_ms') or 0
    returned = h['return_export_ms'] + h['return_clear_ms'] + h['return_import_ms']
    cpu = row['cpu']['prefill_ms']
    htp = max(0., prepare-gap_ms) + h['prefill_ms'] + returned
    return {'cpu_ms': cpu, 'htp_ms': htp, 'prepare_ms': prepare,
            'return_ms': returned, 'choice': 'htp' if htp < cpu else 'cpu'}


def path_cost(row, gap_ms, decode_tokens=None):
    """四路径在当前驻留状态上的成本；Gap只隐藏历史准备，不隐藏阶段间交接。"""
    if gap_ms < 0:
        raise ValueError('gap_ms must be nonnegative')
    trace = row['decode_prefix_ms']
    if decode_tokens is not None and not 1 <= decode_tokens <= len(trace):
        raise ValueError('decode prefix not measured')
    entry = sum(row[k] for k in ['entry_export_ms', 'entry_clear_ms', 'entry_import_ms'])
    switch = sum(row[k] for k in ['switch_export_ms', 'switch_clear_ms', 'switch_import_ms'])
    decode = trace[-1] if decode_tokens is None else trace[decode_tokens-1]
    return {'total_ms': max(0., entry-gap_ms)+row['prefill_ms']+switch+decode,
            'prepare_ms': entry, 'phase_switch_ms': switch, 'decode_ms': decode,
            'end_backend': 'cpu' if row['path'][1] == 'C' else 'htp'}
