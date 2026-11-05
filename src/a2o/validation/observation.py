"""主图最小采样矩阵：全域后端扫描与代表条件的完整反事实重放。"""
def measurement_grid(config):
    if not config.get('observation_scan'):
        return [(p, g, s, r) for p in config['new_tokens'] for g in config['gap_ms']
                for s in config['strategies'] for r in (*config['routes'], 'router')]
    system = config['system_inputs']
    if not system or len(set(system)) != len(system) or not set(system) <= set(config['new_tokens']):
        raise ValueError('system_inputs 必须是扫描 P 的非空不重复子集')
    gap = config['scan_gap_ms']
    if gap not in config['gap_ms']:
        raise ValueError('scan_gap_ms 必须在 gap_ms 中')
    return [(p, g, s, r) for p in config['new_tokens'] for g in config['gap_ms']
            for s in config['strategies'] for r in (*config['routes'], 'router')
            if p in system or (g == gap and (s, r) in [('after_return', 'CC'), ('always', 'HC')])]
