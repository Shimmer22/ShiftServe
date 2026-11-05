"""Tool 配置准入与跨轮生命周期验收。"""


def validate_tool_config(config, scenario, mechanism):
    if (config.get('mode') != 'hybrid' or config.get('kv_policy') != 'none'
            or config.get('pre_move') is not False or config.get('cpu_repack') is not False
            or config.get('timing_valid_for_performance') is not False):
        raise ValueError('tool session requires explicit hybrid, no compression/premove/repack, diagnostic timing')
    if config.get('htp_execution_prepare',False):
        if not config.get('execution_prepare_dir') or not 1<=config.get('predicted_input_tokens',64)<=min(64,config['batch_size']):
            raise ValueError('execution preparation requires private runtime and supported prediction shape')
        if not 0<=config.get('prefault_bytes',0)<=64*1024*1024:
            raise ValueError('execution prefault budget must be 0..64MiB')
    for t in scenario['turns']:
        if t['tool']['type'] not in {'device_display', 'http_get'}:
            raise ValueError('tool is not on the read-only allowlist')
        if not mechanism and (t.get('preparation') == 'hold' or t.get('wait_ready') or t.get('router_fixture')):
            raise ValueError('barriers/fixture costs require mechanism test mode')


def validate_lifecycle(events, turns, mechanism):
    def index(name, turn):
        return next((i for i, e in enumerate(events) if e['event'] == name and e['turn'] == turn), -1)
    checks = []
    for i, turn in enumerate(turns):
        order = [index(n, i) for n in ('tool_started', 'observation_ready', 'route_selected',
                                      'prefill_started', 'action_complete', 'cpu_state_committed', 'response_published')]
        ok = min(order) >= 0 and order == sorted(order) and len(set(order)) == len(order)
        ok &= turn['committed_version'] == i + 1 and turn['authority_checked']
        checks.append({'turn': i, 'passed': bool(ok)})
    cross_round = None
    if mechanism:
        # 本套件指定 turn1 持有旧任务、turn2 继续 CPU，之后才释放。
        held = index('barrier_entered', 1)
        pub1, pub2 = index('response_published', 1), index('response_published', 2)
        released = index('barrier_released', 3)
        cancelled = index('prepare_work_cancelled', 1)
        cross_round = (0 <= held < pub1 < index('tool_started', 2) < pub2 < released
                       and cancelled > pub2 and turns[1]['background_occupied_at_publish']
                       and turns[2]['background_occupied_at_publish']
                       and turns[1]['prefill_backend'] == turns[2]['prefill_backend'] == 'cpu'
                       and turns[2]['availability'] == 'busy'
                       and turns[3]['route_reason'] == 'cpu_lower_cost'
                       and turns[4]['route_reason'] == 'htp_lower_cost')
    closed = next((e for e in events if e['event'] == 'session_closed'), None)
    shutdown = (closed is not None and closed['started_tasks'] == closed['reaped_tasks'])
    return {'passed': bool(turns) and all(c['passed'] for c in checks) and cross_round is not False and shutdown,
            'turns': checks, 'cross_round_nonblocking': cross_round, 'shutdown_reaped_all': shutdown}
