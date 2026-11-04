"""配置、响应、状态与旧异步夹具的检查契约。"""
import json


def validate(config):
    if config['mode'] not in {'cpu', 'npu', 'hybrid'}:
        raise ValueError('mode must be cpu, npu, hybrid')
    if config['kv_policy'] not in {'none', 'recent'}:
        raise ValueError('kv_policy must be none, recent')
    if config['pre_move'] and config['mode'] != 'hybrid':
        raise ValueError('pre_move requires hybrid mode')
    if config.get('format', 'chat') not in {'raw', 'chat'}:
        raise ValueError('format must be raw, chat')
    if config.get('vision_backend', 'auto') not in {'auto', 'cpu', 'opencl', 'htp'}:
        raise ValueError('vision_backend must be auto, cpu, opencl, htp')
    if config.get('vision_prepare', 'arrival') not in {'arrival', 'a2o'}:
        raise ValueError('vision_prepare must be arrival, a2o')
    if not 1 <= config.get('batch_size', 512) <= 512:
        raise ValueError('batch_size must be within [1,512]')
    if config.get('vision_flash', 'auto') not in {'auto', 'off', 'on'}:
        raise ValueError('vision_flash must be auto, off, on')
    if config.get('htp_queue', 16) < 1:
        raise ValueError('htp_queue must be positive')
    if config.get('async_smoke', False):
        if (config['mode'] != 'hybrid' or config['kv_policy'] != 'none' or config['pre_move']
                or config.get('compare_prefill', True)):
            raise ValueError('async smoke requires hybrid, no compression/premove/paired prefill')
        cases = config.get('preparation_cases', [])
        if not cases or cases[0] != 'none' or any(c not in {'none', 'ready', 'early', 'cancel'} for c in cases):
            raise ValueError('invalid async preparation cases')
        if (len(config.get('prefill_routes', [])) != len(cases)
                or config.get('decode_routes') != ['cpu'] * len(cases)
                or any(r != 'cpu' for r in config['prefill_routes'])):
            raise ValueError('async smoke uses CPU authority and readiness-selected prefill')


def check_async_preparation(result, cases):
    """只验事件顺序与状态契约，不以耗时阈值判断机制。"""
    checks = []
    turns = result.get('turns', [])
    for i, (turn, case) in enumerate(zip(turns, cases)):
        meta = turn.get('async_preparation', {})
        events = [e['event'] for e in meta.get('events', [])]
        def before(a, b):
            return a in events and b in events and events.index(a) < events.index(b)
        ok = (meta.get('case') == case and meta.get('snapshot_version') == i
              and meta.get('committed_version') == i + 1 and turn.get('decode_backend') == 'cpu')
        if case == 'none':
            ok &= meta.get('status') == 'disabled' and turn.get('prefill_backend') == 'cpu'
        else:
            reason = {'ready': 'ready', 'early': 'pending', 'cancel': 'cancelled'}[case]
            ok &= (meta.get('authority_checked') is True and meta.get('reason') == reason
                   and turn.get('prefill_backend') == ('htp' if case == 'ready' else 'cpu')
                   and before('observation_arrived', 'bind_' + reason)
                   and before('action_complete', 'cpu_state_committed')
                   and before('cpu_state_committed', 'drain_started_after_response'))
            if case == 'ready':
                ok &= meta.get('status') == 'ready' and before('prepare_ready', 'bind_ready')
            elif case == 'early':
                ok &= (meta.get('status') == 'ready' and meta.get('stale_rejected') is True
                       and before('bind_pending', 'prepare_started')
                       and before('cpu_state_committed', 'bind_stale'))
            else:
                ok &= meta.get('status') == 'cancelled' and 'prepare_started' not in events
        checks.append({'turn': i, 'case': case, 'passed': bool(ok)})
    return {'passed': (len(turns) == len(cases) and all(c['passed'] for c in checks)
                       and result.get('timing_valid_for_performance') is False), 'turns': checks}


def check_response(result, scenario):
    if 'turns' in scenario:
        turns = result.get('turns', [])
        checks = [check_response(r, s) for r, s in zip(turns, scenario['turns'])]
        responses_ok = len(turns) == len(scenario['turns']) and all(c['passed'] for c in checks)
        state_ok = all(r['prefix_match'] and r['restore_checked'] and r['eog_committed'] for r in turns)
        image_ok = (result['image_encode_calls'] == sum(bool(t.get('image')) for t in scenario['turns'])
                    and all(r['image_encode_calls'] == int(bool(s.get('image'))) for r, s in zip(turns, scenario['turns'])))
        numeric_ok = all(not r['compared'] or (r['logits_cosine'] >= .999 and r['logits_nrmse'] <= .05
                                               and r['logits_top1_equal']) for r in turns)
        compared = sum(r['compared'] for r in turns)
        return {'passed': responses_ok and state_ok and image_ok and numeric_ok,
                'outputs_passed': responses_ok, 'state_passed': state_ok,
                'single_encode_passed': image_ok, 'numerical_passed': numeric_ok if compared else None,
                'numerical_compared_turns': compared, 'turns': checks}
    text = result['output'].strip()
    check = scenario['check']
    if check['type'] == 'exact':
        return {'passed': text == check['value'], 'expected': check['value']}
    if check['type'] == 'nonempty':
        return {'passed': bool(text)}
    if check['type'] == 'one_of':
        return {'passed': text in check['values'], 'expected_values': check['values'],
                'strict_format_passed': text == check['strict_value']}
    if check['type'] == 'json_action':
        fenced = text.startswith('```json\n') and text.endswith('\n```')
        payload = text[len('```json\n'):-len('\n```')] if fenced else text
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            return {'passed': False, 'reason': str(exc)}
        ok = (isinstance(obj, dict) and obj.get('action') == check['action']
              and all(key in obj for key in check.get('keys', []))
              and all(obj.get(key) == value for key, value in check.get('equals', {}).items()))
        result = {'passed': ok, 'json_action_passed': ok, 'parsed': obj,
                  'response_format': 'json_code_fence' if fenced else 'json',
                  'strict_format_passed': not fenced}
        if 'screen_contains' in check:
            screen = obj.get('screen') if isinstance(obj, dict) else None
            screen_ok = (isinstance(screen, str)
                         and any(name.casefold() in screen.casefold() for name in check['screen_contains']))
            result['screen_passed'] = screen_ok
            result['passed'] = ok and screen_ok
        return result
    raise ValueError('unknown response check')
