"""只生成合成测试输入，不读取、复制或上传 results/ 实验数据。"""
import json


def router_calibration(root):
    root.mkdir()
    manifest = {'execution': 'synthetic', 'cpu_repack': False, 'kv_type': 'F16',
                'llm_flash_attention': False, 'binary_sha256': 'synthetic-test-only',
                'config': {'repetitions': 3, 'context_size': 5120, 'batch_size': 128,
                           'htp_queue': 1, 'htp_opbatch': 1}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    for h in (1024, 4096):
        rows = []
        for p in (16, 64):
            for route in ('CC', 'HC', 'HH', 'CH'):
                for repetition in range(3):
                    rows.append({'home': 'cpu', 'path': route, 'history_tokens': h,
                                 'new_tokens': p, 'repetition': repetition, 'eog': True,
                                 'numerical_passed': not (h == 1024 and route == 'HC'),
                                 'generated_ids_equal': True, 'output': '{"action":"wait"}',
                                 'entry_export_ms': 10 if route == 'HC' else 0,
                                 'entry_clear_ms': 0, 'entry_import_ms': 40 if route == 'HC' else 0,
                                 'prefill_ms': 190 if route == 'HC' else (100 if p == 16 else 300),
                                 'switch_export_ms': 5 if route == 'HC' else 0,
                                 'switch_clear_ms': 0, 'switch_import_ms': 0,
                                 'decode_prefix_ms': [1, 2, 3, 5]})
        (root / f'H{h}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return root


def replay_fixtures(root):
    data = {
        'single': ({'config': {}, 'scenario': {'check': {'type': 'nonempty'}}},
                   {'response_check': {'passed': True}, 'mode': 'cpu', 'kv_policy': 'none',
                    'pre_move': False, 'format': 'chat', 'output': 'synthetic', 'history_tokens': 0,
                    'removed_tokens': 0, 'transfer_checked': False, 'upload_bytes': 0,
                    'download_bytes': 0, 'timing_ms': {}}),
        'multi': ({'config': {}, 'scenario': {'turns': [{'prompt': 'synthetic'}]}},
                  {'session': True, 'response_check': {'passed': True}, 'image_encode_calls': 0,
                   'turns': [{'turn': 0, 'compared': False, 'prefill_backend': 'cpu',
                              'cached_tokens': 0, 'image_encode_calls': 0}]}),
        'tool': ({'config': {}, 'scenario': {'turns': [{'tool': {'type': 'http_get'}}]}},
                 {'session': True, 'turns': []})}
    for name, (manifest, result) in data.items():
        target = root / 'results' / name
        target.mkdir(parents=True)
        (target / 'manifest.json').write_text(json.dumps(manifest))
        (target / 'result.json').write_text(json.dumps(result))
