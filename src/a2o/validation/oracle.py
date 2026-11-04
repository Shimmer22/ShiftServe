"""真实 HTTP Oracle 原始记录完整性复核，不重写原始响应与数值门槛。"""
import hashlib
import json
from pathlib import Path
from statistics import median


def active_model_processes(process_table):
    """Android ps 可能只显示 basename，不能只查可执行文件的绝对路径。"""
    active = []
    for line in process_table.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        executable = fields[1].split(maxsplit=1)[0]
        name = executable.rsplit('/', 1)[-1]
        if name in ('agent', 'agent_runtime') or name.startswith('llama-') or executable.startswith('/data/local/tmp/a2o_'):
            active.append(line)
    return active


def verify_oracle_run(directory):
    directory = Path(directory)
    read = lambda name: [json.loads(line) for line in (directory / name).read_text().splitlines()]
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = manifest['config']; raw = read('device/trials.jsonl'); checked = read('checked-rows.jsonl')
    http = read('http-server.jsonl'); summary = json.loads((directory / 'summary.json').read_text())
    expected = 6 * len(cfg['new_tokens']) + len(cfg['new_tokens']) * len(cfg['gap_ms']) * 9 * cfg['repetitions']
    http_by_id = {h['id']: h for h in http}
    checks = {
        'complete_counts': len(raw) == len(checked) == len(http) == expected,
        'raw_rows_preserved': len(raw) == len(checked) and all(all(c[k] == v for k, v in r.items()) for c, r in zip(checked, raw)),
        'http_ids_complete': len(http_by_id) == expected and set(http_by_id) == set(range(expected)),
        'http_body_hashes': all(h['body_sha256'] == hashlib.sha256((directory / f'P{h["P"]}.txt').read_bytes()).hexdigest() for h in http),
        'http_conditions_match': all((http_by_id[r['id']]['P'], http_by_id[r['id']]['gap_requested_ms']) == (r['P'], r['gap_requested_ms']) for r in checked),
        'publication_order': summary['response_publication_lifecycle_checked'],
        'accepted_output_and_state': summary['accepted_response_and_state_passed'],
        'binary_hash': hashlib.sha256((directory / 'device/agent').read_bytes()).hexdigest() == manifest['binary_sha256'],
    }
    if cfg.get('input_fixture') == 'neutral_padding_v1':
        head = '\n<|im_start|>user\n'; tail = '\nWait.<|im_end|>\n<|im_start|>assistant\n'
        def valid_body(p):
            body = (directory / f'P{p}.txt').read_text()
            return (body.startswith(head) and body.endswith(tail)
                    and all(word == 'note' for word in body[len(head):-len(tail)].split()))
        checks['neutral_fixture_complete_words'] = all(valid_body(p) for p in cfg['new_tokens'])
    if not all(checks.values()):
        raise ValueError(checks)
    measurements = [r for r in checked if r['phase'] == 'measurement']
    early = [r for r in measurements if r['prepared_early']]
    stats = {'actual_gap_median_ms': {g: median(r['actual_gap_ms'] for r in measurements if r['gap_requested_ms'] == g) for g in cfg['gap_ms']},
        'early_preparation_trials': len(early), 'ready_at_return': sum(r['ready_at_return'] for r in early),
        'observed_overlap_trials': sum(min(r['prepare_completed_ms'], r['http_returned_ms']) > max(r['prepare_started_ms'], r['http_issued_ms']) for r in early),
        'prepare_ms_median': median(r['prepare_ms'] for r in early) if early else None}
    audit = {'passed': True, 'checks': checks, 'statistics': stats,
             'HC_qualified': summary['HC_numerically_qualified'], 'diagnostic_failures': summary['diagnostic_failures']}
    (directory / 'final-checks.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    return audit
