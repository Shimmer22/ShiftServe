#!/usr/bin/env python3
"""Offline response and state-transfer audit. Never accesses a device."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.runner import ROOT, check_response


def main():
    rows = []
    history_check = json.loads((ROOT / 'scenarios/history.json').read_text())['check']
    for path in sorted((ROOT / 'results').glob('*/result.json')):
        result = json.loads(path.read_text())
        if result.get('session'):
            continue
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if 'scenario' not in manifest:
            continue
        recorded = result['response_check']
        # Explicitly rescore the connection-name fixture against its accepted
        # spellings, retaining the original strict four-letter check unchanged.
        check = manifest['scenario']['check']
        history_case = (manifest.get('scenario_file') == 'scenarios/history.json'
                        or check.get('value') == 'WIFI'
                        or check.get('values') == ['WIFI', 'WI-FI'])
        evaluation = check_response(result, {'check': history_check}) if history_case else recorded
        rows.append({
            'run': str(path.parent.relative_to(ROOT)),
            'mode': result['mode'], 'policy': result['kv_policy'], 'pre_move': result['pre_move'],
            'vision_provider': result.get('vision_provider', 'mtmd'),
            'format': result['format'], 'output': result['output'],
            'initial_check': recorded, 'evaluation': evaluation,
            'history_tokens': result['history_tokens'], 'removed_tokens': result['removed_tokens'],
            'transfer_checked': result['transfer_checked'],
            'upload_bytes': result['upload_bytes'], 'download_bytes': result['download_bytes'],
            'timing_ms': result['timing_ms'],
            'same_backend_roundtrip': 'same_backend_roundtrip_checked_ms' in result['timing_ms'],
            'matrix_case': history_case and not manifest['config'].get('roundtrip', False),
        })
    latest = {}
    for row in rows:
        if row['matrix_case']:
            latest[(row['mode'], row['policy'], row['pre_move'])] = row
    matrix = list(latest.values())
    summary = {'runs': rows, 'matrix': {
        'conditions': len(matrix),
        'semantic_passed': sum(r['evaluation']['passed'] for r in matrix),
        'original_strict_format_passed': sum(
            r['initial_check'].get('strict_format_passed', r['initial_check']['passed']) for r in matrix),
        'hybrid_transfer_passed': sum(r['transfer_checked'] for r in matrix if r['mode'] == 'hybrid'),
        'connection_check': history_check,
    }}
    (ROOT / 'results/validation-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary['matrix'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
