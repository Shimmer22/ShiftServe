#!/usr/bin/env python3
"""Offline audit of screenshot correctness, transfers, and memory pressure."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.environment import ROOT
from a2o.validation.contracts import check_response


def main():
    rows = []
    for manifest_path in sorted((ROOT / 'results').glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('scenario', {}).get('image'):
            continue
        folder = manifest_path.parent
        row = {'run': str(folder.relative_to(ROOT)), 'config': manifest['config'],
               'scenario_file': manifest.get('scenario_file'), 'device': manifest['device']}
        if (folder / 'aborted.json').exists():
            row.update(status='aborted', details=json.loads((folder / 'aborted.json').read_text()))
        elif (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
            evaluation = check_response(result, manifest['scenario'])
            r = result.get('resources', {})
            start, end = r.get('decode_start', {}), r.get('decode_end', {})
            row.update(status='complete', output=result['output'], initial_check=result['response_check'],
                       vision_provider=result.get('vision_provider', 'mtmd'),
                       evaluation=evaluation, timing_ms=result['timing_ms'],
                       transfer_checked=result['transfer_checked'],
                       upload_bytes=result['upload_bytes'], download_bytes=result['download_bytes'],
                       history_tokens=result['history_tokens'], removed_tokens=result['removed_tokens'],
                       decode_major_faults=end.get('major_faults', 0)-start.get('major_faults', 0),
                       decode_start_swap_kib=start.get('VmSwap_kib'),
                       decode_end_swap_kib=end.get('VmSwap_kib'),
                       process_wall_ms=result['process_wall_ms'])
        else:
            row['status'] = 'incomplete'
        rows.append(row)
    (ROOT / 'results/vision-summary.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for row in rows:
        print(row['run'], row['status'], row.get('evaluation', {}),
              'decode faults=', row.get('decode_major_faults'))


if __name__ == '__main__':
    main()
