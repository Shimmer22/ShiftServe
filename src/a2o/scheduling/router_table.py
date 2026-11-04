"""从原始四路径记录导出精确 H/P/D 的 CC/HC 表；拒绝数值失败与不足样本。"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from ..validation.contracts import check_response


def export_table(run, output, decode_evaluations=4):
    run, output = Path(run), Path(output)
    if decode_evaluations <= 0:
        raise ValueError('D must be positive')
    manifest_bytes = (run / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    cfg = manifest['config']
    if (manifest.get('cpu_repack') is not False or manifest.get('kv_type') != 'F16'
            or manifest.get('llm_flash_attention') is not False):
        raise ValueError('unsupported calibration configuration')
    digest = hashlib.sha256(manifest_bytes)
    groups = defaultdict(list)
    for path in sorted(run.glob('H*.jsonl')):
        raw = path.read_bytes()
        digest.update(raw)
        for line in raw.decode().splitlines():
            r = json.loads(line)
            if r['home'] == 'cpu' and r['path'] in {'CC', 'HC'}:
                groups[(r['history_tokens'], r['new_tokens'], r['path'])].append(r)
    rows, rejected = [], []
    for (h, p, route), trials in sorted(groups.items()):
        valid = []
        for r in trials:
            ok = (r['eog'] and r['numerical_passed'] and r['generated_ids_equal']
                  and check_response(r, {'check': {'type': 'json_action', 'action': 'wait'}})['passed']
                  and len(r['decode_prefix_ms']) >= decode_evaluations)
            if ok:
                entry = sum(r[k] for k in ('entry_export_ms', 'entry_clear_ms', 'entry_import_ms'))
                future = r['prefill_ms'] + sum(r[k] for k in ('switch_export_ms', 'switch_clear_ms', 'switch_import_ms'))
                future += r['decode_prefix_ms'][decode_evaluations - 1]
                if all(math.isfinite(v) and v >= 0 for v in (entry, future)):
                    valid.append((future, entry))
        # 不从失败的同条件试验中仅挑成功样本填表。
        if len(valid) != len(trials) or len(valid) != cfg['repetitions'] or len(valid) < 3:
            rejected.append({'H': h, 'P': p, 'path': route, 'total': len(trials), 'valid': len(valid)})
            continue
        rows.append((h, p, decode_evaluations, route, median(v[0] for v in valid),
                     median(v[1] for v in valid), len(valid)))
    source = digest.hexdigest()
    header = '\t'.join(map(str, ['a2o-router-v1', source, cfg['context_size'], cfg['batch_size'],
                                 cfg['htp_queue'], cfg['htp_opbatch']]))
    output.write_text(header + '\n' + ''.join('\t'.join(map(str, r)) + '\n' for r in rows))
    metadata = {'source_run': str(run), 'source_sha256': source, 'rows': len(rows), 'rejected': rejected,
                'decode_evaluations': decode_evaluations,
                'scope': 'exact H/P; CPU authority and CPU decode; fixed D compute horizon, not complete-response prediction; no extrapolation',
                'config': cfg, 'calibration_binary_sha256': manifest['binary_sha256']}
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return metadata
