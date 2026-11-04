#!/usr/bin/env python3
"""离线汇总 NPU–CPU 路径；保留条件与诊断追踪开销的区别。"""
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['hybrid-cpu-vision', 'hybrid-opencl-demand', 'hybrid-htp-probe',
         'hybrid-htp-demand', 'hybrid-htp-a2o', 'hybrid-htp-fused-probe']
rows = []
for name in NAMES:
    candidates = []
    for manifest in sorted((ROOT / 'results').glob('*/manifest.json')):
        data = json.loads(manifest.read_text())
        if data.get('config_file') == 'configs/' + name + '.json':
            candidates.append((manifest.parent, data))
    if not candidates:
        continue
    path, manifest = candidates[-1]
    row = {'condition': name, 'run': str(path.relative_to(ROOT)),
           'source_sha256': manifest['source_sha256'], 'config': manifest['config']}
    result = path / 'result.json'
    if result.exists():
        r = json.loads(result.read_text())
        row.update(passed=r['response_check']['passed'], generated_ids=r['generated_ids'],
                   timing_ms=r['timing_ms'], budget_adjusted_response_ms=r['budget_adjusted_response_ms'])
    else:
        row['passed'] = False
        row['failure'] = json.loads((path / 'failure.json').read_text()) if (path / 'failure.json').exists() else 'incomplete'
    if (path / 'runtime.log').exists():
        counts = Counter()
        for line in (path / 'runtime.log').read_text().splitlines():
            m = re.match(r'VISION_NODE (\S+) .* buffer=(\S+) shape=', line)
            if m:
                counts[m.groups()] += 1
        if counts:
            row['vision_nodes'] = [{'op': op, 'buffer': buf, 'count': n}
                                   for (op, buf), n in sorted(counts.items())]
    rows.append(row)
report = {'note': '单次固定截图回放；trace_vision 会引入诊断开销。节点数不代表 FLOPs 或耗时占比。A2O 为模拟等待，模型上下文初始化在等待前完成。',
          'rows': rows}
target = ROOT / 'results/npu-cpu-summary.json'
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for row in rows:
    t = row.get('timing_ms', {})
    print(row['condition'], 'passed=', row['passed'],
          {k: round(t[k], 2) for k in ['vision_init_ms', 'vision_encode_ms', 'llm_prefill_ms',
           'decode_ms', 'observation_to_response_ms', 'prep_overrun_ms'] if k in t})
print(target)
