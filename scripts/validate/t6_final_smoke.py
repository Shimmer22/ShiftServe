#!/usr/bin/env python3
"""验证 Android 封版的首轮、当前输入保护、回调和延后快照。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.collect.tau2_e2e_worker import _agent_chat_resilient, _http_post_resilient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = output / f'phone-{time.time_ns()}'
    build = args.build.resolve()
    manifest = json.loads((build/'manifest.json').read_text())
    base = os.environ['A2O_PHONE_HTTP_BASE']
    model = json.loads((ROOT/'configs/model-default.json').read_text())
    case = next(c for c in json.loads(args.fixtures.read_text())['cases'] if c['id']=='multi_turn_base_123')
    service = [sys.executable, str(ROOT/'scripts/run/route_phone.py')]
    common = ['--condition', 'route', '--output', str(run), '--logs-only', '--retry-network']
    evidence = []
    try:
        subprocess.run(service + ['start'] + common + [
            '--runtime-overlay', str(build), '--shared-weights', '--cpu-repack', '0',
            '--model', model['model_path'], '--model-id', model['model_id'], '--model-sha256', model['model_sha256'],
            '--context-size', '9472', '--kv-reduce', '1', '--kv-policy', 'prefill_prior',
            '--kv-recent-tokens', '32', '--kv-keep-tokens', '2350', '--compact-anchor', '1',
            '--verify-kv-state', '0', '--memory-ledger', '--kv-hwm-trim', '0', '--gap-ms', '0',
        ], check=True)
        health = json.loads((run/'health.json').read_text())
        assert health['build_id']==manifest['build_id'] and health['git_commit']==manifest['git_commit']
        for i, turn in enumerate(case['turns'][:4]):
            if i:
                _http_post_resilient(base+'/a2o/tool-start', {})
            body = dict(turn['request'], frozen_graph_profile=True, max_tokens=turn['D'],
                        expected_decode_tokens=turn['D'])
            if i:
                body['frozen_force_path'] = 'hc_compact' if i == 3 else 'hc_full'
            response = _agent_chat_resilient(base, body, f'{run.name}:{i}')
            (run/f'turn-{i:02}.json').write_text(json.dumps(response, indent=2)+'\n')
            timing = response['timings']
            assert response['__verbose']['tokens']==turn['request']['frozen_decode_ids']
            assert (timing['H_anchor'], timing['H_effective'], timing['P_evaluated']) == (turn['H'], turn['H'], turn['P'])
            kv = timing['kv_reduce']
            trace = timing['graph_profile']
            assert trace['prefill']['callback_graphs']==int(i==3)
            assert trace['decode']['callback_graphs']==0
            if i == 3:
                assert kv['prior_ranked_tokens']>0 and timing['prior_ms']>0
                assert kv['recent_tokens']>=turn['P']>32
                assert kv['evicted_count']<=turn['H']-timing['pinned_tokens']
            else:
                assert kv is None and timing['prior_ms']==0
            if i:
                assert timing['route']=='htp' and timing['deferred_hc_anchor']
            ledger = timing['memory_ledger']['samples']
            peak_swap = max(s['process']['VmSwap_kib'] for s in ledger)
            assert peak_swap==0
            evidence.append(dict(turn=i, h=turn['H'], p=turn['P'], kv_reduce=kv,
                                 prefill_callback_graphs=trace['prefill']['callback_graphs'],
                                 decode_callback_graphs=trace['decode']['callback_graphs'],
                                 deferred_hc_anchor=timing['deferred_hc_anchor'], peak_swap_kib=peak_swap))
    finally:
        subprocess.run(service + ['stop'] + common, check=True)
    (output/'acceptance.json').write_text(json.dumps(dict(status='passed', build_id=manifest['build_id'],
                     git_commit=manifest['git_commit'], run=run.name, turns=evidence), indent=2)+'\n')


if __name__=='__main__':
    main()
