#!/usr/bin/env python3
"""串行执行 T6 的四轮图开销对照和 Telecom 交接内存对照。"""

import argparse
import fcntl
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run.frozen_ablation import (
    BASE, Campaign, _agent_chat_resilient, kv_cost_payload, read, request, save,
)

SOURCE = ROOT / 'results/mainline-eval/t6-8b-table1-revised'
CLEAN = {'send_message_with_contact_content_cellular_off': 3, 'multi_turn_base_123': 1}
TELECOM = 'telecom-52b4b12a5622-10'
SHORT_PAIRS = (('router', 'serve'), ('static', 'router'))
ARMS = {'static': 'shared_static_hc', 'router': 'shared_m1', 'serve': 'shiftserve_joint'}


def run_block(campaign, directory, case_lengths, *, mode='static', legacy=False,
              serial=False, prior_probe=False, force_hc=False):
    arm = ARMS[mode]
    cases = {c['id']: c for c in campaign.cases}
    status = campaign.output / 'status.json'
    try:
        save(status, dict(status='running', block=directory.name, stage='startup'))
        campaign.service('start', arm, directory)
        if mode != 'static':
            payload = read(campaign.args.route_costs)
            payload['m1_explorer'] = {'enabled': True}
            payload['kv_model'] = kv_cost_payload(campaign.args.kv_costs)
            save(directory/'costs-response.json', request('/a2o/costs', payload))
        elif serial:
            save(directory/'serial-context.json', {'enabled': True})

        summarized = []
        for name, last in case_lengths.items():
            dest = directory/name
            dest.mkdir()
            save(dest/'reset.json', request('/a2o/reset', {}))
            run_turns = last + (1 if prior_probe and name == 'multi_turn_base_123' else 0)
            evicted = 0
            for i, turn in enumerate(cases[name]['turns'][:run_turns+1]):
                save(status, dict(status='running', block=directory.name, stage='replay',
                                  case=name, turn=i, target_last=last))
                prep = request('/a2o/tool-start', {'frozen_serial_context': serial}, timeout=12) if i else None
                body = dict(turn['request'], max_tokens=turn['D'], expected_decode_tokens=turn['D'],
                            frozen_legacy_snapshot=legacy, frozen_graph_profile=True)
                smoke = i > last
                if smoke:
                    body.update(frozen_force_path='hc_compact', frozen_kv_keep_tokens=2200)
                elif i and (mode == 'static' or force_hc):
                    body['frozen_force_path'] = 'hc_full'
                rid = f'p0:{directory.name}:{name}:{i}'
                response = _agent_chat_resilient(BASE, body, rid, total_timeout=2400)
                save(dest/f'turn-{i:02}.json', dict(response=response, prepare=prep))
                timing = response['timings']
                profile = timing['graph_profile']
                assert response['__verbose']['tokens'] == turn['request']['frozen_decode_ids']
                assert response['__verbose']['stop_type'] == 'forced'
                assert (timing['H_anchor'], timing['H_effective'], timing['P_evaluated'], timing['predicted_n']) == (
                    turn['H'], turn['H']-evicted, turn['P'], turn['D'])
                assert timing['decode_backend'] == 'cpu' and profile['decode']['callback_graphs'] == 0
                if i and (mode == 'static' or force_hc):
                    assert timing['route'] == 'htp' and timing['handoff_bytes'] > 0
                    assert timing['deferred_hc_anchor'] is not legacy
                    assert timing['context_probe']['serial_context'] is serial
                kv = timing['kv_reduce']
                if kv and kv.get('compact_anchor') and i < run_turns:
                    evicted = cases[name]['turns'][i+1]['H'] - kv['anchor_live_tokens']
                if smoke:
                    assert timing['prior_ms'] > 0 and profile['prefill']['callback_graphs'] == 1
                    assert kv and kv['prior_ranked_tokens'] > 0
                elif i:
                    assert timing['prior_ms'] == 0 and profile['prefill']['callback_graphs'] == 0
                if i:
                    samples = timing['memory_ledger']['samples']
                    stage = {s['stage']: s for s in samples}
                    checkpoint = stage.get('handoff_peak')
                    decode = timing['decode_resources']
                    summarized.append(dict(
                        case=name, turn=i, prior_probe=smoke, route=timing['route'], reason=timing['route_reason'],
                        history=timing['H_effective'], prompt_tokens=timing['P_evaluated'], tokens=timing['predicted_n'],
                        compact=kv and kv['policy'], prefill_ms=timing['prompt_ms'], decode_ms=timing['predicted_ms'],
                        handoff_ms=timing['handoff_ms'], total_ms=timing['total_ms'],
                        context_probe=timing.get('context_probe'), graph_profile=profile,
                        peak_pss_dma_mib=max(s['pss_plus_identified_dma_kib'] for s in samples
                                             if s['pss_plus_identified_dma_kib'] is not None)/1024,
                        peak_swap_mib=max(s['process']['VmSwap_kib'] for s in samples)/1024,
                        swap_before_prefill_mib=stage['before_prefill']['process']['VmSwap_kib']/1024,
                        handoff_pss_dma_mib=checkpoint and checkpoint['pss_plus_identified_dma_kib']/1024,
                        handoff_swap_mib=checkpoint and checkpoint['process']['VmSwap_kib']/1024,
                        decode_swap_mib=[decode[edge]['VmSwap_kib']/1024 for edge in ('before','after')],
                        decode_major_faults=decode['after']['major_faults']-decode['before']['major_faults']))
    finally:
        campaign.service('stop', arm, directory)
    return summarized


def compare_short(blocks):
    pairs = {}
    for (a,b), repetitions in zip(SHORT_PAIRS, blocks):
        arms = {a: [], b: []}
        for mode, rows in zip((a,b,b,a), repetitions):
            arms[mode].append({(row['case'], row['turn']): row for row in rows if not row['prior_probe']})
        keys = set(arms[a][0]) & set(arms[b][0])
        valid, excluded = [], []
        for key in sorted(keys):
            rows = [rep[key] for mode in (a,b) for rep in arms[mode]]
            states = {(r['history'],r['prompt_tokens'],r['tokens']) for r in rows}
            if (len(states) != 1 or any(r['route'] != 'htp' or r['compact'] or
                    any(r['decode_swap_mib']) or r['decode_major_faults'] for r in rows)):
                excluded.append(list(key))
                continue
            valid.append(list(key))
        values = {}
        for mode in (a,b):
            cells = [rep[tuple(key)] for key in valid for rep in arms[mode]]
            if cells:
                values[mode] = dict(prefill_ms=statistics.mean(r['prefill_ms'] for r in cells),
                                    decode_ms_per_token=sum(r['decode_ms'] for r in cells)/sum(r['tokens'] for r in cells))
        pairs[f'{a}_vs_{b}'] = dict(valid=valid, excluded=excluded, values=values)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    out = cli.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(build=cli.build.resolve(), output=out, fixtures=SOURCE/'fixtures.json',
                           route_costs=SOURCE/'t5-route-priors.json', kv_costs=SOURCE/'t5-kv-priors.json',
                           model=ROOT/'configs/model-default.json',
                           cpu_registry=ROOT/'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so',
                           context_size=9472, gap_ms=0, verify_kv_state=0, kv_hwm_trim=0,
                           device='xiaomi14pro', memory_ledger=True, compact_anchor=1,
                           kv_layer_stride=4, kv_min_evict_tokens=256, conditions=list(ARMS.values()),
                           case=None, force_path='none', single_turn=False, m1_explorer=True)
    with (out/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign = Campaign(args, read(args.fixtures))
        plan = dict(campaign.plan, clean=CLEAN, short_pairs=[list(pair) for pair in SHORT_PAIRS], telecom=TELECOM,
                    memory_mode='router-forced-hc',
                    p1_threshold_mib=128, p1_trigger='peak Swap in either target return')
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise RuntimeError('P0 plan changed')
        save(out/'plan.json', plan)
        def block(name, cases, **options):
            result = out/f'{name}.json'
            if result.exists():
                return read(result)
            directory = out/f'{name}-{time.time_ns()}'
            rows = run_block(campaign, directory, cases, **options)
            save(result, rows)
            return rows

        try:
            blocks = []
            for group, (a,b) in enumerate(SHORT_PAIRS):
                blocks.append([block(f'{group}-{i:02}-{mode}', CLEAN, mode=mode,
                                     force_hc=(group==0), prior_probe=(group==0 and i==2))
                               for i, mode in enumerate((a,b,b,a))])
                save(out/'short-comparison.json', compare_short(blocks))
            deferred = block('telecom-deferred', {TELECOM: 4}, mode='router', force_hc=True)
            targets = [row for row in deferred if row['turn'] in (1,4)]
            added_swap = max(max(r['handoff_swap_mib'],r['decode_swap_mib'][0])-r['swap_before_prefill_mib']
                             for r in targets)
            peak_swap = max(r['peak_swap_mib'] for r in targets)
            outcome = dict(legacy=None, deferred=deferred, incremental_target_swap_mib=added_swap,
                           target_peak_swap_mib=peak_swap, serial_triggered=peak_swap>=plan['p1_threshold_mib'])
            save(out/'memory-comparison.json', outcome)
            if outcome['serial_triggered']:
                outcome['serial'] = block('telecom-serial', {TELECOM: 4}, mode='router', force_hc=True, serial=True)
                save(out/'memory-comparison.json', outcome)
            outcome['legacy'] = block('telecom-legacy', {TELECOM: 4}, mode='router', force_hc=True, legacy=True)
            save(out/'memory-comparison.json', outcome)
            save(out/'status.json', dict(status='passed', stage='complete',
                                        serial_triggered=outcome['serial_triggered']))
        except Exception as exc:
            status = out/'status.json'
            state = read(status) if status.exists() else {}
            save(status, dict(state, status='failed', error=str(exc)))
            raise


if __name__=='__main__':
    main()
