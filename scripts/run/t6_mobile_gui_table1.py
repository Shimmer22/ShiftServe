#!/usr/bin/env python3
"""按轨迹轮转采集 Android 封版的 Mobile GUI Table 1。"""

import argparse
from collections import Counter
import csv
import fcntl
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run.frozen_ablation import Campaign, COST_ARMS, digest, kv_cost_payload, read, request, save
from scripts.run.t6_table1_summary import TABLE1_ARMS, write_table1
from a2o.profiling.memory_ledger import parse_runtime_log

SHA = '05b88107d89c7253d33f91603c878ffe348c0593'
FIXTURE_SHA = '0269a614ae252bcfab7a3404b2800d092a6a024c64995267f2d48f7122f553c3'
INPUT = ROOT/'results/mainline-eval/mobile-gui-hpd/fixtures.json'
PRIORS = ROOT/'results/mainline-eval/t6-8b-table1-revised'
GATE = 'ac_ep_2405'
ALTERNATE_GATE = 'ac_ep_6119'
ARMS = tuple(TABLE1_ARMS)


def fixture_plan(build):
    manifest = read(build/'manifest.json')
    if manifest['git_commit'] != SHA or manifest['dirty']:
        raise ValueError('Android build is not the requested clean commit')
    if any(digest(build/name) != checksum for name, checksum in manifest['artifacts'].items()):
        raise ValueError('Android artifact checksum mismatch')
    if digest(INPUT) != FIXTURE_SHA:
        raise ValueError('fixture checksum mismatch')
    corpus = read(INPUT)
    cases = corpus['cases']
    counts = Counter(c['benchmark'] for c in cases)
    if counts != {'droidtask': 3, 'androidcontrol_low': 5} or len({c['id'] for c in cases}) != 8:
        raise ValueError('unexpected Mobile GUI cohort')
    max_length, warm_rounds = 0, 0
    for case in cases:
        previous = []
        for turn in case['turns']:
            body = turn['request']
            ids, decode = body['frozen_prompt_ids'], body['frozen_decode_ids']
            if (turn['H'], turn['P'], turn['D']) != (len(previous), len(ids)-len(previous), len(decode)):
                raise ValueError(f'invalid frozen sizes {case["id"]}')
            if ids[:len(previous)] != previous or ids[-4:] != [151667,271,151668,271]:
                raise ValueError(f'invalid frozen anchor {case["id"]}')
            previous = ids[:-4]
            max_length = max(max_length,len(ids)+len(decode))
        warm_rounds += len(case['turns'])-1
    if warm_rounds != 50 or max_length > 7892:
        raise ValueError(f'unexpected fixture size: {warm_rounds} returns, {max_length} tokens')
    return corpus, max_length


def balanced_order(cases):
    # Even-order Williams square: the first six rows balance positions and direct carryover.
    shifts = {'droidtask': iter((0,2,4)), 'androidcontrol_low': iter((1,3,5,0,2))}
    return [
        dict(case=case['id'], benchmark=case['benchmark'],
             order=[ARMS[(i+shift)%len(ARMS)] for i in (0,1,5,2,4,3)])
        for case in cases for shift in [next(shifts[case['benchmark']])]
    ]


class MobileCampaign(Campaign):
    def replay_turns(self, arm, case, path, turns, routed, force_path):
        for i, record in enumerate(super().replay_turns(arm,case,path,turns,routed,force_path)):
            t = record['response']['timings']
            profile, kv = t['graph_profile'], t['kv_reduce']
            prior = bool(kv and kv['policy']=='prefill_prior' and kv['prior_ranked_tokens']>0)
            clocks = profile['clocks']
            expected_clocks = ('before_prefill','after_prefill','before_decode','after_decode')
            if any(not clocks[k]['thermal_sysfs_raw'] or not any(v is not None for v in clocks[k]['cpu_policy_khz'].values())
                   for k in expected_clocks):
                raise RuntimeError('missing per-stage temperature or CPU frequency')
            if (profile['decode']['callback_graphs'] != 0 or profile['prefill']['callback_graphs'] != int(prior)
                    or (t['prior_ms']>0) != prior or (not i and kv)):
                raise RuntimeError('callback or cold compaction contract failed')
            if kv and (not kv['compacted'] or kv['recent_tokens'] < t['P_evaluated'] or t['route'] != 'htp'
                       or kv['policy'] not in ('prefill_prior','recent')):
                raise RuntimeError('compaction did not preserve the current input')
            if i and arm not in ('full_cpu','full_htp'):
                actual = 'cc' if t['route']=='cpu' else 'hc_compact' if kv else 'hc_full'
                selected = (t['prediction'] or {}).get('selected_path')
                if t['route'] not in ('cpu','htp') or bool(t['handoff_bytes']) != (t['route']=='htp'):
                    raise RuntimeError('unexpected backend or handoff')
                if selected and selected != actual:
                    raise RuntimeError(f'mismatched selected path {selected} vs {actual}')
                if t['route']=='htp' and not t['deferred_hc_anchor']:
                    raise RuntimeError('shared HC did not defer its new anchor')
            yield record

    def run_cell(self, case, arm, target, stage, audit=False):
        if (target/'summary.json').is_file():
            return read(target/'summary.json')
        attempt = self.output/'services'/f'{stage}-{case["id"]}-{arm}-{time.time_ns()}'
        raw = attempt/'capture'
        self.state(status='running',stage=f'{stage}_startup',case=case['id'],condition=arm,turn=None)
        try:
            self.service('start', arm, attempt)
            raw.mkdir()
            health = read(attempt/'health.json')
            if health['git_commit'] != SHA or health['gap_ms'] != 0 or health['verify_kv_state'] or health['kv_hwm_trim']:
                raise RuntimeError('phone health does not match sealed measurement')
            if audit:
                cleared = {c['id']: self.verify_renderer(c) for c in self.cases}
                save(self.output/'renderer-audit.json',cleared)
                if any(cleared.values()):
                    raise RuntimeError('phone renderer disagrees with frozen fixtures')
            if arm in COST_ARMS:
                payload = read(self.args.route_costs)
                payload['kv_model'] = kv_cost_payload(self.args.kv_costs)
                if arm in ('shared_m1','shiftserve_joint'):
                    payload['m1_explorer'] = {'enabled': True}
                save(attempt/'cost-request.json',payload)
                save(attempt/'cost-response.json',request('/a2o/costs',payload))
            self.replay(arm,case,raw)
        finally:
            self.service('stop',arm,attempt)
        return self.finalize_cell(attempt,target)

    @staticmethod
    def finalize_cell(attempt,target):
        raw = attempt/'capture'
        row = read(raw/'summary.json')
        phases = row['post_return_phases_ms']
        if not math.isclose(math.fsum(v for k,v in phases.items() if k!='prior'),row['post_return_ms'],abs_tol=1e-6):
            raise RuntimeError('phase mismatch')
        with (attempt/'device/runtime.log').open() as stream:
            _, ledger, _ = parse_runtime_log(stream)
        samples = [entry['sample'] for entry in ledger]
        physical = [s for s in samples if s.get('pss_plus_identified_dma_kib') is not None]
        if not physical:
            raise RuntimeError('no physical memory ledger samples')
        peak = max(physical,key=lambda s:s['pss_plus_identified_dma_kib'])
        row['memory_ledger_peak_pss_plus_dma'] = dict(stage=peak['stage'],kib=peak['pss_plus_identified_dma_kib'])
        peak = max(samples,key=lambda s:s['process']['VmSwap_kib'])
        row['memory_ledger_peak_swap'] = dict(stage=peak['stage'],swap_kib=peak['process']['VmSwap_kib'])
        row['memory_ledger_missing_pss_plus_dma_samples'] = len(samples)-len(physical)
        save(raw/'summary.json',row)
        save(raw/'service.json',{'directory': str(attempt.relative_to(attempt.parent.parent))})
        target.parent.mkdir(parents=True,exist_ok=True)
        raw.rename(target)
        return row


def write_details(output, order):
    rows = []
    for scope in ('overall','droidtask','androidcontrol_low'):
        cases = [c['case'] for c in order if scope=='overall' or c['benchmark']==scope]
        summaries, peaks = [], {}
        detail = {}
        for arm in ARMS:
            summaries_arm = []
            waits = Counter()
            for case in cases:
                d = output/arm/case
                if not (d/'summary.json').is_file():
                    continue
                summaries_arm.append(read(d/'summary.json'))
                for file in sorted(d.glob('turn-*.json'))[1:]:
                    t = read(file)['response']['timings']
                    kv = t['kv_reduce'] or {}
                    path = ('HH' if arm=='full_htp' else 'CC' if t['route']=='cpu'
                            else 'HC-Compact' if kv else 'HC-Full')
                    waits[path] += 1
                    waits['prior_rounds'] += int(kv.get('policy')=='prefill_prior')
                    waits['recent_rounds'] += int(kv.get('policy')=='recent')
                    waits['newly_evicted_tokens'] += kv.get('evicted_count',0)
                    logical = t['H_anchor'] + t['P_evaluated']
                    waits['logical_tokens'] += logical
                    waits['logical_tokens_removed'] += t['H_anchor']-t['H_effective']+kv.get('evicted_count',0)
            summaries.extend(summaries_arm)
            peaks[arm] = dict(
                peak_pss_plus_dma_mib=max((r['memory_ledger_peak_pss_plus_dma']['kib']/1024 for r in summaries_arm
                                          if 'memory_ledger_peak_pss_plus_dma' in r),default=None),
                peak_vmswap_mib=max((r['memory_ledger_peak_swap']['swap_kib']/1024 for r in summaries_arm),default=None),
                missing_pss_plus_dma_samples=sum(r['memory_ledger_missing_pss_plus_dma_samples'] for r in summaries_arm))
            detail[arm] = waits
        destination = output if scope=='overall' else output/scope
        destination.mkdir(exist_ok=True)
        write_table1(destination,cases,summaries,peaks)
        for result, arm in zip(csv.DictReader((destination/'table1.csv').open()),ARMS):
            counts = detail[arm]
            result.update(scope=scope, **{k:counts[k] for k in (
                'CC','HC-Full','HC-Compact','HH','prior_rounds','recent_rounds','newly_evicted_tokens','logical_tokens')})
            result['cumulative_eviction_ratio'] = (counts['logical_tokens_removed']/counts['logical_tokens']
                                                   if counts['logical_tokens'] else None)
            rows.append(result)
    pending=output/'table1-breakdown.tmp'
    with pending.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(output/'table1-breakdown.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args()
    out, build=cli.output.resolve(),cli.build.resolve()
    if out != (ROOT/'results/mainline-eval/t6-8b-table1-mobile-gui').resolve():
        raise ValueError('use the dedicated Mobile GUI results directory')
    corpus, max_length=fixture_plan(build)
    out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(build=build,output=out,fixtures=INPUT,
                         route_costs=PRIORS/'t5-route-priors.json',kv_costs=PRIORS/'t5-kv-priors.json',
                         model=ROOT/'configs/model-default.json',
                         cpu_registry=ROOT/'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so',
                         context_size=9472,gap_ms=0,verify_kv_state=0,kv_hwm_trim=0,device='xiaomi14pro',
                         memory_ledger=True,compact_anchor=1,kv_layer_stride=4,kv_min_evict_tokens=256,
                         conditions=list(ARMS),case=None,force_path='none',single_turn=False,
                         m1_explorer=True,graph_profile=True)
    with (out/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=MobileCampaign(args,corpus)
        cases={c['id']:c for c in campaign.cases}
        order=balanced_order(corpus['cases'])
        plan=dict(campaign.plan,order=order,max_logical_tokens=max_length,gate=GATE,graph_profile=True,
                  adaptive_state='reset per trajectory; online within trajectory')
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise RuntimeError('resume requires the original Mobile GUI plan')
        save(out/'plan.json',plan)
        try:
            if not (out/'gate.json').exists():
                rejected={}
                for candidate in (GATE,ALTERNATE_GATE):
                    gate={}
                    for arm in ('shared_static_hc','shiftserve_joint'):
                        row=campaign.run_cell(cases[candidate],arm,out/'gate'/arm/candidate,'gate',
                                              audit=not (out/'renderer-audit.json').exists())
                        gate[arm]=row
                        if row['memory_ledger_peak_swap']['swap_kib'] != 0:
                            break
                    if (len(gate)==2 and gate['shiftserve_joint']['memory_ledger_peak_swap']['swap_kib']==0
                            and gate['shiftserve_joint']['prior_ranked_tokens']>0):
                        save(out/'gate.json',{'status':'passed','case':candidate,'arms':gate})
                        break
                    rejected[candidate]=gate
                    save(out/'gate-failure.json',rejected)
                else:
                    raise RuntimeError('neither gate candidate met zero Swap and positive ShiftServe prior ranking')
            for cell in order:
                case=cell['case']
                for arm in cell['order']:
                    campaign.run_cell(cases[case],arm,out/arm/case,'formal')
                    finished=list(out.glob('*/*/summary.json'))
                    campaign.state(stage='cell_complete',formal_cells=len(finished),
                                   completed=dict(Counter(p.parent.parent.name for p in finished)))
                write_details(out,order)
            campaign.state(status='passed',stage='complete',formal_cells=48)
        except Exception as exc:
            campaign.state(status='failed',error=str(exc))
            raise


if __name__=='__main__':
    main()
