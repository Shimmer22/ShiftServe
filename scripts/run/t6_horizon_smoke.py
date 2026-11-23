#!/usr/bin/env python3
"""在 590 对 Prefill 完成后，定向验收两条 Mobile GUI 轨迹。"""
import argparse
import csv
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.run import t6_mobile_gui_table1 as mobile
from scripts.run.frozen_ablation import read, save

PREFILL = ROOT / 'results/mainline-eval/prefill-motivation-h4096-8b'
OUTPUT = ROOT / 'results/mainline-eval/t6-horizon-mobile-smoke'
ORDERS = {
    'ac_ep_6119': ('shared_static_hc', 'shiftserve_joint', 'shared_m1', 'shifttrim'),
    'ac_ep_2405': ('shifttrim', 'shared_m1', 'shiftserve_joint', 'shared_static_hc'),
}
RETURN_TARGETS = {'ac_ep_6119': 6, 'ac_ep_2405': 5}
COMPACT_TARGETS = {'ac_ep_6119': (4, 5), 'ac_ep_2405': (3, 5)}


def timing(path, turn):
    return read(path / f'turn-{turn:02d}.json')['response']['timings']


def evaluate():
    rows, targeted, checks, speedups = [], [], {}, {}
    for case, arms in ORDERS.items():
        post_ms = {}
        for arm in arms:
            path = OUTPUT / arm / case
            summary = read(path / 'summary.json')
            phases = summary['post_return_phases_ms']
            turns = range(summary['logical_rounds'])
            records = [timing(path, t) for t in turns]
            post_ms[arm] = summary['post_return_ms']
            major_faults = sum(t['decode_resources']['after']['major_faults'] -
                               t['decode_resources']['before']['major_faults'] for t in records)
            invalid_callbacks = sum(t['graph_profile']['prefill']['callback_graphs'] for t in records
                                    if not t['prior_ms'])
            invalid_callbacks += sum(t['graph_profile']['decode']['callback_graphs'] for t in records)
            swap = summary['memory_ledger_peak_swap']['swap_kib']
            rows.append(dict(case=case, arm=arm, returns=summary['warm_rounds'],
                             post_return_s=post_ms[arm] / 1000, prefill_s=phases['prefill_including_prior'] / 1000,
                             prep_wait_s=phases['prep_wait'] / 1000, cc=sum(t['route'] == 'cpu' for t in records[1:]),
                             compact_turns=' '.join(str(i) for i, t in enumerate(records) if t['kv_reduce']),
                             peak_swap_kib=swap, decode_major_faults=major_faults,
                             unexpected_callback_graphs=invalid_callbacks))
            checks[f'{case}/{arm}/resources'] = swap == major_faults == invalid_callbacks == 0
            if arm == 'shiftserve_joint':
                checks[f'{case}/compact_on_formerly_blocked_turn'] = any(records[i]['kv_reduce'] for i in COMPACT_TARGETS[case])
            turn = RETURN_TARGETS[case]
            t = records[turn]
            targeted.append(dict(case=case, arm=arm, turn=turn, route=t['route'], reason=t['route_reason'],
                                 wait_ms=t['wait_ms'], pending_wait=t.get('pending_wait'),
                                 H=t['H_effective'], P_evaluated=t['P_evaluated']))
            if arm in ('shared_m1', 'shiftserve_joint'):
                checks[f'{case}/{arm}/target_htp'] = t['route'] == 'htp'
                checks[f'{case}/{arm}/target_wait_approx_0p3_to_0p5_s'] = 0.3 <= round(t['wait_ms'] / 1000, 1) <= 0.5
        checks[f'{case}/serve_le_trim'] = post_ms['shiftserve_joint'] <= post_ms['shifttrim']
        # Record the actual ratio and test the direction without inventing a percentage threshold.
        checks[f'{case}/serve_lt_static'] = post_ms['shiftserve_joint'] < post_ms['shared_static_hc']
        speedups[case] = dict(vs_trim=post_ms['shifttrim']/post_ms['shiftserve_joint'],
                              vs_static_hc=post_ms['shared_static_hc']/post_ms['shiftserve_joint'])
    with (OUTPUT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(OUTPUT / 'acceptance.json', dict(passed=all(checks.values()), checks=checks,
                                         target_turns=targeted, shiftserve_speedups=speedups))
    return all(checks.values())


def run(build, commit):
    mobile.SHA = commit
    corpus, max_length = mobile.fixture_plan(build)
    corpus['cases'] = [case for case in corpus['cases'] if case['id'] in ORDERS]
    args = SimpleNamespace(build=build, output=OUTPUT, fixtures=mobile.INPUT,
                           route_costs=mobile.PRIORS/'t5-route-priors.json', kv_costs=mobile.PRIORS/'t5-kv-priors.json',
                           model=ROOT/'configs/model-default.json',
                           cpu_registry=ROOT/'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so',
                           context_size=9472, gap_ms=0, verify_kv_state=0, kv_hwm_trim=0,
                           device='xiaomi14pro', memory_ledger=True, compact_anchor=1, kv_layer_stride=4,
                           kv_min_evict_tokens=256, conditions=list(ORDERS['ac_ep_6119']), case=None,
                           force_path='none', single_turn=False, m1_explorer=True, graph_profile=True)
    campaign = mobile.MobileCampaign(args, corpus)
    plan = dict(campaign.plan, order={k:list(v) for k,v in ORDERS.items()}, max_fixture_logical_tokens=max_length)
    plan_path = OUTPUT/'plan.json'
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError('冒烟续跑需要同一构建、夹具和试验顺序')
    save(plan_path, plan)
    cases = {c['id']: c for c in campaign.cases}
    try:
        # Reverse the condition order on the second trajectory. Only one cell owns the phone at a time.
        for case, arms in ORDERS.items():
            for arm in arms:
                campaign.run_cell(cases[case], arm, OUTPUT/arm/case, 'horizon-smoke',
                                  audit=not (OUTPUT/'renderer-audit.json').exists())
        passed = evaluate()
        campaign.state(status='passed' if passed else 'criteria_not_met', stage='complete')
        return 0 if passed else 2
    except Exception as exc:
        campaign.state(status='failed', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run', action='store_true', help=argparse.SUPPRESS)
    cli = parser.parse_args()
    if cli.run:
        return run(cli.build.resolve(), cli.commit)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mobile.SHA = cli.commit
        mobile.fixture_plan(cli.build.resolve())
        save(OUTPUT/'queue.json', dict(status='waiting_for_590_pairs', commit=cli.commit))
        with (PREFILL/'queue.lock').open('r') as upstream:
            while True:
                try:
                    fcntl.flock(upstream, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(30)
            state = read(PREFILL/'status.json')
            if state.get('phase') != 'complete' or state.get('completed') != 590 or state.get('total') != 590:
                save(OUTPUT/'queue.json', dict(status='upstream_incomplete', upstream=state))
                return 1
            save(OUTPUT/'queue.json', dict(status='starting_smoke', upstream=state))
            return subprocess.call([sys.executable, '-u', 'scripts/run/termux_http_forward.py', '--port', '28792', '--',
                                    sys.executable, '-u', str(Path(__file__).resolve()), '--run',
                                    '--commit', cli.commit, '--build', str(cli.build.resolve())], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
