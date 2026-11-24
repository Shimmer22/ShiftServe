#!/usr/bin/env python3
"""在最终封版上只回放 ac_ep_2405 的 ShiftRouter 与 ShiftServe。"""
import argparse
import fcntl
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.run import t6_mobile_gui_table1 as mobile
from scripts.run.frozen_ablation import read, save

CASE = 'ac_ep_2405'
ARMS = ('shared_m1', 'shiftserve_joint')
THREADS = {'cpu_threads': 4, 'cpu_threads_batch': 4, 'htp_host_threads': 4}


class FinalGuardCampaign(mobile.MobileCampaign):
    def service(self, action, arm, directory):
        super().service(action, arm, directory)
        if action == 'start':
            manifest = read(directory/'manifest.json')
            health = read(directory/'health.json')
            if (any(manifest.get(name) != value for name, value in THREADS.items())
                    or any(health.get(name) != THREADS[name] for name in ('cpu_threads', 'htp_host_threads'))
                    or not health.get('m1_explorer')):
                raise RuntimeError('正式线程口径或 M1 模式不匹配')


def result(path, summary):
    timings = [read(p)['response']['timings'] for p in sorted(path.glob('turn-*.json'))]
    if len(timings) != 9:
        raise RuntimeError('ac_ep_2405 必须含首轮和 8 个返回轮')
    target = timings[5]
    phases = summary['post_return_phases_ms']
    return dict(
        post_return_s=summary['post_return_ms']/1000,
        prefill_s=phases['prefill_including_prior']/1000,
        cc_turns=[i for i, t in enumerate(timings) if i and t['route']=='cpu'],
        compact_turns=[i for i, t in enumerate(timings) if t['kv_reduce']],
        peak_swap_kib=summary['memory_ledger_peak_swap']['swap_kib'],
        decode_major_faults=sum(t['decode_resources']['after']['major_faults']
                                - t['decode_resources']['before']['major_faults'] for t in timings),
        unexpected_callback_graphs=sum(t['graph_profile']['decode']['callback_graphs']
                                       + (t['graph_profile']['prefill']['callback_graphs'] if not t['prior_ms'] else 0)
                                       for t in timings),
        target_turn=dict(turn=5, route=target['route'], reason=target['route_reason'],
                         availability=target['availability'], H=target['H_effective'], P=target['P_evaluated'],
                         wait_ms=target['wait_ms'], pending_wait=target.get('pending_wait'),
                         prediction=target.get('prediction')),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    cli = parser.parse_args()
    build, output = cli.build.resolve(), cli.output.resolve()
    output.relative_to(ROOT/'results')
    mobile.SHA = cli.commit
    corpus, _ = mobile.fixture_plan(build)
    corpus['cases'] = [case for case in corpus['cases'] if case['id']==CASE]
    args = SimpleNamespace(build=build, output=output, fixtures=mobile.INPUT,
                           route_costs=mobile.PRIORS/'t5-route-priors.json', kv_costs=mobile.PRIORS/'t5-kv-priors.json',
                           model=ROOT/'configs/model-default.json',
                           cpu_registry=ROOT/'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so',
                           context_size=9472, gap_ms=0, verify_kv_state=0, kv_hwm_trim=0,
                           device='xiaomi14pro', memory_ledger=True, compact_anchor=1, kv_layer_stride=4,
                           kv_min_evict_tokens=256, conditions=list(ARMS), case=None,
                           force_path='none', single_turn=False, m1_explorer=True, graph_profile=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = FinalGuardCampaign(args, corpus)
        plan = dict(campaign.plan, threads=THREADS)
        if (output/'plan.json').exists() and read(output/'plan.json') != plan:
            raise RuntimeError('续跑须使用原封版与线程配置')
        save(output/'plan.json', plan)
        rows = {}
        try:
            for arm in ARMS:
                path = output/arm/CASE
                summary = campaign.run_cell(campaign.cases[0], arm, path, 'final-guard-smoke',
                                            audit=not (output/'renderer-audit.json').exists())
                rows[arm] = result(path, summary)
                save(output/'comparison.json', rows)
        except Exception as exc:
            campaign.state(status='failed', error=str(exc))
            raise
        checks = dict(router_turn_05_uses_htp=rows['shared_m1']['target_turn']['route']=='htp',
                      serve_compacts=bool(rows['shiftserve_joint']['compact_turns']))
        for arm, row in rows.items():
            checks[f'{arm}_resources'] = row['peak_swap_kib']==row['decode_major_faults']==row['unexpected_callback_graphs']==0
        passed = all(checks.values())
        save(output/'acceptance.json', dict(passed=passed, commit=cli.commit, threads=THREADS, checks=checks, rows=rows))
        campaign.state(status='passed' if passed else 'criteria_not_met', stage='complete')
        return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
