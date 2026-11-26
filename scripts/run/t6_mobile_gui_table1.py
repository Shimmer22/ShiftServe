#!/usr/bin/env python3
"""按轨迹轮转采集 Android 封版的 Mobile GUI Table 1。"""

import argparse
from collections import Counter
import csv
import fcntl
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run.frozen_ablation import Campaign, COST_ARMS, CPU_REPLAY_ARMS, STATIC_HC_ARMS, digest, kv_cost_payload, read, request, save
from scripts.run.t6_table1_summary import TABLE1_ARMS, TABLE1_EIGHT_ARMS, TABLE1_SEVEN_ARMS, write_table1
from a2o.profiling.memory_ledger import parse_runtime_log
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network

SHA = '7e3dd64eed08e1fa07387e722b3aea352fbbff51'
FIXTURE_SHA = '0269a614ae252bcfab7a3404b2800d092a6a024c64995267f2d48f7122f553c3'
INPUT = ROOT/'results/mainline-eval/mobile-gui-hpd/fixtures.json'
PRIORS = ROOT/'results/mainline-eval/t6-8b-table1-revised'
DEFAULT_ROUTE_COSTS = ROOT/'configs/qwen3-8b-route-costs.json'
GATE = 'ac_ep_2405'
ALTERNATE_GATE = 'ac_ep_6119'
ARMS = tuple(TABLE1_ARMS)


def fixture_plan(build, expected_sha=None, fixtures_path=None, allow_dirty_build=False):
    fpath = (fixtures_path or INPUT).resolve()
    manifest = read(build/'manifest.json')
    target_sha = expected_sha or SHA
    if manifest['git_commit'] != target_sha or (manifest['dirty'] and not allow_dirty_build):
        raise ValueError(f'Android build {manifest["git_commit"]} != {target_sha} or dirty={manifest["dirty"]}')
    if any(digest(build/name) != checksum for name, checksum in manifest['artifacts'].items()):
        raise ValueError('Android artifact checksum mismatch')
    if fpath == INPUT.resolve() and digest(fpath) != FIXTURE_SHA:
        raise ValueError('fixture checksum mismatch')
    corpus = read(fpath)
    cases = corpus['cases']
    if fpath == INPUT.resolve():
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
    if fpath == INPUT.resolve() and (warm_rounds != 50 or max_length > 7892):
        raise ValueError(f'unexpected fixture size: {warm_rounds} returns, {max_length} tokens')
    if max_length > 9472:
        raise ValueError(f'fixture max_length={max_length} > 9472')
    return corpus, max_length


def balanced_order(cases, arms=ARMS):
    # Balanced Williams / Latin-square rotation per benchmark to cancel slow drift across arms.
    if len(arms) == 6 and len(cases) == 8:
        shifts = {'droidtask': iter((0,2,4)), 'androidcontrol_low': iter((1,3,5,0,2))}
        return [
            dict(case=case['id'], benchmark=case['benchmark'],
                 order=[arms[(i+shift)%len(arms)] for i in (0,1,5,2,4,3)])
            for case in cases for shift in [next(shifts[case['benchmark']])]
        ]
    base = (
        (0, 1, 7, 2, 6, 3, 5, 4) if len(arms) == 8
        else (0, 1, 6, 2, 5, 3, 4) if len(arms) == 7
        else tuple(range(len(arms)))
    )
    bench_idx = Counter()
    rows = []
    for idx, case in enumerate(cases):
        b = case['benchmark']
        shift = (bench_idx[b] * 2 + (1 if b == 'androidcontrol_low' else 0)) % len(arms)
        bench_idx[b] += 1
        seq = base if idx % 2 == 0 else tuple(reversed(base))
        rows.append(dict(case=case['id'], benchmark=b,
                         order=[arms[(i+shift)%len(arms)] for i in seq]))
    return rows


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
                if arm not in STATIC_HC_ARMS and t['route']=='htp' and not t['deferred_hc_anchor']:
                    raise RuntimeError('shared HC did not defer its new anchor')
                if arm in STATIC_HC_ARMS and (t['deferred_hc_anchor'] or t['availability'] != 'absent'):
                    raise RuntimeError('static HC unexpectedly used deferred anchor or background prep')
            yield record

    def cleanup_failed_phone_cell(self, attempt_name):
        transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
        q = shlex.quote
        remote_service = f'{transport.remote_root}/{attempt_name}/service'
        lock_dir = f'{transport.remote_root}/model.lock'
        cmd = (
            'for sig in TERM KILL; do '
            'pids=$(/system/bin/ps -A -o PID,ARGS | awk -v pat=' + q(remote_service) +
            ' \'$0 ~ pat {print $1}\'); '
            '[ -z "$pids" ] && break; '
            'for p in $pids; do kill -$sig "$p" 2>/dev/null || true; done; '
            'sleep 1; '
            'done; '
            'PATH=/data/data/com.termux/files/usr/bin:"$PATH" termux-wake-unlock >/dev/null 2>&1 || true; '
            'if [ -d ' + q(lock_dir) + ' ]; then '
            'owner=$(cat ' + q(lock_dir + '/owner') + ' 2>/dev/null || true); '
            'alive=$(/system/bin/ps -A -o PID,ARGS | grep -F "/.local/a2o/runs/" | grep -F "/service " || true); '
            'if [ "$owner" = ' + q(attempt_name) + ' ] || [ -z "$alive" ]; then '
            'rm -rf ' + q(lock_dir) + '; fi; '
            'fi'
        )
        retry_network(lambda: transport.call('shell', cmd), report=lambda msg: print(msg, flush=True))
        time.sleep(3)

    def record_cell_failure(self, case, arm, target, attempt, exc):
        raw = attempt / 'capture'
        turns_done = sorted(raw.glob('turn-*.json')) if raw.exists() else []
        last_turn = len(turns_done) - 1 if turns_done else None
        fail_archive = self.output / 'failures' / arm / case['id']
        if fail_archive.exists():
            shutil.rmtree(fail_archive)
        fail_archive.parent.mkdir(parents=True, exist_ok=True)
        if raw.exists() and any(raw.iterdir()):
            shutil.copytree(raw, fail_archive / 'capture')
        if (attempt / 'device').exists():
            shutil.copytree(attempt / 'device', fail_archive / 'device')
        target.mkdir(parents=True, exist_ok=True)
        failure_info = {
            'status': 'failed',
            'arm': arm,
            'case': case['id'],
            'benchmark': case.get('benchmark'),
            'error': str(exc),
            'completed_turns': len(turns_done),
            'last_turn_index': last_turn,
            'attempt_dir': str(attempt.relative_to(self.output)),
            'archive_dir': str(fail_archive.relative_to(self.output)),
            'failed_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        }
        save(target / 'FAILED.json', failure_info)
        ledger_path = self.output / 'failed-cells.json'
        existing = read(ledger_path) if ledger_path.exists() else []
        existing = [x for x in existing if not (x.get('arm') == arm and x.get('case') == case['id'])]
        existing.append(failure_info)
        save(ledger_path, existing)
        print(f'[CELL FAILED - CONTINUING] arm={arm} case={case["id"]} err={exc}', flush=True)
        return None

    def run_cell(self, case, arm, target, stage, audit=False):
        if (target/'summary.json').is_file():
            return read(target/'summary.json')
        continue_on_error = bool(getattr(self.args, 'continue_on_cell_error', False))
        if continue_on_error and (target/'FAILED.json').is_file():
            return None
        attempt = self.output/'services'/f'{stage}-{case["id"]}-{arm}-{time.time_ns()}'
        raw = attempt/'capture'
        self.state(status='running',stage=f'{stage}_startup',case=case['id'],condition=arm,turn=None)
        try:
            try:
                self.service('start', arm, attempt)
                raw.mkdir()
                health = read(attempt/'health.json')
                expected_threads = getattr(self.args, 'cpu_threads', None)
                if (health['git_commit'] != self.git_commit or health['gap_ms'] != 0 or health['verify_kv_state'] or health['kv_hwm_trim']
                        or (expected_threads is not None and health.get('cpu_threads') != expected_threads)
                        or health.get('htp_host_threads', 4) != 4):
                    raise RuntimeError('phone health does not match sealed measurement')
                if audit:
                    cleared = {c['id']: self.verify_renderer(c) for c in self.cases}
                    save(self.output/'renderer-audit.json',cleared)
                    if any(cleared.values()):
                        raise RuntimeError('phone renderer disagrees with frozen fixtures')
                if arm in COST_ARMS or (getattr(self.args, 'cold_costs', None) is not None and arm not in CPU_REPLAY_ARMS and arm not in STATIC_HC_ARMS and arm != 'full_htp'):
                    payload = read(self.args.route_costs) if arm in COST_ARMS else {}
                    if getattr(self.args, 'cold_costs', None) is not None:
                        payload['cold_model'] = read(self.args.cold_costs)
                    if arm in COST_ARMS:
                        payload['kv_model'] = kv_cost_payload(self.args.kv_costs)
                    if arm in ('shared_m1','shiftserve_joint'):
                        payload['m1_explorer'] = {'enabled': True}
                    save(attempt/'cost-request.json',payload)
                    save(attempt/'cost-response.json',request('/a2o/costs',payload))
                self.replay(arm,case,raw)
            finally:
                try:
                    self.service('stop',arm,attempt)
                except Exception as stop_exc:
                    if not continue_on_error:
                        raise
                    print(f'[STOP CLEANUP FALLBACK] arm={arm} case={case["id"]} err={stop_exc}', flush=True)
                    self.cleanup_failed_phone_cell(attempt.name)
            return self.finalize_cell(attempt,target)
        except Exception as exc:
            if not continue_on_error:
                raise
            self.cleanup_failed_phone_cell(attempt.name)
            return self.record_cell_failure(case, arm, target, attempt, exc)

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


def write_details(output, order, arms=ARMS):
    if tuple(arms) == tuple(TABLE1_EIGHT_ARMS):
        arm_labels = TABLE1_EIGHT_ARMS
    elif tuple(arms) == tuple(TABLE1_SEVEN_ARMS):
        arm_labels = TABLE1_SEVEN_ARMS
    elif tuple(arms) == tuple(TABLE1_ARMS):
        arm_labels = TABLE1_ARMS
    else:
        arm_labels = {arm: TABLE1_EIGHT_ARMS.get(arm, arm) for arm in arms}
    rows = []
    scopes = ('overall', *sorted({c['benchmark'] for c in order}))
    for scope in scopes:
        cases = [c['case'] for c in order if scope=='overall' or c['benchmark']==scope]
        summaries, peaks = [], {}
        detail = {}
        for arm in arms:
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
        write_table1(destination,cases,summaries,peaks,arms=arm_labels)
        for result, arm in zip(csv.DictReader((destination/'table1.csv').open()),arms):
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
    parser.add_argument('--fixtures',type=Path,default=INPUT)
    parser.add_argument('--commit',default=None)
    parser.add_argument('--route-costs',type=Path,default=DEFAULT_ROUTE_COSTS)
    parser.add_argument('--cold-costs',type=Path,default=None)
    parser.add_argument('--kv-costs',type=Path,default=PRIORS/'t5-kv-priors.json')
    parser.add_argument('--opbatch',type=int,default=1024)
    parser.add_argument('--opqueue',type=int,default=16)
    parser.add_argument('--htp-big-cores',type=int,choices=[0,1],default=0)
    parser.add_argument('--cases',nargs='+',default=None)
    parser.add_argument('--allow-dirty-build',action='store_true')
    parser.add_argument('--cpu-threads',type=int,default=None,choices=range(1,17))
    parser.add_argument('--include-shared-cpu',action='store_true',default=True)
    parser.add_argument('--six-arms',action='store_true',help='仅跑6组（不含 shared_cc）')
    parser.add_argument('--eight-arms',action='store_true',help='跑8组（含 Vanilla-Static-HC 与 Shared-Static-HC）')
    parser.add_argument('--arms',nargs='+',default=None,help='显式指定要运行的 arm 列表')
    parser.add_argument('--max-quiet-c',type=float,default=39.0)
    parser.add_argument('--min-prime-khz',type=int,default=2000000)
    parser.add_argument('--min-mem-avail-kib',type=int,default=1500000,help='模型加载后可用内存门槛（开服前固定为 4000000 KiB）')
    parser.add_argument('--cooldown-timeout-s',type=int,default=360)
    parser.add_argument('--skip-gate',action='store_true',help='跳过 AC-Low 双案前置 gate（补跑/容错队列用）')
    parser.add_argument('--continue-on-cell-error',action='store_true',help='单个 cell 异常（如 OOM/LMKD）时清理现场并记录 FAILED.json 后继续')
    cli=parser.parse_args()
    out, build=cli.output.resolve(),cli.build.resolve()
    fixtures_path = cli.fixtures.resolve()
    out.relative_to(ROOT/'results')
    if out == (ROOT/'results/mainline-eval/t6-8b-table1-mobile-gui').resolve():
        raise ValueError('keep historical t6-8b-table1-mobile-gui untouched; choose a new output directory')
    manifest_commit = cli.commit or read(build/'manifest.json')['git_commit']
    corpus, max_length=fixture_plan(build, expected_sha=manifest_commit, fixtures_path=fixtures_path, allow_dirty_build=cli.allow_dirty_build)
    out.mkdir(parents=True,exist_ok=True)
    route_dst = out/'route-costs.json'
    route = read(route_dst if route_dst.exists() else cli.route_costs.resolve())
    route_threads = route.get('model',{}).get('cpu_threads')
    cpu_threads = cli.cpu_threads if cli.cpu_threads is not None else (int(route_threads) if route_threads is not None else 4)
    if route_threads is not None and int(route_threads) != cpu_threads:
        raise ValueError(f'route_costs cpu_threads={route_threads} 与 --cpu-threads={cpu_threads} 不一致')
    if not route_dst.exists():
        route['model']['routing_p_min'] = 16
        save(route_dst, route)
    if cli.arms:
        arms = list(cli.arms)
    elif cli.eight_arms:
        arms = list(TABLE1_EIGHT_ARMS)
    else:
        arms = list(TABLE1_ARMS if cli.six_arms else TABLE1_SEVEN_ARMS)
    args=SimpleNamespace(build=build,output=out,fixtures=fixtures_path,
                         route_costs=route_dst,kv_costs=cli.kv_costs.resolve(),
                         model=ROOT/'configs/model-default.json',
                         cpu_registry=ROOT/'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so',
                         context_size=9472,gap_ms=0,verify_kv_state=0,kv_hwm_trim=0,device='xiaomi14pro',
                         memory_ledger=True,compact_anchor=1,kv_layer_stride=4,kv_min_evict_tokens=256,
                         conditions=arms,case=cli.cases,force_path='none',single_turn=False,
                         m1_explorer=True,graph_profile=True,cpu_threads=cpu_threads,
                         opbatch=cli.opbatch,opqueue=cli.opqueue,htp_big_cores=cli.htp_big_cores,
                         cold_costs=(cli.cold_costs.resolve() if cli.cold_costs else None),
                         max_quiet_c=cli.max_quiet_c,min_prime_khz=cli.min_prime_khz,
                         min_mem_avail_kib=cli.min_mem_avail_kib,cooldown_timeout_s=cli.cooldown_timeout_s,
                         continue_on_cell_error=cli.continue_on_cell_error)
    with (out/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=MobileCampaign(args,corpus)
        cases={c['id']:c for c in campaign.cases}
        order=balanced_order(campaign.cases if cli.cases else corpus['cases'], arms=arms)
        has_ac_gate = (not cli.skip_gate) and all(g in cases for g in (GATE, ALTERNATE_GATE))
        plan=dict(campaign.plan,order=order,max_logical_tokens=max_length,gate=(GATE if has_ac_gate else None),graph_profile=True,
                  adaptive_state='reset per trajectory; online within trajectory')
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise RuntimeError('resume requires the original Mobile GUI plan')
        save(out/'plan.json',plan)
        try:
            if has_ac_gate and not (out/'gate.json').exists():
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
                    campaign.run_cell(cases[case],arm,out/arm/case,'formal',
                                      audit=not (out/'renderer-audit.json').exists())
                    finished=[p for p in out.glob('*/*/summary.json') if p.parent.parent.name in arms]
                    failed_cells=[p for p in out.glob('*/*/FAILED.json') if p.parent.parent.name in arms]
                    campaign.state(stage='cell_complete',formal_cells=len(finished),
                                   failed_cells=len(failed_cells),
                                   completed=dict(Counter(p.parent.parent.name for p in finished)),
                                   failed=dict(Counter(p.parent.parent.name for p in failed_cells)))
                write_details(out,order,arms=arms)
            finished=[p for p in out.glob('*/*/summary.json') if p.parent.parent.name in arms]
            failed_cells=[p for p in out.glob('*/*/FAILED.json') if p.parent.parent.name in arms]
            campaign.state(status=('passed_with_cell_failures' if failed_cells else 'passed'),
                           stage='complete',formal_cells=len(finished),failed_cells=len(failed_cells))
        except Exception as exc:
            campaign.state(status='failed',error=str(exc))
            raise


if __name__=='__main__':
    main()
