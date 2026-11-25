#!/usr/bin/env python3
"""冻结 ToolSandbox 回放，区分 HTP 上下文重建和线程交接。"""

import argparse
import fcntl
from pathlib import Path
import shlex
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from a2o.runtime.device_transport import DeviceTransport
from a2o.validation.oracle import active_model_processes
from scripts.run.frozen_ablation import BASE, Campaign, _agent_chat_resilient, read, request, save

CASE = 'find_current_city_low_battery_mode'
ORDERS = (
    ('rebuild_async', 'rebuild_sync', 'retain'),
    ('rebuild_sync', 'retain', 'rebuild_async'),
    ('retain', 'rebuild_async', 'rebuild_sync'),
)


def archive_stale_lock(transport, owner, output):
    processes = transport.call('shell', '/system/bin/ps -A -o PID,ARGS').stdout.decode()
    (output / 'phone-processes-before.txt').write_text(processes)
    if active_model_processes(processes) or any(
            '/runs/' in line and '/service ' in line for line in processes.splitlines()):
        raise RuntimeError('phone still has a model process')
    if not owner:
        return
    lock = transport.remote_root + '/model.lock'
    old_pid = transport.remote_root + '/' + owner + '/pid'
    archive = lock + '.stale-' + output.name
    q = shlex.quote
    command = (
        f'if [ ! -d {q(lock)} ]; then exit 0; fi; '
        f'test "$(cat {q(lock + "/owner")})" = {q(owner)} || exit 1; '
        f'p=$(cat {q(old_pid)}) || exit 1; '
        'case "$p" in ""|*[!0-9]*) exit 1;; esac; '
        f'test "$p" -gt 1 && test ! -d "/proc/$p" && test ! -e {q(archive)} && '
        f'mv {q(lock)} {q(archive)} && printf "%s\\n" {q(archive)}'
    )
    moved = transport.call('shell', command).stdout.decode().strip()
    save(output / 'stale-lock.json', {'owner': owner, 'archive': moved or None})


def run_replay(output, case, mode, label):
    dest = output / label
    dest.mkdir()
    save(dest / 'reset.json', request('/a2o/reset', {}, timeout=30, retries=1))
    rows = []
    for i, turn in enumerate(case['turns'][:3]):
        body = dict(turn['request'], max_tokens=turn['D'], frozen_graph_profile=True)
        prep = None
        probe_mode = ('retain_htp_decode' if i == 1 else 'retain') if mode == 'previous_htp' else mode
        if i and mode != 'full_htp':
            if probe_mode != 'rebuild_sync':
                prep = request('/a2o/tool-start', {}, timeout=20, retries=1)
                if not prep['started']:
                    raise RuntimeError('HTP preparation failed to start')
            body.update(frozen_force_path='hc_full', frozen_hc_lifecycle_probe=probe_mode)
        response = _agent_chat_resilient(BASE, body, f'htp-hc-probe:{label}:{i}', total_timeout=180)
        save(dest / f'turn-{i:02}.json', {'prepare': prep, 'response': response})
        t = response['timings']
        if (t['H_anchor'], t['H_effective'], t['P_evaluated'], t['predicted_n']) != (
                turn['H'], turn['H'], turn['P'], turn['D']):
            raise RuntimeError('frozen shape differs')
        if response['__verbose']['tokens'] != turn['request']['frozen_decode_ids']:
            raise RuntimeError('frozen decode differs')
        if t['decode_backend'] != ('htp' if mode == 'full_htp' or probe_mode == 'retain_htp_decode' else 'cpu'):
            raise RuntimeError('unexpected decode backend')
        if not i:
            continue
        if t['route'] != 'htp' or bool(t['handoff_bytes']) != (mode != 'full_htp'):
            raise RuntimeError('unexpected prefill backend or handoff')
        graph = t['graph_profile']['prefill']
        if graph['callback_graphs'] or t['kv_reduce']:
            raise RuntimeError('unexpected compaction/callback')
        if mode != 'full_htp' and t['hc_lifecycle_probe']['retained_after_handoff'] != probe_mode.startswith('retain'):
            raise RuntimeError('unexpected HTP context lifetime')
        print(f'{label} turn {i}: dispatch {graph["dispatch_us"]/1e6:.3f} s', flush=True)
        rows.append({'label': label, 'mode': mode, 'turn': i,
                     'dispatch_s': graph['dispatch_us']/1e6, 'prefill_s': t['prompt_ms']/1000,
                     'restore_s': t['restore_ms']/1000, 'prepare_s': t['prepare_ms']/1000,
                     'wait_s': t['wait_ms']/1000})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stale-owner')
    parser.add_argument('--previous-decode', action='store_true', help='固定驻留和交接，只改变前一轮的 Decode 后端')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixtures = ROOT / 'results/20261124-toolsandbox-10cases/fixtures.json'
    corpus = read(fixtures)
    case = next(c for c in corpus['cases'] if c['id'] == CASE)
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    settings = SimpleNamespace(
        build=args.build.resolve(), output=out, fixtures=fixtures, case=[CASE], model=ROOT / 'configs/model-default.json',
        route_costs=None, kv_costs=None, conditions=['full_htp', 'shared_static_hc'],
        cpu_registry=ROOT / 'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so',
        context_size=9472, gap_ms=0, verify_kv_state=0, kv_hwm_trim=0, device='xiaomi14pro',
        memory_ledger=True, compact_anchor=0, kv_layer_stride=4, kv_min_evict_tokens=256,
        force_path='none', single_turn=False, m1_explorer=False, cpu_threads=6,
    )
    campaign = Campaign(settings, corpus)
    orders = (('retain', 'previous_htp', 'previous_htp', 'retain'),) if args.previous_decode else ORDERS
    hc = [(f'block-{i}-{j}-{mode}', mode)
          for i, order in enumerate(orders, 1) for j, mode in enumerate(order, 1)]
    phases = [('shared', 'shared_static_hc', hc)]
    if not args.previous_decode:
        phases.insert(0, ('htp-before', 'full_htp', [('htp-before', 'full_htp')]))
        phases.append(('htp-after', 'full_htp', [('htp-after', 'full_htp')]))
    save(out / 'plan.json', dict(campaign.plan, case=CASE, orders=orders, compare_turn=2, http_base=BASE))
    rows = []
    with (ROOT / 'results/.termux-model.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive_stale_lock(transport, args.stale_owner, out)
        for label, arm, blocks in phases:
            service = out / f'hc-lifecycle-{label}-{time.time_ns()}'
            save(out / 'status.json', {'status': 'running', 'service': service.name})
            try:
                campaign.service('start', arm, service)
                for run_label, mode in blocks:
                    save(out / 'status.json', {'status': 'running', 'replay': run_label})
                    rows.extend(run_replay(out, case, mode, run_label))
                    save(out / 'rows.json', rows)
            finally:
                if service.exists():
                    campaign.service('stop', arm, service)
    summary = {}
    for mode in dict.fromkeys(row['mode'] for row in rows):
        vals = [r['dispatch_s'] for r in rows if r['mode'] == mode and r['turn'] == 2]
        summary[mode] = {'n': len(vals), 'dispatch_s': vals, 'median_dispatch_s': statistics.median(vals)}
    save(out / 'summary.json', summary)
    save(out / 'status.json', {'status': 'passed'})


if __name__ == '__main__':
    main()
