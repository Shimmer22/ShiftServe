#!/usr/bin/env python3
"""在 wins 定点验收 Mobile GUI Table 1，随后运行固定历史的 Prefill 扫描。"""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from a2o.runtime.device_transport import DeviceTransport
from scripts.run.t6_mobile_gui_table1 import ARMS, SHA, fixture_plan

MAIN = ROOT / 'results/mainline-eval'
T6 = MAIN / 't6-8b-table1-mobile-gui'
OUT = MAIN / 'prefill-motivation-h4096-8b'
CONFIG = ROOT / 'configs/prefill-motivation-h4096-8b.json'
BUILD = ROOT / 'results/20261122-kv-window-final-android'
SERVICE = OUT / 'prefill-motivation-h4096-8b-service'
BEIJING = ZoneInfo('Asia/Shanghai')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    pending.replace(path)


def accept_table():
    corpus, max_length = fixture_plan(BUILD)
    plan = read(T6 / 'plan.json')
    gate = read(T6 / 'gate.json')
    if (plan['context'] != 9472 or gate['status'] != 'passed'
            or gate['arms']['shiftserve_joint']['prior_ranked_tokens'] <= 0
            or any(a['memory_ledger_peak_swap']['swap_kib'] for a in gate['arms'].values())):
        raise ValueError('Table 1 gate failed')
    rows = {}
    totals = Counter()
    for arm in ARMS:
        for case in corpus['cases']:
            case_id = case['id']
            directory = T6 / arm / case_id
            row = read(directory / 'summary.json')
            source = T6 / read(directory / 'service.json')['directory']
            health = read(source / 'health.json')
            if (health['git_commit'] != SHA or health['gap_ms'] != 0 or health['verify_kv_state']
                    or health['context_size'] != 9472 or not (source / 'device/runtime.log').stat().st_size):
                raise ValueError(f'{arm}/{case_id} sealed service audit')
            phases = row['post_return_phases_ms']
            if (row['warm_rounds'] != len(case['turns']) - 1
                    or not math.isclose(math.fsum(value for key, value in phases.items() if key != 'prior'),
                                        row['post_return_ms'], rel_tol=0, abs_tol=1e-6)
                    or row['memory_ledger_missing_pss_plus_dma_samples']):
                raise ValueError(f'{arm}/{case_id} incomplete phase or memory ledger')
            for i, turn in enumerate(case['turns']):
                response = read(directory / f'turn-{i:02d}.json')['response']
                t = response['timings']
                ids = turn['request']['frozen_decode_ids']
                expected_route = {'full_cpu': 'cpu', 'full_htp': 'htp'}.get(arm)
                expected_decode = 'htp' if arm == 'full_htp' else 'cpu'
                kv = t['kv_reduce']
                prior = bool(kv and kv['policy'] == 'prefill_prior' and kv['prior_ranked_tokens'] > 0)
                profile = t['graph_profile']
                if (t['P_evaluated'] != turn['P'] or response['__verbose']['tokens'] != ids
                        or t['route'] not in ('cpu', 'htp')
                        or (expected_route and t['route'] != expected_route)
                        or t['decode_backend'] != expected_decode
                        or profile['prefill']['callback_graphs'] != int(prior)
                        or profile['decode']['callback_graphs'] != 0
                        or (kv and (i == 0 or kv['recent_tokens'] < turn['P']))):
                    raise ValueError(f'{arm}/{case_id}/turn-{i}: frozen input or backend mismatch')
            totals[arm] += row['warm_rounds']
            rows[f'{arm}/{case_id}'] = dict(swap=row['memory_ledger_peak_swap'],
                                            pss_dma=row['memory_ledger_peak_pss_plus_dma'])
    if len(rows) != 48 or any(warm != 50 for warm in totals.values()):
        raise ValueError('missing Table 1 cases or warm rounds')
    for parent in (T6, T6 / 'droidtask', T6 / 'androidcontrol_low'):
        if not (parent / 'table1.csv').stat().st_size:
            raise ValueError('missing Table 1 summary')
    if not (T6 / 'table1-breakdown.csv').stat().st_size:
        raise ValueError('missing breakdown')
    return dict(status='passed', build=SHA, max_logical_tokens=max_length, cells=len(rows),
                warm_rounds=dict(totals), captured_at=datetime.now(BEIJING).isoformat(),
                swap_nonzero={key: m['swap'] for key, m in rows.items() if m['swap']['swap_kib']},
                peak_pss_dma=max((m['pss_dma'] for m in rows.values()), key=lambda m: m['kib']))


def verify_sweep_fixture():
    corpus, _ = fixture_plan(BUILD)
    fixture = read(OUT / 'fixture.json')
    case = next(c for c in corpus['cases'] if c['id'] == fixture['case'])
    turn = case['turns'][fixture['turn']]['request']
    if (fixture['choices'][0]['prompt_token_ids'] != turn['frozen_prompt_ids'][:6144]
            or len(fixture['choices'][0]['prompt_token_ids']) != 6144
            or fixture['choices'][0]['token_ids'] != turn['frozen_decode_ids'][:8]):
        raise ValueError('motivation tokens differ from the frozen source')


def snapshot_phone():
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    base = shlex.quote(transport.remote_root)
    output = transport.call(
        'shell', f'd={base}; if [ -f "$d/model.lock/owner" ]; then '
        'a=$(cat "$d/model.lock/owner"); printf "owner=%s\n" "$a"; '
        's="$d/$a"; if [ -f "$s/pid" ]; then p=$(cat "$s/pid"); '
        'if ! kill -0 "$p" 2>/dev/null; then printf "service_alive=0\\n"; '
        'elif [ -r "/proc/$p/cmdline" ] && tr "\\000" " " < "/proc/$p/cmdline" | '
        'grep -F -- "$s/service " >/dev/null; then printf "service_alive=1\\n"; fi; '
        '/system/bin/ps -p "$p" -o PID,STAT,TIME,ARGS; fi; '
        'if [ -f "$s/runtime.log" ]; then stat -c "%y %s" "$s/runtime.log"; '
        'tail -n 2 "$s/runtime.log" | cut -c1-200; fi; fi').stdout
    return output.decode()


def stop_sweep_service():
    subprocess.run([sys.executable, 'scripts/run/route_phone.py', 'stop', '--condition', 'route',
                    '--output', str(SERVICE), '--logs-only', '--retry-network'], cwd=ROOT, check=True)


def sweep(plot_python):
    verify_sweep_fixture()
    command = [sys.executable, '-u', 'scripts/run/termux_http_forward.py', '--port', '28791', '--',
               sys.executable, '-u', 'scripts/collect/pd_stage_sweep.py', '--config', str(CONFIG),
               '--output', str(OUT), '--binary', str(BUILD / 'agent'), '--runtime-overlay', str(BUILD)]
    results = OUT / 'progress.json'
    with (OUT / 'driver.log').open('a') as log:
        for attempt in range(1, 4):
            started = datetime.now(BEIJING)
            deadline = time.monotonic() + 3600
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            save(results, dict(status='running', attempt=attempt, pid=p.pid,
                               started=started.isoformat(), next_health=(started + timedelta(hours=1)).isoformat()))
            checked = False
            while p.poll() is None:
                if not checked and time.monotonic() >= deadline:
                    health = dict(time=datetime.now(BEIJING).isoformat(), attempt=attempt, driver_pid=p.pid)
                    health['saved_pairs'] = len([x for x in (OUT / 'measurements').glob('r*.json') if '.raw.' not in x.name])
                    health['state'] = read(OUT / 'status.json')
                    try:
                        health['phone'] = snapshot_phone()
                        health['ssh'] = 'ok'
                        phase = health['state'].get('phase')
                        if phase in ('measuring', 'building_history_cache', 'waiting_for_network'):
                            if 'service_alive=0\n' in health['phone'] and f'owner={SERVICE.name}\n' in health['phone']:
                                health['action'] = 'restart_missing_phone_service'
                                p.send_signal(signal.SIGINT)
                    except (OSError, subprocess.SubprocessError) as error:
                        health.update(ssh='retrying', error=str(error),
                                      action='phone_listener_supervisor_and_forwarder_reconnecting')
                    save(OUT / f'health-1h-attempt-{attempt}.json', health)
                    checked = True
                time.sleep(15)
            if p.returncode == 0:
                if not checked:
                    save(OUT / f'health-1h-attempt-{attempt}.json',
                         dict(status='completed_before_health_check', time=datetime.now(BEIJING).isoformat()))
                break
            state = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
            save(OUT / f'failed-attempt-{attempt}.json', state)
            if attempt == 3:
                save(results, dict(status='failed', attempt=attempt, returncode=p.returncode))
                raise RuntimeError('Prefill controller failed three times')
            phone = snapshot_phone()
            if f'owner={SERVICE.name}\n' in phone and 'service_alive=0\n' in phone:
                stop_sweep_service()
            # 仍活跃的服务只重接；已落盘的有效配对会跳过。
            time.sleep(30)
    subprocess.run([str(plot_python), 'scripts/plot/pd_prefill_dense.py', '--clean-upper-tail',
                    '--input', str(OUT / 'stage-times.csv')], cwd=ROOT, check=True)
    layout = read(OUT / 'figures-cleaned/layout-check.json')
    if layout['overlaps'] or layout['outside']:
        save(results, dict(status='plot_needs_review', layout=layout))
        return
    save(results, dict(status='complete', completed=datetime.now(BEIJING).isoformat(),
                       measurements_sha256=hashlib.sha256((OUT / 'stage-times.csv').read_bytes()).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=datetime.fromisoformat)
    parser.add_argument('--plot-python', type=Path)
    parser.add_argument('--accept-table', action='store_true')
    args = parser.parse_args()
    approval = T6 / 'final-acceptance.json'
    if args.accept_table:
        state = read(T6 / 'status.json')
        if state.get('status') != 'passed' or state.get('formal_cells') != 48:
            parser.error('Table 1 尚未完成')
        acceptance = accept_table()
        save(approval, acceptance)
        print(json.dumps(acceptance, indent=2, ensure_ascii=False))
        return
    if not args.first or not args.plot_python:
        parser.error('后台等待需要 --first 和 --plot-python')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_sweep_fixture()
        when = args.first.astimezone(BEIJING)
        save(OUT / 'queue.json', dict(status='waiting_for_acceptance', earliest_start=when.isoformat(), pid=os.getpid()))
        time.sleep(max(0, when.timestamp() - time.time()))
        while not approval.exists():
            time.sleep(10)
        acceptance = read(approval)
        if acceptance.get('status') != 'passed' or acceptance.get('build') != SHA:
            raise ValueError('Table 1 acceptance does not match the sealed build')
        # 原控制器退出并结束 HTTP 转发后才启动下一个控制器。
        time.sleep(30)
        save(OUT / 'queue.json', dict(status='table1_accepted', started=datetime.now(BEIJING).isoformat()))
        sweep(args.plot_python)


if __name__ == '__main__':
    main()
