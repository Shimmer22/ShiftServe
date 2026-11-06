"""可脱离对话运行的 SSH 串行复测：只采样和验收，不出图或生成报告。"""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from a2o.environment import ROOT
from a2o.runtime.device_transport import DeviceTransport


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_raw(run, count):
    read = lambda n: [json.loads(x) for x in (run / n).read_text().splitlines()]
    rows, raw, http = map(read, ['checked-rows.jsonl', 'device/trials.jsonl', 'http-server.jsonl'])
    completion = json.loads((run / 'completion.json').read_text())
    if len(rows) != count or len(raw) != count or len(http) != count or not completion['passed']:
        raise ValueError('记录数量或completion验收失败')
    if sorted(r['id'] for r in rows) != list(range(count)):
        raise ValueError('trial ID不完整')
    if any(any(r[k] != value for k, value in a.items()) for r, a in zip(rows, raw)):
        raise ValueError('原始设备数据与checked行不一致')
    if not all(r['passed'] and r['response_check']['passed'] and r['generated_ids_equal'] and r['eog'] for r in rows):
        raise ValueError('完整输出或状态验收失败')
    if not all(r['numerical_passed'] for r in rows if r['phase'] == 'calibration'):
        raise ValueError('数值校准失败')
    by_http = {h['id']: h for h in http}
    if len(by_http) != count or set(by_http) != set(range(count)):
        raise ValueError('HTTP ID不完整')
    for r in rows:
        h = by_http[r['id']]
        if (h['P'], h['gap_requested_ms']) != (r['P'], r['gap_requested_ms']):
            raise ValueError('HTTP条件不一致')
        if h['body_sha256'] != hashlib.sha256((run / f'P{h["P"]}.txt').read_bytes()).hexdigest():
            raise ValueError('HTTP正文hash错误')
        if abs(r['actual_gap_ms'] + r['response_after_return_ms'] - r['tool_to_response_ms']) > .01:
            raise ValueError('主时钟恒等式失败')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--preflight-run', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'results/.termux-model.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='starting', pid=os.getpid(), transport='ssh', runs=[],
                 expected_report_trials=854, completed_report_trials=0,
                 preflight=str(args.preflight_run.resolve()), report_generation=False,
                 started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 estimated_duration_minutes=90,
                 estimated_finish=(datetime.now(ZoneInfo('Asia/Shanghai')) + timedelta(minutes=90)).isoformat())
    active = None
    started = time.monotonic()

    def save():
        state['updated'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        tmp = out / 'campaign.json.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(out / 'campaign.json')

    def interrupt(signum, frame):
        raise KeyboardInterrupt('campaign interrupted')

    signal.signal(signal.SIGTERM, interrupt)
    try:
        audit_raw(args.preflight_run.resolve(), 6)
        plan = [('observation-main-h2048-termux', 381),
                ('observation-main-h4096-termux', 381),
                ('post-tool-p31-paired-h2048-termux', 46),
                ('post-tool-p31-paired-h4096-termux', 46)]
        # 固化本次配置和源码来源，不能受后续配置编辑影响。
        prepared = []
        for stem, expected in plan:
            config = ROOT / 'configs' / (stem + '.json')
            snapshot = out / (stem + '.json')
            snapshot.write_bytes(config.read_bytes())
            prepared.append((stem, snapshot, expected))
        save()
        source_binary = None
        for stem, config, expected in prepared:
            log_path = out / (stem + '.log')
            state.update(status='running', active_condition=stem, active_run=None, active_trials=0)
            with log_path.open('w') as log:
                active = subprocess.Popen([sys.executable, str(ROOT / 'scripts/profile/oracle.py'),
                    '--transport', 'ssh', '--config', str(config)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state['active_pid'] = active.pid
                save()
                while active.poll() is None:
                    lines = log_path.read_text().splitlines()
                    if lines and Path(lines[0]).is_dir():
                        run = Path(lines[0]); state['active_run'] = str(run)
                        checked = run / 'checked-rows.jsonl'
                        state['active_trials'] = len(checked.read_text().splitlines()) if checked.exists() else 0
                    save()
                    if time.monotonic() - started > 3 * 3600:
                        raise TimeoutError('campaign exceeded three-hour limit')
                    time.sleep(5)
                if active.returncode:
                    raise RuntimeError(f'{stem}失败，见{log_path}')
            run = Path(log_path.read_text().splitlines()[0])
            rows = audit_raw(run, expected)
            manifest = json.loads((run / 'manifest.json').read_text())
            if source_binary and source_binary != manifest['binary_sha256']:
                raise ValueError('批次之间二进制发生变化')
            source_binary = manifest['binary_sha256']
            if stem.startswith('observation-main'):
                load_module('observation_audit', 'scripts/plot/observations.py').load_run(run)
                detail = dict(passed=True, trials=len(rows), kind='observation')
            else:
                detail = load_module('paired_audit', 'scripts/analyze/post_tool_p31_retest.py').analyze(run)
            (run / 'campaign-acceptance.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2) + '\n')
            state['runs'].append(dict(config=stem, directory=str(run), trials=len(rows), accepted=True))
            state['completed_report_trials'] += len(rows)
            save()
        state.update(status='complete', active_condition=None, active_pid=None, active_run=None,
                     active_trials=0, elapsed_minutes=(time.monotonic() - started) / 60)
        save()
    except BaseException as exc:
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT)
            try:
                active.wait(timeout=150)
            except subprocess.TimeoutExpired:
                state['cleanup_warning'] = 'runner did not finish cleanup; retain remote lock; do not start another model'
        state.update(status='failed', error=repr(exc), elapsed_minutes=(time.monotonic() - started) / 60)
        save()
        raise
    finally:
        # 释放本次实验持锁，保留SSH服务；不修改系统设置。
        transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
        try:
            result = subprocess.run(transport.prefix + ['termux-wake-unlock'],
                                    capture_output=True, timeout=30)
            (out / 'wake-unlock.log').write_bytes(result.stdout + result.stderr)
            state['wake_unlock_returncode'] = result.returncode
        except Exception as exc:
            state['wake_unlock_error'] = repr(exc)
        save()


if __name__ == '__main__':
    main()
