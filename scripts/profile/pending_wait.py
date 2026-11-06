#!/usr/bin/env python3
"""串行执行有限等待配对，复用Oracle入口与原始数据验收，不生成PDF。"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling.pending_wait import analyze_pending_run
from a2o.runtime.device_transport import DeviceTransport


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'configs/pending-wait-h4096-termux.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'results/.termux-model.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = out / 'config.json'; cfg.write_bytes(a.config.read_bytes())
    state = dict(status='running', config=str(cfg), transport='ssh',
                 started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    def save():
        state['updated'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        (out / 'campaign.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    save()
    try:
        log = out / 'runtime-console.log'
        with log.open('w') as f:
            subprocess.run([sys.executable, str(ROOT / 'scripts/profile/oracle.py'), '--transport', 'ssh',
                            '--config', str(cfg)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        run = Path(log.read_text().splitlines()[0])
        state['run'] = str(run)
        summary = analyze_pending_run(run)
        (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        state.update(status='complete', accepted=True, trials=summary['trials'], pairs=summary['pairs'])
        save()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); save(); raise
    finally:
        transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
        try:
            r = subprocess.run(transport.prefix + ['termux-wake-unlock'], capture_output=True, timeout=30)
            (out / 'wake-unlock.log').write_bytes(r.stdout + r.stderr)
            state['wake_unlock_returncode'] = r.returncode
        except Exception as exc:
            state['wake_unlock_error'] = repr(exc)
        save()


if __name__ == '__main__':
    main()
