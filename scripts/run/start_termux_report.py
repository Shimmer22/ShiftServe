#!/usr/bin/env python3
"""脱离当前对话启动原报告复跑，日志和进度留在具名results目录。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight-run', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    out = (a.output or ROOT / 'results' / (stamp + '-termux-report-rerun')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = ['/usr/bin/caffeinate', '-i', sys.executable,
        str(ROOT / 'scripts/profile/termux_campaign.py'), '--output', str(out),
        '--preflight-run', str(a.preflight_run.resolve())]
    with (out / 'console.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (out / 'detached.json').write_text(json.dumps(dict(pid=proc.pid, command=command,
        log=str(out / 'console.log'), detached=True), indent=2) + '\n')
    print(str(out))
    print('Detached PID:', proc.pid)


if __name__ == '__main__':
    main()
