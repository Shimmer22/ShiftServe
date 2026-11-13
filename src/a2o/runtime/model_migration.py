#!/usr/bin/env python3
"""独立后台顺序执行模型安装、基础准入及正式成本采集；任何失败停止。"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import json

ROOT = Path(__file__).resolve().parents[3]
from a2o.profiling.model_cost_campaign import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'shared/models/Qwen3-8B-Q4_0.gguf')
    parser.add_argument('--binary', type=Path, default=ROOT / 'results/20261113-qwen3-8b-migration/build-single-resident/agent')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/20261113-qwen3-8b-migration')
    parser.add_argument('--after-install-pid', type=int, help='等待本次已启动安装器退出，校验完成和清理记录后继续')
    parser.add_argument('--collect-only', action='store_true', help='复用已通过的准入和仍存活的同一服务，只接续成本采集')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.output / 'pipeline-status.json'

    def status(phase, **fields):
        save(status_path, dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **fields))
        print(phase, fields, flush=True)

    stages = [
        ('installing_model', [sys.executable, '-u', str(ROOT / 'scripts/run/install_model.py'),
                             '--source', str(args.source.resolve()), '--output', str(args.output / 'model-install'), '--resume']),
        ('preflight', [sys.executable, '-u', str(ROOT / 'scripts/collect/model_cost_campaign.py'),
                       '--binary', str(args.binary.resolve()), '--phase', 'preflight']),
        ('collecting', [sys.executable, '-u', str(ROOT / 'scripts/collect/model_cost_campaign.py'),
                        '--binary', str(args.binary.resolve()), '--phase', 'collect']),
    ]
    if args.collect_only:
        if args.after_install_pid:
            parser.error('--collect-only不能与--after-install-pid同时使用')
        stages=stages[2:]
    try:
        if args.after_install_pid:
            while True:
                check=subprocess.run(['ps','-p',str(args.after_install_pid),'-o','command='],capture_output=True,text=True)
                if check.returncode:
                    break
                if 'scripts/run/install_model.py' not in check.stdout:
                    raise RuntimeError('安装器PID已被其他进程复用，拒绝继续')
                progress=args.output/'model-install/progress.json'
                status('installing_model', child_pid=args.after_install_pid, heartbeat=True,
                       transfer=json.loads(progress.read_text()) if progress.exists() else None)
                time.sleep(15)
            installed=json.loads((args.output/'model-install/completion.json').read_text())
            cleanup=json.loads((args.output/'model-install/cleanup.json').read_text())
            assert installed['installed'] and cleanup['lock_released'] and cleanup['wake_unlocked']
            stages=stages[1:]
        for phase, command in stages:
            with (args.output / (phase + '.log')).open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status(phase, child_pid=child.pid, command=command)
                while child.poll() is None:
                    progress=args.output/'model-install/progress.json'
                    fields=dict(child_pid=child.pid,heartbeat=True)
                    if phase=='installing_model' and progress.exists():
                        fields['transfer']=json.loads(progress.read_text())
                    detail=ROOT/'results/20261113-qwen3-8b-key-costs/status.json'
                    if phase in ['preflight','collecting'] and detail.exists():
                        fields['work']=json.loads(detail.read_text())
                    status(phase,**fields)
                    time.sleep(15)
                returncode = child.returncode
                if returncode:
                    raise RuntimeError(f'{phase}失败，退出码{returncode}；后续阶段未启动')
        status('complete')
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
