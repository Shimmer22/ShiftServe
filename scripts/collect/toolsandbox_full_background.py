#!/usr/bin/env python3
"""Full 三条件串行重复测量；可等待既有首轮结束后接续，不覆盖原始结果。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', 'wins']


def remote_collector(remote, launch=False, config_tag='', preflight_args=()):
    """短连接查询；采集在 wins 独立会话运行，断线后不重新启动已有采集。"""
    code = r'''
import fcntl,json,pathlib,subprocess,sys
name,launch=sys.argv[1],sys.argv[2]=="True"
config_tag=sys.argv[3]
base=pathlib.Path("/home/wins/ai-workspace/outputs/a2o-gap-profiling")
p=base/name
lock=(base/(name+".launch.lock")).open("a")
fcntl.flock(lock,fcntl.LOCK_EX)
completion=p/"completion.json"
if completion.exists():
 d=json.loads(completion.read_text())
 if not d.get("completed") or d.get("groups")!=73:raise RuntimeError("invalid completion")
 print(json.dumps(dict(phase="complete",completed=73)));sys.exit()
processes=subprocess.check_output(["ps","-eo","pid,args"],text=True).splitlines()
active=[x.split()[0] for x in processes if "a2o.workloads" not in x and "toolsandbox.py --config" in x and "--output /workspace/outputs/a2o-gap-profiling/"+name in x and "python3 -c" not in x]
if not active:
 if not launch or p.exists():raise RuntimeError("collector absent; partial output retained: "+str(p))
 condition=name.rsplit("-",1)[1]
 command=["docker","exec",*json.loads(sys.argv[4]),"a2o-toolsandbox-py311","/workspace/tools/a2o-toolsandbox-py311-env/bin/python","/workspace/projects/a2o-gap-profiling/scripts/collect/toolsandbox.py","--config","/workspace/projects/a2o-gap-profiling/configs/toolsandbox-full-"+condition+config_tag+".json","--output","/workspace/outputs/a2o-gap-profiling/"+name]
 with (base/(name+".detached.log")).open("a") as log:
  proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 active=[str(proc.pid)]
summary=p/"summary.json"
d=json.loads(summary.read_text()) if summary.exists() else {}
print(json.dumps(dict(phase="collecting",pids=active,completed=d.get("completed",0))))
'''
    command = SSH + [shlex.join(['python3', '-c', code, remote, str(launch), config_tag, json.dumps(preflight_args)])]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode == 255:
        raise ConnectionError(result.stderr[-1000:])
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return json.loads(result.stdout)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.workloads.toolsandbox import post
from a2o.runtime.network_retry import retry_network
from a2o.runtime.phone_preflight import docker_preflight_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--start-round', type=int, default=1)
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('--binary', type=Path, default=ROOT/'results/20261107-toolsandbox-route-e2e/build-service-143119/agent')
    parser.add_argument('--cost-model', type=Path)
    parser.add_argument('--config-tag', default='')
    parser.add_argument('--adopt-first-cpu', action='store_true', help='接管已经运行的首轮 CPU，不重启手机服务')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/20261107-toolsandbox-full')
    args = parser.parse_args()
    if args.rounds < 1 or args.start_round < 1:
        parser.error('rounds/start-round 必须大于零')
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    control = base / 'repeats'
    control.mkdir(exist_ok=True)

    def status(**fields):
        tmp = control / 'progress.tmp'
        tmp.write_text(json.dumps(dict(updated_unix=time.time(), **fields), ensure_ascii=False, indent=2))
        tmp.replace(control / 'progress.json')

    def run(command, folder, name):
        while True:
            with (folder / (name + '.log')).open('a') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                return
            if command[0] == 'rsync' and result.returncode in (10, 12, 30, 35, 255):
                status(phase='reconnecting_download', operation=name, exit_code=result.returncode)
                time.sleep(30)
                continue
            raise subprocess.CalledProcessError(result.returncode, command)

    repeat_lock = (control / 'runner.lock').open('a')
    fcntl.flock(repeat_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (control / 'plan.json').write_text(json.dumps(dict(start_round=args.start_round, rounds=args.rounds,
        after_pid=args.after_pid, order_rotation=True, seed_unchanged=True,
        noise_scope='端到端总波动，包含工具时间/UUID变化；非纯硬件噪声'), ensure_ascii=False, indent=2) + '\n')
    try:
        if args.after_pid:
            while True:
                try:
                    os.kill(args.after_pid, 0)
                except ProcessLookupError:
                    break
                status(phase='waiting_for_first_round', pid=args.after_pid)
                time.sleep(15)
            previous = json.loads((base / 'progress.json').read_text())
            if previous.get('phase') != 'complete':
                raise RuntimeError('首轮未完整结束，禁止与遗留服务争抢设备；检查首轮日志')
            for condition in ('cpu', 'htp', 'route'):
                complete = json.loads((base / condition / 'completion.json').read_text())
                if not complete.get('completed') or complete.get('groups') != 73:
                    raise RuntimeError('首轮未完整尝试73题：' + condition)
        plan = dict(start_round=args.start_round, rounds=args.rounds,
                    order_rotation=True, seed_unchanged=True,
                    noise_scope='端到端重复波动，包含未固定的时钟/UUID/工具结果；非纯硬件噪声')
        (control / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        conditions = ['cpu', 'htp', 'route']
        for number in range(args.start_round, args.start_round + args.rounds):
            folder = base if number == 1 else base / f'round-{number:02d}'
            if number != 1:
                folder.mkdir(exist_ok=True)
            order = conditions[(number-1) % 3:] + conditions[:(number-1) % 3]
            for condition in order:
                service = folder / f'service-full-r{number:02d}-{condition}'
                adopt = args.adopt_first_cpu and number == 1 and condition == 'cpu'
                if adopt:
                    service = ROOT / 'results/20261107-toolsandbox-route-e2e/service-full-cpu'
                checkpoint = folder / (condition + '-stage.json')
                stage = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
                if stage.get('stopped'):
                    continue
                status(round=number, condition=condition, phase='starting', order=order)
                command = [sys.executable, 'scripts/run/route_phone.py', 'start', '--retry-network',
                           '--condition', condition, '--binary',
                           str(args.binary),
                           '--registry', 'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so',
                           '--output', str(service)]
                if not adopt and not stage.get('started'):
                    run(command, folder, condition + '-start')
                stage['started'] = True
                checkpoint.write_text(json.dumps(stage))
                if condition == 'route' and not stage.get('collecting'):
                    retry_network(lambda:post('http://10.29.131.50:18780', '/a2o/costs', json.loads(
                        (args.cost_model or ROOT / 'results/20261107-toolsandbox-route-e2e/calibration/costs.json').read_text()), timeout=30),
                        report=lambda message:status(round=number, condition=condition, phase='reconnecting_costs', error=message))
                remote = f'20261107-full-{condition}' if number == 1 else f'20261107-full-r{number:02d}-{condition}'
                status(round=number, condition=condition, phase='collecting', expected=73)
                gate = docker_preflight_args(ROOT, service, expected_owner=service.name)
                while True:
                    try:
                        current = remote_collector(remote, launch=not adopt, config_tag=args.config_tag, preflight_args=gate)
                    except (ConnectionError, subprocess.TimeoutExpired) as exc:
                        status(round=number, condition=condition, phase='reconnecting', error=str(exc))
                        time.sleep(30)
                        continue
                    stage['collecting'] = True
                    checkpoint.write_text(json.dumps(stage))
                    status(round=number, condition=condition, expected=73, **current)
                    if current['phase'] == 'complete':
                        break
                    time.sleep(30)
                run(['rsync', '-a', '--timeout=30', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2', f'wins:/home/wins/ai-workspace/outputs/a2o-gap-profiling/{remote}/',
                     str(folder / condition) + '/'], folder, condition + '-download')
                run([sys.executable, 'scripts/run/route_phone.py', 'stop', '--retry-network', '--condition', condition,
                     '--output', str(service)], folder, condition + '-stop')
                stage['stopped'] = True
                checkpoint.write_text(json.dumps(stage))
            run([sys.executable, 'scripts/analyze/toolsandbox_repeats.py', str(base)], control, 'analysis')
        status(phase='complete', additional_rounds=args.rounds)
    except BaseException as exc:
        status(phase='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
