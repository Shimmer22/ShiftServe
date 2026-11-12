#!/usr/bin/env python3
"""等待旧 BFCL 复测完整落盘，串行切换缓存服务并从本机直连采集。"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network

PREVIOUS = '/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261108-bfcl-text-pilot/phone-frozen-retest'
LOCAL_PREVIOUS = ROOT / 'results/20261108-bfcl-text-pilot/phone-frozen-retest'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--previous-pid', required=True, type=int)
    parser.add_argument('--previous-start-ticks', required=True)
    parser.add_argument('--accept-stopped-previous', action='store_true')
    parser.add_argument('--resume-collection', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    device = transport.remote_root + '/' + out.name
    endpoint = 'http://10.29.131.50:18780'
    plan = dict(binary=str(args.binary.resolve()), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                requests_sha256=hashlib.sha256((LOCAL_PREVIOUS / 'requests.json').read_bytes()).hexdigest(),
                previous_pid=args.previous_pid, previous_start_ticks=args.previous_start_ticks,
                rounds=3, context_size=8192, accept_stopped_previous=args.accept_stopped_previous,
                conditions=['cpu', 'htp', 'route'],
                scope='冻结 token 机制重放；不包含实际 Tool Gap 异步重叠或质量测量')
    plan_path = out / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('拒绝混用不同接管配置')
    else:
        write(plan_path, plan)

    def status(phase, **fields):
        write(out / 'status.json', dict(phase=phase, updated_unix=time.time(), **fields))

    def checked(command):
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout

    def remote(command):
        return retry_network(lambda: transport.call('shell', command)).stdout.decode()

    def post(path, body):
        request = urllib.request.Request(endpoint + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=2400) as response:
            return json.load(response)

    def stop_owned(owner, pid, start_ticks, executable):
        q = shlex.quote
        # 同时核对锁、PID 启动时间和可执行路径，防止误杀复用 PID。
        guard = ('test "$(cat ' + q(transport.remote_root + '/model.lock/owner') + ')" = ' + q(owner) +
                 ' && test "$(awk \'{print $22}\' /proc/' + str(pid) + '/stat)" = ' + q(str(start_ticks)) +
                 ' && tr "\\000" " " < /proc/' + str(pid) + '/cmdline | grep -F ' + q(executable))
        remote(guard + ' && kill ' + str(pid))
        for _ in range(30):
            if remote('if kill -0 ' + str(pid) + ' 2>/dev/null; then echo alive; else echo stopped; fi').strip() == 'stopped':
                return
            time.sleep(1)
        raise RuntimeError('服务退出超时，保留锁和现场')

    try:
        service_out = out / (out.name + '-service')
        device = transport.remote_root + '/' + service_out.name
        model_path = LOCAL_PREVIOUS / 'cost-model.json'
        if args.resume_collection:
            q=shlex.quote
            remote('test "$(cat '+q(transport.remote_root+'/model.lock/owner')+')" = '+q(service_out.name))
            digest=remote('sha256sum '+q(device+'/service')).split()[0]
            assert digest==plan['binary_sha256'], 'running service binary changed'
            with urllib.request.urlopen(endpoint+'/health',timeout=2400) as response:
                health=json.load(response)
            assert health['condition']=='route' and health['context_size']==8192 and health['replay_cache_version']==2
        else:
            while True:
                previous = json.loads(retry_network(lambda: checked([
                    'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'wins',
                    'cat ' + shlex.quote(PREVIOUS + '/status.json')])))
                status('waiting_previous', previous=previous)
                if args.accept_stopped_previous and previous['phase'] == 'stopped_for_cached_replay':
                    break
                if previous['phase'] == 'failed':
                    raise RuntimeError('旧复测失败，保留其服务和现场，不自动接管')
                if previous['phase'] == 'complete':
                    if previous['completed'] != 24:
                        raise RuntimeError('旧复测完成数不符')
                    break
                time.sleep(30)
            status('syncing_previous')
            retry_network(lambda: checked(['rsync', '-a', 'wins:' + PREVIOUS + '/', str(LOCAL_PREVIOUS) + '/']))
            requests = json.loads((LOCAL_PREVIOUS / 'requests.json').read_text())
            for i, request in enumerate(requests):
                path = LOCAL_PREVIOUS / f'{i:02d}.json'
                if not path.exists() and args.accept_stopped_previous:
                    continue
                old = json.loads(path.read_text())
                if len(old['response']['results']) != 3 or any(
                    any(row[k] != request[k] for k in ['H', 'P', 'D']) for row in old['response']['results']):
                    raise RuntimeError('旧复测 H/P/D 校验失败，拒绝接管')
            status('switching_service')
            old_device = transport.remote_root + '/phone-cpu-replay'
            stop_owned('phone-cpu-replay', args.previous_pid, args.previous_start_ticks, old_device + '/replay-agent')
            retry_network(lambda: transport.call('pull', old_device + '/paired-runtime.log', str(LOCAL_PREVIOUS / 'paired-runtime.log')))
            transport.lock_owner = 'phone-cpu-replay'
            transport.release_lock()
            service_out = out / (out.name + '-service')
            command = [sys.executable, str(ROOT / 'scripts/run/route_phone.py'), 'start', '--condition', 'route',
                       '--binary', str(args.binary.resolve()), '--context-size', '8192', '--replay-cache',
                       '--output', str(service_out), '--retry-network']
            # 与服务目录保持一致，不能与其他实验共用目录或模型锁。
            with (out / 'service-launch.log').open('a') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            device = transport.remote_root + '/' + service_out.name
            status('cache_smoke')
            model_path = LOCAL_PREVIOUS / 'cost-model.json'
            model = json.loads(model_path.read_text())
            post('/a2o/costs', {'model': model.get('model', model)})
            # 仅验证机制；正式测量下方继续使用原始完整 H/P/D。
            smoke = dict(prompt_ids=requests[0]['prompt_ids'][:9], generated_ids=requests[0]['generated_ids'][:1], H=8)
            cold = post('/a2o/replay', dict(smoke, cache_only=True))
            memory = post('/a2o/replay', smoke)
            write(out / 'cache-smoke.json', dict(scope='短 token 机制检查，不用于性能或质量结论', cold=cold, memory=memory))
            assert cold['cache']['source'] == 'bootstrap' and memory['cache']['source'] == 'memory'
            assert len(memory['results']) == 3
        status('collecting', rounds=3, boundaries=24)
        with (out / 'collector.log').open('a') as log:
            cache_host = '/home/wins/ai-workspace/cache/a2o-frozen-replay/' + plan['binary_sha256']
            backup_command = [sys.executable, str(ROOT / 'scripts/collect/bfcl_cache_archive.py'), 'backup',
                              '--phone-directory', device + '/replay-cache', '--host-directory', cache_host,
                              '--phone-owner', service_out.name,
                              '--output', str(out / 'cache-archive.json')]
            subprocess.run([sys.executable, str(ROOT / 'scripts/collect/bfcl_phone_cached.py'),
                            '--requests', str(LOCAL_PREVIOUS / 'requests.json'), '--cost-model', str(model_path),
                            '--output', str(out / 'measurements'), '--rounds', '3',
                            '--backup-command-json', json.dumps(backup_command)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        rows = [json.loads(path.read_text()) for path in sorted((out / 'measurements').glob('r*-b*.json'))]
        assert len(rows) == 72
        sources = {}
        for row in rows:
            source = row['response']['cache']['source']
            sources[source] = sources.get(source, 0) + 1
        assert sources.get('disk', 0) > 0, '未观察到磁盘缓存命中'
        write(out / 'summary.json', dict(boundaries=24, rounds=3, condition_evaluations=216, cache_sources=sources,
              fixture_setup_s=sum(row['response']['bootstrap_ms'] for row in rows) / 1000,
              measured_s={condition:sum(r['total_ms'] for row in rows for r in row['response']['results']
                                       if r['condition']==condition)/1000 for condition in ['cpu','htp','route']},
              scope=plan['scope']))
        status('stopping_service')
        pid = int(remote('cat ' + shlex.quote(device + '/pid')).strip())
        ticks = remote("awk '{print $22}' /proc/" + str(pid) + '/stat').strip()
        stop_owned(service_out.name, pid, ticks, device + '/service')
        retry_network(lambda: transport.call('pull', device + '/runtime.log', str(out / 'runtime.log')))
        transport.lock_owner = service_out.name
        transport.release_lock()
        checked(transport.prefix + ['termux-wake-unlock'])
        status('complete', rounds=3, boundaries=24, condition_evaluations=216)
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
