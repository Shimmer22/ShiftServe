#!/usr/bin/env python3
"""通过 Termux SSH 启停 Qwen3.5 手机 CPU 服务，持有项目模型锁。"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport

REMOTE = '/data/data/com.termux/files/home/.local/a2o'
OWNER = 'qwen35-phone-server'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/toolsandbox-qwen35-phone.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/20261106-qwen35-toolsandbox')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['backend'] != 'CPU' or cfg['concurrency'] != 1:
        raise ValueError('当前 server 构建仅支持显式 CPU 单槽配置')
    args.output.mkdir(parents=True, exist_ok=True)
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    service, lock = REMOTE + '/qwen35-server', REMOTE + '/runs/model.lock'

    def remote(command, timeout=120):
        result = subprocess.run(transport.prefix + [command], capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    if args.action == 'start':
        model = REMOTE + '/models/' + cfg['model']
        expected = cfg['model_sha256']
        # 模型未完整传输、锁被占用、已有进程时直接失败。
        remote('test ! -e ' + lock + ' && test ! -e ' + service + '/pid')
        digest = remote('sha256sum ' + shlex.quote(model), 300).split()[0]
        if digest != expected:
            raise RuntimeError('手机模型 SHA-256 不一致，拒绝启动')
        binary_digest = remote('sha256sum ' + shlex.quote(service + '/llama-server')).split()[0]
        if binary_digest != cfg['server_binary_sha256']:
            raise RuntimeError('手机 CPU server SHA-256 不一致，拒绝启动')
        remote('termux-wake-lock; mkdir ' + lock + ' && echo ' + OWNER + ' > ' + lock + '/owner')
        clock = remote(REMOTE + '/probes/clock-probe-static')
        (args.output / 'server-clock.jsonl').write_text(clock)
        rows = [json.loads(line) for line in clock.splitlines()]
        elapsed = rows[-1]['monotonic_s'] - rows[0]['monotonic_s']
        suspend = rows[-1]['boottime_s'] - rows[0]['boottime_s'] - elapsed
        if not 4.9 <= elapsed <= 7 or abs(suspend) >= .01:
            raise RuntimeError('手机持锁双时钟检查失败；保留现场')
        command = [service + '/llama-server', '-m', model, '--alias', 'qwen35-9b-q4_0',
                   '--host', '0.0.0.0', '--port', '18780', '--device', 'none', '-ngl', '0',
                   '-c', str(cfg['context_size']), '-np', '1', '-t', '4', '-tb', '4',
                   '-b', '256', '-ub', '256', '-fa', 'on', '--jinja',
                   '--chat-template-kwargs', '{"enable_thinking":false}',
                   '--no-context-shift', '--cache-reuse', '0', '--cache-ram', '0',
                   '--no-cache-idle-slots', '--ctx-checkpoints', '1', '--verbosity', '10']
        launch_ns = time.monotonic_ns()
        output = remote('nohup ' + shlex.join(command) + ' > ' + service +
                        '/server.log 2>&1 < /dev/null & echo $! > ' + service + '/pid; cat ' + service + '/pid')
        manifest = dict(command=command, pid=int(output.strip()), phone_model_sha256=digest,
                        phone_binary_sha256=binary_digest,
                        backend='CPU only; compiled without OpenCL or Hexagon',
                        lock_owner=OWNER, kv_policy='none', pre_move=False,
                        state='single process/context, native recurrent checkpoint=1 for prefix reuse',
                        endpoint='http://10.29.131.50:18780', transport='direct LAN HTTP; control Termux SSH')
        manifest['health_ready'] = False
        manifest_path = args.output / 'server-manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(manifest['endpoint'] + '/health', timeout=3) as response:
                    health = json.load(response)
                if health.get('status') == 'ok':
                    manifest.update(health_ready=True, launch_to_health_ms=(time.monotonic_ns()-launch_ns)/1e6,
                                    initialization_timing='includes process/context/server startup and health polling')
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        if not manifest['health_ready']:
            raise RuntimeError('手机服务未在 600 秒内 ready，保留日志和锁供诊断')
        print(json.dumps(manifest, indent=2))
    elif args.action == 'status':
        print(remote('cat ' + service + '/pid; tail -35 ' + service + '/server.log'))
    else:
        remote('test "$(cat ' + lock + '/owner)" = ' + OWNER)
        remote('if [ -f ' + service + '/pid ]; then p=$(cat ' + service +
               '/pid); if tr "\\0" " " < /proc/$p/cmdline | grep -Fq ' +
               shlex.quote(service + '/llama-server') + '; then kill "$p"; else exit 1; fi; rm ' +
               service + '/pid; fi; rmdir ' + lock +
               ' 2>/dev/null || { rm ' + lock + '/owner; rmdir ' + lock + '; }; termux-wake-unlock')
        print('手机服务已停止，实验锁与 wake lock 已释放。')


if __name__ == '__main__':
    main()
