"""显式设备传输；SSH 在 Termux UID 执行，禁止自动回退。"""
import json
from pathlib import Path
import shlex
import subprocess
import time


class DeviceTransport:
    def __init__(self, root, kind, device=None, config=None):
        self.kind, self.forward, self.lock_owner = kind, None, None
        self.root = Path(root)
        if kind == 'adb':
            if not device:
                raise ValueError('ADB requires an explicit device')
            self.prefix = ['adb', '-s', device]
            self.remote_root = '/data/local/tmp'
            self.identity = device
        else:
            self.config = json.loads(Path(config).read_text())
            c = self.config
            self.identity = f"{c['user']}@{c['host']}:{c['port']}"
            known = Path(c['known_hosts'])
            if not known.is_absolute():
                known = self.root / known
            self.options = ['-i', str(Path(c['identity_file']).expanduser()),
                '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'UserKnownHostsFile=' + str(known)]
            self.target = c['user'] + '@' + c['host']
            self.prefix = ['ssh', '-T', '-p', str(c['port']), *self.options, self.target]
            self.remote_root = c['remote_root']

    def shell_command(self, command):
        if self.kind == 'adb':
            return self.prefix + ['shell', '-T', command]
        # 使用 Android 工具的参数语义；Termux 的 ps 等工具参数不同。
        return self.prefix + ['export PATH=/system/bin:/data/data/com.termux/files/usr/bin; '
                              'unset LD_PRELOAD; ' + command]

    def call(self, *args):
        if self.kind == 'adb':
            return subprocess.run(self.prefix + list(args), check=True, capture_output=True, timeout=60)
        op, *rest = args
        if op == 'get-state':
            return subprocess.run(self.prefix + ['id'], check=True, capture_output=True, timeout=30)
        if op == 'shell':
            command = rest[0] if len(rest) == 1 else shlex.join(rest)
            return subprocess.run(self.shell_command(command), check=True, capture_output=True, timeout=60)
        if op in ('push', 'pull'):
            command = ['scp', '-r', '-P', str(self.config['port']), *self.options]
            if op == 'push':
                command += [str(rest[0]), self.target + ':' + str(rest[1])]
            else:
                command += [self.target + ':' + str(rest[0]), str(rest[1])]
            return subprocess.run(command, check=True, capture_output=True, timeout=120)
        if op == 'reverse':
            if rest[0] == '--remove':
                if self.forward:
                    self.forward.terminate()
                    self.forward.wait(timeout=15)
                    self.forward = None
                return subprocess.CompletedProcess(args, 0, b'', b'')
            remote, local = (int(x.split(':')[1]) for x in rest)
            self.forward = subprocess.Popen(['ssh', '-N', '-p', str(self.config['port']),
                *self.options, '-o', 'ExitOnForwardFailure=yes', '-R',
                f'127.0.0.1:{remote}:127.0.0.1:{local}', self.target],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.5)
            if self.forward.poll() is not None:
                raise RuntimeError('SSH reverse forwarding failed: ' + self.forward.stderr.read().decode())
            return subprocess.CompletedProcess(args, 0, b'', b'')
        raise ValueError('unsupported device operation: ' + op)

    def prepare_ssh(self, output, resume_lock=False):
        """先持锁并检查含挂起时钟；DUMP 权限缺失不伪报 Awake。"""
        subprocess.run(self.prefix + ['termux-wake-lock'], check=True, capture_output=True, timeout=30)
        self.call('shell', 'mkdir', '-p', self.remote_root)
        proc = subprocess.run(self.prefix + [shlex.quote(self.config['clock_probe'])],
                              check=True, capture_output=True, timeout=30)
        (output / 'ssh-clock.jsonl').write_bytes(proc.stdout)
        rows = [json.loads(line) for line in proc.stdout.decode().splitlines()]
        if len(rows) != 2:
            raise RuntimeError('invalid SSH clock probe output')
        delta = {k: rows[1][k] - rows[0][k] for k in rows[0]}
        suspend = delta['boottime_s'] - delta['monotonic_s']
        result = dict(device_delta_s=delta, suspend_s=suspend,
                      passed=4.9 <= delta['monotonic_s'] <= 7 and abs(suspend) < .01)
        (output / 'ssh-clock-check.json').write_text(json.dumps(result, indent=2) + '\n')
        if not result['passed']:
            raise RuntimeError('SSH wake-lock/clock precondition failed')
        lock = self.remote_root + '/model.lock'
        acquire = ('mkdir ' + shlex.quote(lock) + ' && echo ' +
                   shlex.quote(output.name) + ' > ' + shlex.quote(lock + '/owner'))
        if resume_lock:
            acquire = ('if [ -d ' + shlex.quote(lock) + ' ]; then test "$(cat ' +
                       shlex.quote(lock + '/owner') + ')" = ' + shlex.quote(output.name) +
                       '; else ' + acquire + '; fi')
        self.call('shell', acquire)
        self.lock_owner = output.name
        return result

    def release_lock(self):
        if self.kind == 'ssh' and self.lock_owner:
            lock = self.remote_root + '/model.lock'
            self.call('shell', 'if [ "$(cat ' + shlex.quote(lock + '/owner') +
                      ')" = ' + shlex.quote(self.lock_owner) + ' ]; then rm -rf ' + shlex.quote(lock) + '; fi')
            self.lock_owner = None
