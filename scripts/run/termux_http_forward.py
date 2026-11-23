#!/usr/bin/env python3
"""运行手机实验入口，恢复断开的 Termux SSH 转发并防止 Mac 空闲休眠。"""

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport


def run(port, command):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
    device = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    ssh = [*device.prefix[:-1], '-N', '-o', 'ExitOnForwardFailure=yes',
           '-L', f'127.0.0.1:{port}:127.0.0.1:18780', device.target]
    env = dict(os.environ, A2O_PHONE_HTTP_BASE=f'http://127.0.0.1:{port}')
    forward = subprocess.Popen(ssh, stdin=subprocess.DEVNULL)
    child = None
    try:
        child = subprocess.Popen(command, env=env, cwd=ROOT)
        while True:
            try:
                return child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if forward.poll() is not None:
                    print(f'Termux SSH 转发退出 ({forward.returncode})，重新连接', flush=True)
                    forward = subprocess.Popen(ssh, stdin=subprocess.DEVNULL)
    finally:
        try:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                child.wait(timeout=60)
        finally:
            if forward.poll() is None:
                forward.terminate()
            forward.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=28781)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 1 <= args.port <= 65535:
        parser.error('需要合法的转发端口和实验入口命令')
    if sys.platform == 'darwin':
        subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                         stdin=subprocess.DEVNULL)
    raise SystemExit(run(args.port, command))


if __name__ == '__main__':
    main()
