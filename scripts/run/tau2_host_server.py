#!/usr/bin/env python3
"""wins 上持锁运行复用的 CUDA 服务；显式 stop 文件退出，只清理本次进程。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'server.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared = Path('/home/wins/ai-workspace/private/a2o-text-pilot/inference.lock').open('a')
    fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if busy.strip():
        raise RuntimeError('CUDA occupied; refusing contention')
    prefix = ['docker', 'exec', 'a2o-bfcl-lc200']
    command = prefix + ['python3', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', '/workspace/cache/pvst-vllm/qwen35-9b-bf16-20261016',
        '--served-model-name', 'qwen35-9b', '--host', '127.0.0.1', '--port', '18769',
        '--dtype', 'bfloat16', '--language-model-only', '--max-model-len', '32768',
        '--max-num-seqs', '4', '--max-num-batched-tokens', '2048',
        '--gpu-memory-utilization', '0.90', '--enforce-eager',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml',
        '--reasoning-parser', 'qwen3']
    (out / 'server-command.json').write_text(json.dumps(command, indent=2))
    started = time.monotonic()
    with (out / 'server.log').open('a') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError('server exited during initialization')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18769/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError('server readiness')
            (out / 'server-ready.json').write_text(json.dumps(dict(
                pid=os.getpid(), server_pid=server.pid, backend='CUDA0', concurrency=4,
                initialization_s=time.monotonic()-started, vision='disabled')))
            while not (out / 'stop-server').exists():
                if server.poll() is not None:
                    raise RuntimeError('model server exited')
                time.sleep(2)
        finally:
            # 端口与模块双重匹配；不停止其他 GPU 服务。
            code = """import os,signal
from pathlib import Path
for path in Path('/proc').iterdir():
 if path.name.isdigit():
  try:
   args=(path/'cmdline').read_bytes().split(b'\\0')
   if b'vllm.entrypoints.openai.api_server' in args and b'18769' in args:
    os.kill(int(path.name),signal.SIGTERM)
  except (OSError,ProcessLookupError): pass
"""
            subprocess.run(prefix + ['python3', '-c', code], check=False)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.terminate()


if __name__ == '__main__':
    main()
