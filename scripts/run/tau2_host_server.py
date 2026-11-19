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
    parser.add_argument('--container', default='a2o-bfcl-lc200')
    parser.add_argument('--model-path', default='/workspace/cache/pvst-vllm/qwen35-9b-bf16-20261016')
    parser.add_argument('--tokenizer')
    parser.add_argument('--gguf-plugin-dir')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--served-model-name', default='qwen35-9b')
    parser.add_argument('--port', type=int, default=18769)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--max-num-seqs', type=int, default=4)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--tool-call-parser', default='qwen3_xml')
    parser.add_argument('--reasoning-parser', default='qwen3')
    parser.add_argument('--language-model-only', action=argparse.BooleanOptionalAction, default=True)
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
    prefix = ['docker', 'exec']
    if args.gguf_plugin_dir:
        prefix += ['-e', f'PYTHONPATH={args.gguf_plugin_dir}']
    prefix.append(args.container)
    command = prefix + ['python3', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model_path,
        '--served-model-name', args.served_model_name, '--host', '127.0.0.1', '--port', str(args.port),
        '--dtype', args.dtype, '--max-model-len', str(args.max_model_len),
        '--max-num-seqs', str(args.max_num_seqs), '--max-num-batched-tokens', '2048',
        '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--enforce-eager',
        '--enable-auto-tool-choice', '--tool-call-parser', args.tool_call_parser,
        '--reasoning-parser', args.reasoning_parser]
    if args.tokenizer:
        command += ['--tokenizer', args.tokenizer]
    if args.gguf_plugin_dir:
        command += ['--quantization', 'gguf']
    if args.language_model_only:
        command.append('--language-model-only')
    (out / 'server-command.json').write_text(json.dumps(command, indent=2))
    started = time.monotonic()
    with (out / 'server.log').open('a') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError('server exited during initialization')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError('server readiness')
            (out / 'server-ready.json').write_text(json.dumps(dict(
                pid=os.getpid(), server_pid=server.pid, backend='CUDA0', concurrency=args.max_num_seqs,
                initialization_s=time.monotonic()-started, vision='disabled')))
            while not (out / 'stop-server').exists():
                if server.poll() is not None:
                    raise RuntimeError('model server exited')
                time.sleep(2)
        finally:
            # 端口与模块双重匹配；不停止其他 GPU 服务。
            code = f"""import os,signal
from pathlib import Path
for path in Path('/proc').iterdir():
 if path.name.isdigit():
  try:
   args=(path/'cmdline').read_bytes().split(b'\\0')
   if b'vllm.entrypoints.openai.api_server' in args and b'{args.port}' in args:
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
