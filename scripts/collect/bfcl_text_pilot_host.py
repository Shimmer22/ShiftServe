#!/usr/bin/env python3
"""在 wins 宿主串行启动已有容器中的文本采集服务，保留日志并只清理本次服务。"""
import fcntl
import json
import pathlib
import subprocess
import time
import urllib.request

BASE = pathlib.Path('/home/wins/ai-workspace')
OUT = BASE/'outputs/a2o-gap-profiling/20261108-bfcl-text-pilot'
PRIVATE = '/workspace/private/a2o-text-pilot'
REMOTE_OUT = '/workspace/outputs/a2o-gap-profiling/20261108-bfcl-text-pilot'

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    lock = (BASE/'private/a2o-text-pilot/inference.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if busy.strip():
        raise RuntimeError('CUDA occupied; refusing contention')
    prefix = ['docker','exec','a2o-bfcl-lc200']
    command = prefix + ['python3','-m','vllm.entrypoints.openai.api_server',
        '--model','/workspace/cache/pvst-vllm/qwen35-9b-bf16-20261016',
        '--served-model-name','qwen35-9b','--host','127.0.0.1','--port','18767',
        '--dtype','bfloat16','--language-model-only','--max-model-len','32768',
        '--max-num-seqs','1','--gpu-memory-utilization','0.90','--enforce-eager']
    (OUT/'server-command.json').write_text(json.dumps(command,indent=2))
    server_log = (OUT/'server.log').open('w')
    started = time.monotonic()
    server = subprocess.Popen(command,stdout=server_log,stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('server exited during initialization')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18767/health',timeout=2) as r:
                    if r.status == 200: break
            except Exception: time.sleep(2)
        else: raise TimeoutError('server readiness')
        (OUT/'initialization.json').write_text(json.dumps({'model_server_ready_s':time.monotonic()-started,'vision':'disabled','backend':'CUDA0','state_migration':'not exercised; host characterization'}))
        with (OUT/'collector.log').open('w') as log:
            subprocess.run(prefix + ['python3',PRIVATE+'/scripts/collect/bfcl_vllm.py',
                '--config',PRIVATE+'/configs/bfcl-text-pilot-qwen35.json','--output',REMOTE_OUT],stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:
        # Identify and signal only the API server with this private port.
        code = "import os,signal; from pathlib import Path; "
        code += "\nfor p in Path('/proc').iterdir():\n if p.name.isdigit():\n  try:\n   a=(p/'cmdline').read_bytes().split(b'\\0')\n   if b'vllm.entrypoints.openai.api_server' in a and b'18767' in a: os.kill(int(p.name),signal.SIGTERM)\n  except (OSError,ProcessLookupError): pass"
        subprocess.run(prefix+['python3','-c',code],check=False)
        try: server.wait(timeout=45)
        except subprocess.TimeoutExpired: server.terminate()
        server_log.close()

if __name__ == '__main__': main()
