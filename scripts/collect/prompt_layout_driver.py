#!/usr/bin/env python3
"""在 wins 独占 CUDA0 启动私有布局验证服务；原模型与手机实验保持独立。"""
import fcntl,json,os,subprocess,time,traceback,urllib.request
from pathlib import Path
BASE=Path('/home/wins/ai-workspace')
OUT=BASE/'outputs/a2o-gap-profiling/20261109-prompt-layout-host'
PRIVATE='/workspace/private/a2o-prompt-layout'


def save(name,value):
    p=OUT/name;temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(p)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'private/a2o-text-pilot/inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if busy.strip():raise RuntimeError('CUDA occupied; refusing contention')
    prefix=['docker','exec','-e','CUDA_VISIBLE_DEVICES=0','-e','VLLM_SERVER_DEV_MODE=1','a2o-bfcl-lc200']
    command=prefix+['python3','-m','vllm.entrypoints.openai.api_server','--model','/workspace/cache/pvst-vllm/qwen35-9b-bf16-20261016','--served-model-name','layout-qwen35','--host','127.0.0.1','--port','18769','--dtype','bfloat16','--language-model-only','--max-model-len','32768','--max-num-seqs','1','--gpu-memory-utilization','0.90','--enforce-eager','--enable-prefix-caching']
    save('server-command.json',command);save('driver-status.json',dict(phase='initializing',pid=os.getpid(),time=time.time(),backend='host CUDA0 BF16; not a phone backend comparison'))
    log=(OUT/'server.log').open('a');server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if server.poll() is not None:raise RuntimeError('server exited; inspect server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18769/health',timeout=2) as r:
                    if r.status==200:break
            except OSError:time.sleep(2)
        else:raise TimeoutError('service readiness')
        save('driver-status.json',dict(phase='collecting',pid=os.getpid(),time=time.time()))
        command=prefix+['python3',PRIVATE+'/scripts/collect/prompt_layout_host.py','--source','/workspace/outputs/a2o-gap-profiling/20261108-bfcl-text-pilot','--output','/workspace/outputs/a2o-gap-profiling/20261109-prompt-layout-host','--model','/workspace/cache/pvst-vllm/qwen35-9b-bf16-20261016']
        with (OUT/'collector.log').open('a') as collector:subprocess.run(command,stdout=collector,stderr=subprocess.STDOUT,check=True)
        save('driver-status.json',dict(phase='complete',time=time.time()))
    except Exception:
        save('driver-status.json',dict(phase='failed',time=time.time(),error=traceback.format_exc()));raise
    finally:
        code="""import os,signal
from pathlib import Path
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   a=(p/'cmdline').read_bytes().split(b'\\0')
   if b'vllm.entrypoints.openai.api_server' in a and b'18769' in a:os.kill(int(p.name),signal.SIGTERM)
  except (OSError,ProcessLookupError):pass
"""
        subprocess.run(prefix+['python3','-c',code],check=False)
        try:server.wait(timeout=45)
        except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=10)
        log.close()

if __name__=='__main__':main()
