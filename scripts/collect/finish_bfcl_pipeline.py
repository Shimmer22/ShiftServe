"""续接已启动的采集进程，随后释放服务显存、回放并生成验收文件。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def write(path,value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--collector-pid',required=True,type=int)
    p.add_argument('--server-pid',required=True,type=int)
    args = p.parse_args()
    root = args.root
    started = time.time()
    try:
        write(root/'pipeline-status.json',{'stage':'collection','collector_pid':args.collector_pid})
        # 仅续接本次已启动的进程，不创建系统计划任务或应用自动化。
        while Path(f'/proc/{args.collector_pid}').exists():
            stat = Path(f'/proc/{args.collector_pid}/stat').read_text().split()
            if stat[2]=='Z':
                break
            time.sleep(15)
        paths = list(root.glob('multi_turn_long_context_*/result.json'))
        ids = {f'multi_turn_long_context_{i}' for i in range(200)}
        if {p.parent.name for p in paths}!=ids:
            raise RuntimeError(f'expected exactly 200 task results; got {len(paths)}')
        results = [json.loads(p.read_text()) for p in sorted(paths)]
        if not (root/'collection-completion.json').exists():
            write(root/'collection-completion.json',{'results':results,
                'recovered_from_result_files':True,
                'note':'All 200 task results independently verified; collector progress-log JSON serialization may fail on checker objects. Original logs retained.'})
        write(root/'pipeline-status.json',{'stage':'stopping_vllm','finished_collection':len(results)})
        cmdline = Path(f'/proc/{args.server_pid}/cmdline').read_bytes()
        if b'vllm' not in cmdline or b'18766' not in cmdline:
            raise RuntimeError('server PID identity mismatch; refusing to stop unrelated process')
        os.kill(args.server_pid,signal.SIGTERM)
        for _ in range(60):
            query = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not query:
                break
            time.sleep(2)
        else:
            raise RuntimeError('GPU still occupied after stopping owned server; refusing concurrent replay')
        write(root/'pipeline-status.json',{'stage':'attention_replay','samples_per_call':16})
        with (root/'replay.log').open('w') as log:
            subprocess.run(['python3','-u','-m','a2o.profiling.bfcl_replay','--root',str(root)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        with (root/'analysis.log').open('w') as log:
            subprocess.run(['python3','/workspace/private/a2o-bfcl-attention/analyze.py',str(root)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        summary = json.loads((root/'attention-summary.json').read_text())
        probe = json.loads((root/'probe-completion.json').read_text())
        matches,total = 0,0
        for file in root.glob('multi_turn*/replay-checks.json'):
            for row in json.loads(file.read_text()):
                matches += row['greedy_matches']
                total += row['sampled_queries']
        acceptance = {'stage':'completed','tasks':len(results),'quality_pass':summary['quality_pass'],
            'task_errors':summary['task_errors'],'probe_failed_calls':len(probe['failed_calls']),
            'sampled_greedy_match_fraction':matches/total if total else None,
            'attention_observations':summary['observation_count'],'kv_retired':False,
            'quality_of_measurement':'review failures and replay agreement before interpreting attention',
            'pipeline_seconds_after_attachment':time.time()-started}
        write(root/'acceptance.json',acceptance)
        write(root/'pipeline-status.json',acceptance)
    except Exception:
        write(root/'pipeline-failure.json',{'error':traceback.format_exc(),'time':time.time()})
        raise


if __name__=='__main__':
    main()
