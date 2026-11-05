#!/usr/bin/env python3
"""串行主图任务；支持等首批结束后接续第二批并统一出图。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume-first',type=Path)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    state_path=out/'campaign.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    runs=[]
    def record(status, **fields):
        state.update(status=status,runs=[str(r) for r in runs],updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**fields)
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    try:
        if args.resume_first:
            first=args.resume_first.resolve();runs.append(first);record('waiting_first')
            while not ((first/'completion.json').exists() and (first/'device/trials.jsonl').exists()):
                if (first/'failure.json').exists():
                    raise RuntimeError('首批失败，拒绝启动第二批：'+str(first))
                time.sleep(2)
            if not json.loads((first/'completion.json').read_text())['passed']:
                raise RuntimeError('首批未通过')
            first_manifest=json.loads((first/'manifest.json').read_text())
            if first_manifest['config'] != json.loads((ROOT/'configs/observation-main-h2048.json').read_text()):
                raise RuntimeError('首批配置与当前计划不一致')
            histories=[4096]
        else:
            histories=[2048,4096]
        for h in histories:
            record('running',active_history=h)
            log_path=out/f'h{h}-campaign-console.log'
            with log_path.open('w') as log:
                code=subprocess.run([sys.executable,str(ROOT/'scripts/profile/oracle.py'),'--device',args.device,
                                     '--config',str(ROOT/f'configs/observation-main-h{h}.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            first_line=log_path.read_text().splitlines()[0]
            run=Path(first_line)
            if run.is_dir(): runs.append(run)
            if code or not (run/'completion.json').exists():
                raise RuntimeError(f'H{h} 失败，见 {log_path}')
        record('plotting',active_history=None)
        with (out/'plot-console.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/plot/observations.py'),*map(str,runs),
                            '--output',str(out/'figures')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        record('figures_rendered',figures=str(out/'figures'),notes='设备采样与原始记录复核完成；图仍须人工视觉检查。')
        print(json.dumps(state,ensure_ascii=False))
    except Exception as exc:
        record('failed',error=repr(exc))
        raise


if __name__=='__main__':main()
