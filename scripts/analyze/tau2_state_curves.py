#!/usr/bin/env python3
"""汇总状态成本及 H 敏感性，不以数值相似性筛选耗时。"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def fit(xs,ys):
    xbar,ybar=statistics.mean(xs),statistics.mean(ys)
    slope=sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))/sum((x-xbar)**2 for x in xs)
    intercept=ybar-slope*xbar
    errors=[y-(intercept+slope*x) for x,y in zip(xs,ys)]
    return dict(intercept_ms=intercept,slope_ms_per_token=slope,
                max_absolute_residual_ms=max(abs(e) for e in errors),
                max_relative_residual=max(abs(e)/y for e,y in zip(errors,ys)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/20261112-tau2-state-curves'))
    a=p.parse_args();out=a.input
    assert json.loads((out/'status.json').read_text())['phase']=='complete'
    state=[];stages=[]
    for path in sorted((out/'measurements').glob('*.json')):
        if '.raw.' in path.name:continue
        data=json.loads(path.read_text());response=data['response']
        if response['mode']=='state_cost_sweep_v1':
            state.extend(r for r in response['rows'] if not r['warmup'])
        else:
            stages.extend(dict(r,numerical_match=response['logit_comparison']['passed']) for r in response['rows'])
    summary=[]
    for h in sorted({r['H'] for r in state}):
        rows=[r for r in state if r['H']==h]
        summary.append(dict(H=h,repetitions=len(rows),snapshot_bytes=rows[0]['snapshot_bytes'],
            **{key:statistics.median(r[key] for r in rows) for key in
               ['snapshot_ms','prepare_ms','prepare_import_work_ms','prepare_verification_ms','htp_to_cpu_ms','cpu_restore_ms']}))
    fits={key:fit([r['H'] for r in summary],[r[key] for r in summary]) for key in ['snapshot_ms','prepare_ms','htp_to_cpu_ms','cpu_restore_ms']}
    prefill=[]
    for backend in ['cpu','htp']:
        for h in sorted({r['H'] for r in stages}):
            for n in [31,40,64,128]:
                rows=[r for r in stages if r['backend']==backend and r['H']==h and r['P']==n and not r['decode_checkpoints']]
                prefill.append(dict(backend=backend,H=h,P=n,repetitions=len(rows),
                    prefill_median_ms=statistics.median(r['prefill_ms'] for r in rows),
                    numerical_matches=sum(r['numerical_match'] for r in rows)))
    result=dict(context_size=8192,cpu_threads=6,htp_host_threads=4,state=summary,linear_fits=fits,prefill=prefill,
                scope='长度受控独立微基准；状态恢复和迁移包含现有生产核验成本；性能不按数值相似性筛选；不证明任务质量或泛化至其他context')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (out/'state-costs.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (out/'prefill-h-sensitivity.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(prefill[0]));writer.writeheader();writer.writerows(prefill)
    lines=['# 状态成本与 H 敏感性','',
        'Qwen3.5-9B Q4_0，context=8192，CPU6线程，HTP主机4线程。每个状态点预热一次、正式5次；下表为中位数。','',
        '| H | 快照导出 ms | CPU→HTP 准备 ms | HTP→CPU 返回迁移 ms | CPU 恢复 ms |',
        '|---:|---:|---:|---:|---:|']
    lines.extend(f"| {r['H']} | {r['snapshot_ms']:.2f} | {r['prepare_ms']:.2f} | {r['htp_to_cpu_ms']:.2f} | {r['cpu_restore_ms']:.2f} |" for r in summary)
    lines.extend(['','准备是从已保存快照恢复，返回迁移包含源状态导出；二者不能混用同一个 M(H)。返回迁移测量状态长度为 H+40。',
        '所有成本包含当前运行时的完整字节核验；导入工作与核验另有原始拆分。模型初始化、历史建立和夹具恢复分别记录。',
        'H 敏感性比较使用相同来源 token 的长度受控切片，并非完全相同文本；数字不用于回答质量验收。',
        '线性拟合仅作为本轮范围内的摘要，原始点及残差见 summary.json。16K context 的正式 pilot 需要独立复核。'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
