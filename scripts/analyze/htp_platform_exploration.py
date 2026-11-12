#!/usr/bin/env python3
"""核对DSP直接路由及量化HMX实际mc/nc，正式耗时独立统计。"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.validation.htp_layout import parse_layout_line

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
records=[];by_point=[]
for f in sorted((a.diagnostic/'measurements').glob('r*.json')):
    if '.raw.' in f.name:continue
    d=json.loads(f.read_text());point=len(d['request']['input_ids']);trace=f.parent/d['profile_trace']['file']
    data=trace.read_bytes();assert hashlib.sha256(data).hexdigest()==d['profile_trace']['sha256']
    lines=data.decode().splitlines();ops=[r for line in lines if (r:=parse_layout_line(line))]
    assert ops and len(ops)==sum('profile-op MUL_MAT:' in line for line in lines),'探针未覆盖MUL_MAT'
    assert not any(r['reason']=='hmx_error' for r in ops)
    for r in ops:r['P']=point;r['round']=d['round']
    records.extend(ops)
    counter=collections.Counter(r['path'] for r in ops)
    usec=collections.Counter()
    for r in ops:usec[r['path']]+=r['usec']
    layouts=collections.Counter((r['M'],r['N'],r['K'],r['weight_type'],r['mc'],r['nc'],r['layout']) for r in ops if r['mc'] and r['nc'])
    all_times=collections.Counter()
    for line in lines:
        match=re.search(r'profile-op\s+(\w+):.*?\susec\s+(\d+)\s+cycles',line)
        if match:all_times[match[1]]+=int(match[2])
    batches=[tuple(map(int,m.groups())) for m in re.finditer(r'profile-batch n-ops (\d+) batch-dur-usec (\d+) htp-ops-usec (\d+)',data.decode())]
    wall=next(r['prefill_ms']/1000 for r in d['response']['rows'] if r['backend']=='htp')
    by_point.append(dict(P=point,matrix_ops=len(ops),path_counts=dict(counter),matrix_usec_by_path=dict(usec),
        diagnostic_wall_s=wall,dsp_operator_s={k:v/1e6 for k,v in all_times.items()},
        dsp_batch_count=len(batches),dsp_batch_total_s=sum(b[1] for b in batches)/1e6,
        delta_cont_example=next((l for l in lines if 'profile-op CONT: key_gdiff-0 (transposed)' in l),None),
        solve_tri_example=next((l for l in lines if 'profile-op SOLVE_TRI:' in l),None),
        layouts=[dict(M=k[0],N=k[1],K=k[2],weight_type=k[3],mc=k[4],nc=k[5],layout=k[6],
                      row_tiles=math.ceil(k[0]/32),mblocks=math.ceil(k[0]/k[4]),nblocks=math.ceil(k[1]/k[5]),count=n) for k,n in sorted(layouts.items())]))
assert records
(a.output/'operator-layouts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
result=dict(diagnostic_only=True,hardware_pmu_values_valid=False,matrix_ops=len(records),points=by_point)
if a.baseline and (a.baseline/'summary.json').exists():
    rows=list(csv.DictReader((a.baseline/'stage-times.csv').open()));means=[]
    for point in sorted({int(r['P']) for r in rows}):
        entry=dict(P=point)
        for backend in ['cpu','htp']:
            values=[float(r['elapsed_ms'])/1000 for r in rows if r['backend']==backend and int(r['P'])==point]
            assert len(values)==10
            entry[backend]=dict(n=10,mean_s=statistics.mean(values),median_s=statistics.median(values),stdev_s=statistics.stdev(values),raw_s=values)
        means.append(entry)
    result['unprofiled_baseline']=means
(a.output/'direct-layout-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
report=['# HTP平台边界探索','','## 直接诊断证据','',
        '私有v75 DSP布局探针；每点一次，仅定位计算路径与内部布局，耗时不混入原版十轮结果。PMU槽是探针元数据，不作为硬件计数。','',
        '| P | HVX矩阵数 | HMX矩阵数 | DSP提交批次数 | CONT累计秒 | 矩阵累计秒 |',
        '|---:|---:|---:|---:|---:|---:|']
for r in by_point:
    report.append(f"| {r['P']} | {r['path_counts'].get('HVX',0)} | {r['path_counts'].get('HMX',0)} | {r['dsp_batch_count']} | {r['dsp_operator_s'].get('CONT',0):.4f} | {r['dsp_operator_s'].get('MUL_MAT',0):.4f} |")
report += ['','## 机制解释','',
           '31/32边界：主导矩阵算子从HVX切换为HMX。64/65边界：主导矩阵仍为HMX；量化矩阵mc从64变为96，部分nc收缩，硬件tile数增加。',
           '同时，Qwen3.5 DeltaNet的chunk size是64，65个token将内部张量补到128、n_chunks从1变为2。日志中的key_gdiff_t与SOLVE_TRI张量直接显示第三维从1变2，CONT时间与若干其他算子增加，DSP提交批次多96。',
           '第二次台阶不能仅归因于HVX/HMX切换或HMX tile；内部模型分块、重排与调度共同参与。尚未通过改变DeltaNet chunk size或HMX布局做单因素A/B。',
           '','## 原版库同批次十次扫描','']
if 'unprofiled_baseline' in result:
    report += ['| P | CPU均值秒 | HTP均值秒 | HTP中位数秒 |','|---:|---:|---:|---:|']
    for r in result['unprofiled_baseline']:report.append(f"| {r['P']} | {r['cpu']['mean_s']:.4f} | {r['htp']['mean_s']:.4f} | {r['htp']['median_s']:.4f} |")
else:
    report += ['原版库P60至72，每点每后端10次的同批次乱序测量尚未完成。不能用上述一次诊断耗时替代正式比较。']
(a.output/'README.md').write_text('\n'.join(report)+'\n')
print(json.dumps(dict(points=[r['P'] for r in by_point],matrix_ops=len(records),baseline_complete='unprofiled_baseline' in result),ensure_ascii=False))
