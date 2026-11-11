#!/usr/bin/env python3
"""复现GGML通用矩阵乘法的列切块规则，对照CPU prefill锯齿。"""
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/20261111-prefill-p24-128-geometric-cpu6-t10'
OUT = ROOT / 'results/20261111-cpu-prefill-sawtooth-audit'
OUT.mkdir(exist_ok=True)


def partition(p):
    """主导投影：nr0是足够大的偶数，ne11=P，ne2=ne3=1，常规chunk16。"""
    chunks = math.ceil(p / 16)
    width = math.ceil(p / chunks)
    lengths = [min(width, p-i*width) for i in range(chunks) if i*width < p]
    # 源码先检查全局ne11奇偶；偶数P再检查每个chunk的实际列数。
    single = p if p % 2 else sum(n for n in lengths if n % 2)
    return chunks, width, lengths, single


rows = [r for r in json.loads((SOURCE / 'figures-cleaned/curve-data.json').read_text()) if r['backend'] == 'cpu']
X, y = [], []
for row in rows:
    p = row['P']
    chunks, width, lengths, single = partition(p)
    X.append([1, p, single]);y.append(row['mean_s'])
    row.update(chunks=chunks, nominal_chunk_width=width, chunk_lengths=lengths,
               single_row_tokens=single, single_row_fraction=single/p)
X, y = np.array(X), np.array(y)
beta = np.linalg.lstsq(X, y, rcond=None)[0]
pred = X @ beta
mask = X[:, 1] <= 48
train_beta = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
for row, estimate in zip(rows, pred):
    row.update(predicted_s=float(estimate), relative_error=float((estimate-row['mean_s'])/row['mean_s']))
summary = dict(
    mechanism='Q4_0 ARM I8MM双行vec_dot；全局ne11为奇数或chunk实际行/列数为奇数时回退单行',
    assumptions='主导Q4投影nr0足够大且为16倍数，ne11=P，ne2=ne3=1；未覆盖其他算子或NUMA分支',
    formula='time_s = intercept_s + fast_s_per_token*P + single_row_penalty_s_per_token*single_row_tokens',
    intercept_s=float(beta[0]), fast_s_per_token=float(beta[1]),
    single_row_penalty_s_per_token=float(beta[2]),
    full_single_row_s_per_token=float(beta[1]+beta[2]),
    fitted_r2=float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),
    fitted_mape=float(np.mean(np.abs(pred-y)/y)),
    held_out=dict(train_P='24..48', train_points=int(mask.sum()), test_points=int((~mask).sum()),
                  mape=float(np.mean(np.abs(X[~mask]@train_beta-y[~mask])/y[~mask]))),
    scope='基于源码推导形状特征的统计交叉验证，尚未做改变切块规则的设备A/B因果干预；不评价质量或E2E收益')
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
(OUT / 'shape-vs-time.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
fields = ['P', 'n', 'mean_s', 'stdev_s', 'chunks', 'nominal_chunk_width', 'single_row_tokens', 'single_row_fraction', 'predicted_s', 'relative_error']
with (OUT / 'shape-vs-time.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
print(json.dumps(summary, ensure_ascii=False, indent=2))
