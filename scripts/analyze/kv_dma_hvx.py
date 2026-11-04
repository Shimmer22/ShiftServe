#!/usr/bin/env python3
"""汇总 DMA/HVX tiles 对照；算子 profile 独立标注，不将跨轮阶段差值当成因果收益。"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
from statistics import median

from kv_prepack import analyze


def profile_calls(directory):
    directory=Path(directory)
    cfg=json.loads((directory/'manifest.json').read_text())['config']
    assert cfg['calibration_only'] and cfg['htp_profile']==1
    assert json.loads((directory/'completion.json').read_text())['passed']
    rows=[json.loads(s) for s in (directory/'checked-rows.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['route']=='HC']
    calls=[];current=[]
    for line in (directory/'runtime.log').read_text().splitlines():
        if 'profile-op ' not in line:continue
        current.append(line)
        if '-> result_norm :' in line:calls.append(current);current=[]
    assert not current and len(calls)==len(rows)
    output=[]
    for row,call in zip(rows,calls):
        attention=[line for line in call if 'profile-op MUL_MAT:' in line and 'f16 x f32' in line]
        assert len(attention)==72 and row['passed'] and row['numerical_passed']
        output.append({'id':row['id'],'P':row['P'],'prepack':row['kv_prepack_enabled'],
                       'attention_ms':sum(int(re.search(r'usec (\d+)',s)[1]) for s in attention)/1000})
    layouts=[]
    for match in re.finditer(r'a2o-kv-layout: vtcm=(\d+) K_chunks=(\d+) V_chunks=(\d+) K_nc=(\d+) V_nc=(\d+)',(directory/'runtime.log').read_text()):
        layouts.append(dict(zip(('vtcm_bytes','K_chunks','V_chunks','K_ncols','V_ncols'),map(int,match.groups()))))
    return {'directory':str(directory.resolve()),'diagnostic_only':True,'calls':output,'layouts':layouts}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('measurement',type=Path)
    parser.add_argument('--profile',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checked=analyze(args.measurement)
    rows=[json.loads(s) for s in (args.measurement/'checked-rows.jsonl').read_text().splitlines()]
    measurement=[r for r in rows if r['phase']=='measurement']
    groups=[]
    for comparison in checked['comparisons']:
        p,g=comparison['P'],comparison['G_target_ms']
        cells={pack:[r for r in measurement if (r['P'],r['gap_requested_ms'],r['strategy'],r['kv_prepack_enabled'])==(p,g,'always',pack)] for pack in (False,True)}
        reference={r['repetition']:r for r in cells[False]}
        diffs=[reference[r['repetition']]['response_after_return_ms']-r['response_after_return_ms'] for r in cells[True]]
        # 小样本描述：按 repetition 配对的中位数 bootstrap，仅 n<=6 时精确枚举。
        assert len(diffs)<=6
        boot=sorted(median(sample) for sample in itertools.product(diffs,repeat=len(diffs)))
        metrics=('prepare_ms','kv_prepack_ms','prefill_ms','handoff_checked_ms','decode_ms','kv_prepack_release_ms','response_after_return_ms','decode_major_faults')
        groups.append({**comparison,'paired_differences_ms':diffs,
            'paired_median_bootstrap_95_interval_ms':[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]],
            'cells':[{'prepack':pack,'ids':[r['id'] for r in cells[pack]],'ready_count':sum(r['ready_at_return'] for r in cells[pack]),
                      'median':{key:median(r[key] for r in cells[pack]) for key in metrics}} for pack in (False,True)]})
    result={'passed':checked['passed'],'measurement':str(args.measurement.resolve()),'comparisons':groups,
            'profiles':[profile_calls(p) for p in args.profile],
            'scope':'固定 Qwen3B 历史/输入；真实 HTTP，hybrid HTP prefill + CPU decode；小样本随机对照；profile 不并入正式计时',
            'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
