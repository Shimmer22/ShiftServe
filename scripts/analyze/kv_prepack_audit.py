#!/usr/bin/env python3
"""定位预整理的算子收益与 CPU decode 长尾；不把诊断当成正式性能收益。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median


def audit(profile, regression, original, output, release=None, fixed=None):
    load_rows=lambda p:[json.loads(s) for s in (p/'checked-rows.jsonl').read_text().splitlines()]
    cfg=json.loads((profile/'manifest.json').read_text())['config']
    assert cfg['calibration_only'] and cfg['htp_profile']==1
    rows=[r for r in load_rows(profile) if r['route']=='HC']
    calls=[];current=[]
    for line in (profile/'runtime.log').read_text().splitlines():
        if 'profile-op ' not in line:continue
        current.append(line)
        if '-> result_norm :' in line:calls.append(current);current=[]
    assert not current and len(calls)==len(rows)==3
    operators=[]
    for row,call in zip(rows,calls):
        attention=[line for line in call if 'profile-op MUL_MAT:' in line and 'f16 x f32' in line]
        assert len(attention)==72 and row['passed'] and row['numerical_passed']
        operators.append({'trial_id':row['id'],'prepack':row['kv_prepack_enabled'],
            'op_records':len(call),'attention_op_records':len(attention),
            'attention_ms':sum(int(re.search(r'usec (\d+)',s)[1]) for s in attention)/1000,
            **{k:row[k] for k in ('kv_prepack_ms','prefill_ms','decode_ms')}})
    def run_groups(directory):
        fresh=load_rows(directory)
        completion=json.loads((directory/'completion.json').read_text())
        assert completion['passed'] and len(fresh)==30
        keys=('response_after_return_ms','prefill_ms','handoff_checked_ms','decode_ms','decode_cpu_ms',
              'decode_minor_faults','decode_major_faults','decode_voluntary_switches','decode_involuntary_switches')
        groups=[]
        for gap in (0,1500):
            for strategy in ('after_return','always'):
                for pack in (False,True):
                    part=[r for r in fresh if r['phase']=='measurement' and (r['gap_requested_ms'],r['strategy'],r['kv_prepack_enabled'])==(gap,strategy,pack)]
                    assert len(part)==3 and all(r['passed'] for r in part)
                    groups.append({'G':gap,'strategy':strategy,'prepack':pack,'ids':[r['id'] for r in part],
                        'ready_count':sum(r['ready_at_return'] for r in part),
                        'median':{k:median(r[k] for r in part) for k in keys},
                        'decode_major_faults_total':sum(r['decode_major_faults'] for r in part)})
        measured=[r for r in fresh if r['phase']=='measurement']
        return {'groups':groups,'trials':len(fresh),'max_response_ms':max(r['response_after_return_ms'] for r in measured),
                'max_decode_major_faults':max(r['decode_major_faults'] for r in measured),
                'by_prepack':[{'prepack':pack,'decode_median_ms':median(r['decode_ms'] for r in measured if r['kv_prepack_enabled']==pack),
                              'decode_major_faults_median':median(r['decode_major_faults'] for r in measured if r['kv_prepack_enabled']==pack)} for pack in (False,True)]}
    held=run_groups(regression)
    old=[r for r in load_rows(original) if r['phase']=='measurement' and r['gap_requested_ms']==1500 and r['strategy']=='always']
    result={'diagnostic_only':True,'profile_calls':operators,'regression_groups':held['groups'],'held':held,
            'original_ready_long_gap_trials':[{k:r[k] for k in ('id','P','kv_prepack_enabled','ready_at_return','prefill_ms','handoff_checked_ms','decode_ms','response_after_return_ms')} for r in old],
            'sources':{name:str(p.resolve()) for name,p in [('profile',profile),('regression',regression),('original',original)]},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name,directory in [('release_only',release),('memory_fixed',fixed)]:
        if directory:
            result[name]=run_groups(directory)
            result['sources'][name]=str(directory.resolve())
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',type=Path);p.add_argument('regression',type=Path);p.add_argument('original',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--release',type=Path);p.add_argument('--fixed',type=Path)
    a=p.parse_args();result=audit(a.profile,a.regression,a.original,a.output,a.release,a.fixed)
    print(json.dumps(result['regression_groups'],ensure_ascii=False))
