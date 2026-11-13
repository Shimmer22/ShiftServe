#!/usr/bin/env python3
"""Qwen3稀疏单token耗时分解，内层算子不与外层执行段重复相加。"""
import json,re
from collections import Counter
from pathlib import Path
from decode_attribution import read_case
import npu_decode_fine as fine
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/20261113-qwen3-sparse-token-profile'

def category(op,names,output=''):
    if 'output.weight' in names.split() or output=='result_output':return 'output_head'
    if '.ffn_' in names or re.search(r'\bffn_',names):return 'ffn'
    if op=='SET_ROWS':return 'kv_write'
    if re.match(r'kq(?:-|v-|_soft_max-)',output) or op=='SOFT_MAX':return 'attention_core'
    if op=='MUL_MAT' and re.search(r'blk\.\d+\.attn_(q|k|v|output)\.weight',names):return 'qkvo_projection'
    if re.search(r'Qcur|Kcur|Vcur|kqv|cache_[kv]',names):return 'attention_other'
    return 'norm_residual_embedding_other'

def main():
    fine.category=category
    d,stats,rep=read_case(OUT/'token-fine');profiles=fine.dsp_rows(OUT/'token-fine/runtime.log');rows=[]
    for r in d['rows']:
        if r['warmup']:continue
        n=r['D'];cpu=Counter();splits=Counter();counts=Counter();details=[]
        for key,v in r['backend_timing'].items():
            if key.startswith('CPU_NODE/'):
                _,op,name=key.split('/',2);names=' '.join([name]+[s['name'] for s in v['metadata']['sources']]);group=category(op,names,name);cpu[group]+=v['us']/n/1000;details.append(dict(name=name,op=op,group=group,ms=v['us']/n/1000))
            else:
                group='cpu' if key.startswith('COMPUTE/CPU/') else 'htp' if key.startswith('COMPUTE/HTP') else 'copy' if key=='COPY' else 'sync'
                splits[group]+=v['us']/n/1000;counts[group]+=v['calls']/n
        dsp=profiles[r['repetition']];dsp_ms={k:v/n/1000 for k,v in dsp['us'].items()};wall=r['elapsed_ms']/n
        parts=dict(cpu);parts.update({'dsp_'+k:v for k,v in dsp_ms.items()});parts.update(cpu_dispatch_and_instrumentation=splits['cpu']-sum(cpu.values()),htp_host_wait_cache_profile=splits['htp']-sum(dsp_ms.values()),explicit_copy=splits['copy'],explicit_sync=splits['sync'],residual_host=wall-sum(splits.values()))
        assert abs(sum(parts.values())-wall)<.01
        rows.append(dict(repetition=r['repetition'],wall_ms=wall,token_ms=r['token_ms'],cpu_ms=dict(cpu),dsp_ms=dsp_ms,parts_ms=parts,outer_ms=dict(splits),calls_per_token=dict(counts),dsp_batches_per_token=dsp['batches']/n,cpu_details=sorted(details,key=lambda x:-x['ms']),major_faults=r['resources_after']['major_faults']-r['resources_before']['major_faults'],swap_kib=r['resources_after']['VmSwap_kib']))
    chosen=next(r for r in rows if r['repetition']==rep['repetition']);summary=dict(stats=stats,repeats=rows,representative=chosen,controls={})
    for name in ['control-before','control-after']:
        _,s,_=read_case(OUT/name);summary['controls'][name]=s
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(chosen,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
