#!/usr/bin/env python3
"""分开复核原版计时、主机分解、DSP路由及受控干预，避免重复计时。"""
from array import array
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.validation.htp_layout import parse_layout_line
OUT=ROOT/'results/20261113-decode-attribution'
FOLLOW=ROOT/'results/20261113-decode-attribution-followup'


def read_case(folder):
    d=json.loads((folder/'response.json').read_text())
    assert d['completed'] and d['normal_response']['passed'] and d['state_restore_verified']
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    r=[v for v in d['rows'] if not v['warmup']]
    assert len(r)==d['config']['rounds']
    for v in r:
        assert math.isfinite(v['elapsed_ms']) and v['elapsed_ms']>0
        assert v['logits_vs_first_repeat']['cosine']>=0.99
    per_token=[v['elapsed_ms']/v['D'] for v in r]
    representative=min(r,key=lambda v:abs(v['elapsed_ms']/v['D']-median(per_token)))
    return d,dict(ms_per_token=median(per_token),repetitions_ms_per_token=per_token,
        graph_reused=[v['graph_reused'] for v in r],D=r[0]['D']),representative


def host_summary(r):
    total=defaultdict(float);calls=Counter();byte_counts=Counter();details=[]
    for k,v in r['backend_timing'].items():
        group='cpu_compute' if k.startswith('COMPUTE/CPU/') else 'htp_call' if k.startswith('COMPUTE/HTP') else 'copy' if k=='COPY' else 'explicit_sync'
        total[group]+=v['us']/r['D']/1000;calls[group]+=v['calls']/r['D'];byte_counts[group]+=v['bytes']/r['D']
        details.append(dict(group=k,ms_per_token=v['us']/r['D']/1000,calls_per_token=v['calls']/r['D']))
    elapsed=r['elapsed_ms']/r['D'];remaining=elapsed-sum(total.values());assert remaining>=-0.1
    return dict(wall_ms_per_token=elapsed,parts_ms=dict(total),calls_per_token=dict(calls),copy_MiB_per_token=byte_counts['copy']/1048576,
        residual_host_ms=remaining,split_groups=sorted(details,key=lambda v:-v['ms_per_token']))


def numerical(folder,reference,meta):
    values=array('f');values.frombytes((folder/'logits.f32').read_bytes());assert len(values)==len(reference)
    n=meta['vocab'];rows=[]
    for t in range(meta['tokens']):
        a=reference[t*n:(t+1)*n];b=values[t*n:(t+1)*n]
        assert all(math.isfinite(v) for v in b)
        aa=sum(x*x for x in a);bb=sum(y*y for y in b);ab=sum(x*y for x,y in zip(a,b));err=sum((x-y)**2 for x,y in zip(a,b))
        rows.append(dict(position=t,cosine=ab/math.sqrt(aa*bb),nrmse=math.sqrt(err/aa),
            max_abs=max(abs(x-y) for x,y in zip(a,b)),argmax_match=max(range(n),key=a.__getitem__)==max(range(n),key=b.__getitem__)))
    return rows


def main():
    initial={};full={}
    for case in json.loads((ROOT/'configs/decode-attribution.json').read_text())['cases']:
        d,s,r=read_case(OUT/case['name']);initial[case['name']]=s;full[case['name']]=(d,r)
    follow={};host={};comparisons={}
    baseline=json.loads((FOLLOW/'baseline/response.json').read_text())
    reference=array('f');reference.frombytes((FOLLOW/'baseline/logits.f32').read_bytes())
    for case in json.loads((ROOT/'configs/decode-attribution-followup.json').read_text())['cases']:
        label=case['name'];d,s,r=read_case(FOLLOW/label);follow[label]=s;host[label]=host_summary(r)
        comparisons[label]=numerical(FOLLOW/label,reference,d['numerical_replay'])
    cpu_log=(OUT/'cpu-reuse/runtime.log').read_text();htp_log=(OUT/'htp-reuse/runtime.log').read_text();fused_log=(FOLLOW/'cpu-fused-gdn/runtime.log').read_text()
    assert 'fused Gated Delta Net (autoregressive) enabled' in cpu_log
    assert 'fused Gated Delta Net (autoregressive) not supported, set to disabled' in htp_log
    assert 'explicitly retaining CPU fused autoregressive GDN' in fused_log
    assert 'fused Gated Delta Net (autoregressive) enabled' in fused_log
    assert 'fused Gated Delta Net (chunked) not supported, set to disabled' in fused_log
    assert all('CPY,GATED_DELTA_NET' in ' '.join(v['backend_timing']) for v in json.loads((FOLLOW/'cpu-fused-gdn/response.json').read_text())['rows'])
    active=False;tokens=0;ops=Counter();usecs=Counter();routes=Counter();reasons=Counter();batch_host=0;batch_dsp=0;batches=0
    for line in (OUT/'htp-dsp-route/runtime.log').read_text().splitlines():
        if 'A2O_DECODE_BEGIN' in line:
            active='rep=-1' not in line
            if active:tokens+=full['htp-dsp-route'][0]['rows'][0]['D']
        if 'A2O_DECODE_END' in line:active=False
        if not active:continue
        m=re.search(r'profile-op (.*?): (.*?) : .*?usec (\d+) cycles',line)
        if m:
            op,name,us=m[1],m[2],int(m[3]);key='FFN_MUL_MAT' if op=='MUL_MAT' and '.ffn_' in name else 'OTHER_MUL_MAT' if op=='MUL_MAT' else op
            ops[key]+=1;usecs[key]+=us
        route=parse_layout_line(line)
        if route:routes[route['path']]+=1;reasons[route['reason']]+=1
        m=re.search(r'profile-batch n-ops (\d+) batch-dur-usec (\d+) htp-ops-usec (\d+)',line)
        if m:batches+=1;batch_host+=int(m[2]);batch_dsp+=int(m[3])
    assert tokens>0 and sum(ops.values())==batches and routes['HMX']==0
    dsp=dict(tokens=tokens,batches_per_token=batches/tokens,ops_per_token={k:v/tokens for k,v in ops.items()},
        op_ms_per_token={k:v/tokens/1000 for k,v in usecs.items()},routes_per_token={k:v/tokens for k,v in routes.items()},
        route_reasons=dict(reasons),batch_response_ms_per_token=batch_host/tokens/1000,dsp_ops_ms_per_token=batch_dsp/tokens/1000,
        scope='私有DSP路由探针、profile=2；计数与PMU槽是路由元数据；此运行墙钟不作为原版速度')
    main_host=host_summary(full['htp-host-timing'][1]);base=follow['baseline']['ms_per_token']
    result=dict(initial=initial,host_attribution=main_host,dsp=dsp,followup=follow,followup_host=host,numerical=comparisons,
        interventions={k:dict(ms_per_token=follow[k]['ms_per_token'],speedup=base/follow[k]['ms_per_token'],reduction=1-follow[k]['ms_per_token']/base)
            for k in ['cpu-fused-gdn','batched-submit','host-one-thread']},
        scope='H4096/P40，Qwen3.5-9B Q4_0，固定token；CPU6线程，HTP主机4线程（host-one-thread为1）；无频率/设置修改；不代表硬件极限或任务E2E')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    m=main_host
    lines=['# NPU decode慢在哪里：实机归因', '',
        '结论：当前HTP路径是大量CPU/HTP交错执行，CPU融合DeltaNet被设备一致性检查关闭，并且逐算子提交。单token的DSP矩阵乘全部走HVX。当前瓶颈不能归结为FFN神经元太多或NPU硬件本身慢。', '',
        'Qwen3.5-9B Q4_0、小米14 Pro；H4096/P40、context8192、batch128、F16状态，CPU6线程、HTP主机4线程；每条件预热一次，正式3次、每次4个token。DSP路由诊断单独测1次×2个token。复用逐token、模型身份、上下文和校验和均核验的完整历史快照。prefill与decode复用一个上下文；恢复包含完整状态字节核验。仅Termux SSH持锁串行运行，无KV整理、预搬移、频率或设置修改，不执行生成动作。', '',
        '## 原版时延与图复用', '',
        '| 条件 | ms/token，中位数 |', '|---|---:|']
    for label,title in [('cpu-reuse','CPU，图复用开'),('htp-reuse','HTP，图复用开'),('htp-rebuild','HTP，图复用关')]:lines.append(f"| {title} | {initial[label]['ms_per_token']:.2f} |")
    lines += ['', '实测n_reused分别确认复用开时每个token复用、关时为0。关闭图复用有开销，但不足以解释CPU/HTP差距。各条件独立短进程，跨条件仍有运行波动，所有重复保留。', '',
        '## 主机侧分解', '',
        '私有ggml-base仅在原函数入口/出口计时，不改变算法、后端选择，不插入额外逐节点同步。CPU与HTP当前graph_compute均同步执行。拷贝嵌套和计算内部同步排除重复计数。下表取正式墙钟中位数对应的同一次重复，非把各分类独立中位数相加。', '',
        '| 部分 | ms/token |', '|---|---:|',f"| CPU计算split（186段） | {m['parts_ms']['cpu_compute']:.2f} |",
        f"| HTP执行调用（185段，含等待） | {m['parts_ms']['htp_call']:.2f} |",f"| 跨后端拷贝调用 | {m['parts_ms']['copy']:.2f} |",
        f"| 显式同步调用 | {m['parts_ms']['explicit_sync']:.2f} |",f"| 剩余主机工作与计时开销 | {m['residual_host_ms']:.2f} |",f"| 总墙钟 | {m['wall_ms_per_token']:.2f} |", '',
        'HTP执行调用包含DSP工作、队列/提交/缓存处理与等待，不能把整项当成DSP算子纯计算。CPU split按所含算子集合归组，混合split时间不冒充单算子独占时间。详细明细见summary.json。', '',
        '## DSP实际做了什么', '',
        f"每token {dsp['batches_per_token']:.0f}次批次提交，当前OPBATCH=1、OPQUEUE=1；{dsp['routes_per_token']['HVX']:.0f}次DSP矩阵乘均为HVX，原因M_lt_32，HMX为0。逐token的输入行数M=1，当前实现没有使用HMX矩阵路径。", '',
        f"独立DSP诊断：算子内部累计{dsp['dsp_ops_ms_per_token']:.2f} ms/token，其中FFN矩阵乘{dsp['op_ms_per_token']['FFN_MUL_MAT']:.2f} ms、其他矩阵乘{dsp['op_ms_per_token']['OTHER_MUL_MAT']:.2f} ms；批次提交至响应累计{dsp['batch_response_ms_per_token']:.2f} ms。两种计时之差包含队列、传输、缓存维护、调度和等待，不能精确归因为某一种开销。", '',
        '该profile运行与主机分解不是同一条测量，不能跨运行相减或拼成严格的端到端分解。路由元数据来自私有v75 DSP探针，原版库身份保留。', '',
        '## 为什么CPU部分特别慢', '',
        '同一运行时的CPU路径启用融合Gated Delta Net；HTP层上的融合节点被分配到CPU后，llama-context.cpp的设备一致性检查关闭融合AR/CH路径，解体为多个小算子。原路径每token交错186段CPU、185段HTP计算，张量拷贝约195.5 MiB。', '',
        'CPU单独的矩阵乘split约42–44 ms/token；加载日志确认输出投影output.weight为Q6_K，当前HTP不支持该权重类型，留在CPU。标准attention也因当前offload_kqv=false保留CPU部分。不得把NPU路径误写为所有算子都在NPU运行。', '',
        '## 受控干预', '',
        '仅私有进程对照：保留CPU上的融合AR GDN，prefill的CH融合策略不变；另独立比较批量提交，以及HTP主机线程1。原共享库与生产配置未修改。全部对照保存4个位置的完整logits，数值比较不计入性能。', '',
        '| 条件 | ms/token，中位数 | 全部正式重复 |', '|---|---:|---|']
    labels={'baseline':'原配置','cpu-fused-gdn':'显式CPU融合GDN＋HTP其他算子','host-one-thread':'HTP主机1线程','batched-submit':'OPBATCH1024 / OPQUEUE16','baseline-after':'原配置末尾回测'}
    for label,title in labels.items():lines.append(f"| {title} | {follow[label]['ms_per_token']:.2f} | "+', '.join(f'{v:.2f}' for v in follow[label]['repetitions_ms_per_token'])+' |')
    lines += ['',f"保留融合GDN减少整体约{result['interventions']['cpu-fused-gdn']['reduction']:.1%}；计算split总数从371降至275，跨后端拷贝从{host['baseline']['copy_MiB_per_token']:.2f}降至{host['cpu-fused-gdn']['copy_MiB_per_token']:.2f} MiB/token。批量提交单独减少约{result['interventions']['batched-submit']['reduction']:.1%}，HTP调用时间从{host['baseline']['parts_ms']['htp_call']:.2f}降至{host['batched-submit']['parts_ms']['htp_call']:.2f} ms/token。两种收益没有相加，也未测组合。", '',
        '主机1线程的最后一次突然变快，同时CPU/HTP耗时均下降，前两次未复现；不挑选该点宣称稳定收益，保留全部原始数据。', '',
        '所有干预的4个位置argmax均与原路径一致；逐位置cosine/NRMSE/max_abs见summary.json。短固定轨迹与正常OK响应只用于诊断，不等于自由生成、长程状态或任务质量验收，不能直接发布为生产优化。', '',
        '## 对稀疏草稿的意义', '',
        '只减少FFN不会消除递归状态在CPU/HTP间的拆分与搬运，也不会消除CPU输出投影或大量小批次。先整理完整模型的融合与提交基线，再评估FFN稀疏，才能避免把运行时损失误当作NPU稀疏能力的上限。当前两个独立修正后仍慢于本轮CPU，尚无NPU草稿净加速结论。', '',
        '原始证据在本目录各条件与../20261113-decode-attribution-followup/；summary.json含全部时延、split明细、DSP计数和数值对照。入口：scripts/build/decode_attribution.py、scripts/run/decode_attribution.py、scripts/analyze/decode_attribution.py。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(initial=initial,host=main_host['parts_ms'],interventions=result['interventions'],dsp=dsp),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
