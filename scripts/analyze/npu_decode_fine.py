#!/usr/bin/env python3
"""按同一正式重复汇总CPU节点、DSP算子和主机调用；张量字节不冒充DRAM流量。"""
from collections import Counter, defaultdict
import json
from math import prod
from pathlib import Path
import re
from decode_attribution import read_case

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/20261113-npu-decode-fine'


def category(op, names, output=''):
    if 'output.weight' in names.split() or output == 'result_output':
        return 'output_projection'
    if '.ffn_' in names or re.search(r'\bffn_', names):
        return 'ffn'
    if op == 'GATED_DELTA_NET':
        return 'deltanet_recurrent_core'
    if re.match(r'kq(?:-|v-|_soft_max-)', output) or op == 'SOFT_MAX':
        return 'attention_kv_core'
    if op == 'SET_ROWS':
        return 'attention_kv_write'
    if op == 'MUL_MAT' and '.weight' in names:
        layer = re.search(r'blk\.(\d+)\.', names)
        if layer:
            return 'attention_projections' if int(layer[1]) % 4 == 3 else 'deltanet_projections'
    if op == 'SSM_CONV' or re.search(r'dnet_|__fgdn_ar__|state_predelta|state_update|new_state|cache_[rs]_l|[qkv]_conv|conv_|ssm_|alpha-|beta-|a_softplus|linear_attn', names):
        return 'deltanet_other'
    if re.search(r'Qcur|Kcur|Vcur|kqv|cache_[kv]|gate_sigmoid|attn_pregate|attn_gated', names):
        return 'attention_other'
    return 'norm_residual_other'


def dsp_rows(path):
    rows = defaultdict(lambda: dict(us=Counter(), calls=Counter(), batches=0, batch_response_us=0, batch_ops_us=0, ops=[]))
    rep = None
    for line in path.read_text().splitlines():
        m = re.search(r'A2O_DECODE_BEGIN rep=(-?\d+)', line)
        if m:
            rep = int(m[1]); continue
        if 'A2O_DECODE_END' in line:
            rep = None; continue
        if rep is None or rep < 0:
            continue
        m = re.search(r'profile-op (.*?): (.*?) : (.*?) : (.*?) : (.*?) : usec (\d+) cycles', line)
        if m:
            op, names, dims, types, strides, us = m.groups(); output = names.rsplit(' -> ', 1)[-1]
            group = category(op, names, output)
            r = rows[rep];r['us'][group] += int(us);r['calls'][group] += 1
            r['ops'].append(dict(op=op, names=names, dims=dims, types=types, strides=strides, us=int(us), category=group))
        m = re.search(r'profile-batch n-ops (\d+) batch-dur-usec (\d+) htp-ops-usec (\d+)', line)
        if m:
            r=rows[rep];r['batches']+=1;r['batch_response_us']+=int(m[2]);r['batch_ops_us']+=int(m[3])
    return rows


def analyze(folder):
    d, stats, representative = read_case(folder)
    assert d['warmup_elapsed_ms'] >= 40000
    if not d['config'].get('host_instrumentation'):
        return dict(stats=stats)
    dsp = dsp_rows(folder/'runtime.log')
    repeats=[]
    for r in d['rows']:
        if r['warmup']:
            continue
        count=r['D'];timing=r['backend_timing'];cpu_us=Counter();cpu_counts=Counter();cpu_details=[];kv_bytes=0;kv_dims=[]
        splits=Counter()
        for key, v in timing.items():
            if key.startswith('CPU_NODE/'):
                _,op,name=key.split('/',2);meta=v['metadata'];names=' '.join([name]+[s['name'] for s in meta['sources']])
                group=category(op,names,name);cpu_us[group]+=v['us'];cpu_counts[group]+=v['calls']
                cpu_details.append(dict(key=key,category=group,ms_per_token=v['us']/count/1000,metadata=meta))
                if group=='attention_kv_core' and op=='MUL_MAT':
                    source=meta['sources'][0];assert source['type']=='f16'
                    # 非连续view的ggml_nbytes可能是地址跨度；按形状计算逻辑元素字节。
                    logical=prod(source['ne'])*2;kv_bytes+=logical*v['calls']/count
                    kv_dims.append(dict(node=name,source=source,logical_element_bytes=logical))
            else:
                group='cpu' if key.startswith('COMPUTE/CPU/') else 'htp' if key.startswith('COMPUTE/HTP') else 'copy' if key=='COPY' else 'sync'
                splits[group]+=v['us']/count/1000
        prof=dsp[r['repetition']];assert prof['ops'] and cpu_counts['deltanet_recurrent_core']==24*count
        assert cpu_counts['attention_kv_core']==24*count and cpu_counts['attention_kv_write']==16*count
        assert abs(sum(prof['us'].values())-prof['batch_ops_us'])<10
        cpu_total=sum(cpu_us.values())/count/1000;dsp_total=sum(prof['us'].values())/count/1000
        assert cpu_total<=splits['cpu']+0.2 and dsp_total<=splits['htp']+0.2
        decomposition=dict(cpu_nodes=cpu_total,cpu_dispatch_and_instrumentation=splits['cpu']-cpu_total,
            dsp_ops=dsp_total,htp_host_queue_cache_wait_profile=splits['htp']-dsp_total,
            explicit_copy=splits['copy'],explicit_sync=splits['sync'],
            residual_host=r['elapsed_ms']/count-sum(splits.values()))
        assert abs(sum(decomposition.values())-r['elapsed_ms']/count)<0.01
        repeats.append(dict(repetition=r['repetition'],wall_ms_per_token=r['elapsed_ms']/count,
            decomposition_ms=decomposition,cpu_ms={k:v/count/1000 for k,v in cpu_us.items()},
            dsp_ms={k:v/count/1000 for k,v in prof['us'].items()},
            cpu_calls_per_token={k:v/count for k,v in cpu_counts.items()},
            dsp_calls_per_token={k:v/count for k,v in prof['calls'].items()},
            dsp_batches_per_token=prof['batches']/count,
            batch_response_ms_sum=prof['batch_response_us']/count/1000,
            kv_logical_input_MiB=kv_bytes/2**20,kv_shapes=kv_dims,
            cpu_node_details=sorted(cpu_details,key=lambda x:-x['ms_per_token']),
            dsp_ops=prof['ops'],
            major_faults=r['resources_after']['major_faults']-r['resources_before']['major_faults']))
    selected=next(r for r in repeats if r['repetition']==representative['repetition'])
    inventory=Counter()
    for w in d['weight_inventory']:
        name=w['name']
        group='embedding_lookup_table' if name=='token_embd.weight' else category('MUL_MAT',name)
        inventory[group]+=w['logical_bytes']
    return dict(stats=stats,H=d['actual_H'],P=d['actual_P'],repeats=repeats,representative=selected,
        weight_storage_MiB={k:v/2**20 for k,v in inventory.items()},
        scope='CPU节点时间包含既有barrier等待；融合组不拆分；DSP内部usec来自原版profile；差额包含提交/缓存维护/等待/计时打印，非纯RPC；KV为逻辑输入量，非DRAM实测')


def main():
    plan=json.loads((ROOT/'configs/npu-decode-fine.json').read_text())
    cases={c['name']:analyze(OUT/c['name']) for c in plan['cases']}
    result=dict(cases=cases,scope='单请求固定token，F16 KV，Qwen3.5-9B Q4_0，CPU/HTP混合；未测硬件带宽/缓存miss计数器')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 优化后decode细粒度归因', '',result['scope'], '',
        '以下分解各取本组正式墙钟中位数对应的同一重复；CPU节点与DSP算子不重复相加到外层执行段。所有预热、正式重复与失败日志均保留。', '',
        '| 条件 | 正式ms/token | 三次正式重复 | KV核心ms/token | KV逻辑输入MiB/token |', '|---|---:|---|---:|---:|']
    for name,c in cases.items():
        r=c.get('representative');stats=c['stats']
        lines.append(f"| {name} | {stats['ms_per_token']:.2f} | "+' / '.join(f'{v:.2f}' for v in stats['repetitions_ms_per_token'])+(f" | {r['cpu_ms']['attention_kv_core']:.2f} | {r['kv_logical_input_MiB']:.1f} |" if r else ' | — | — |'))
    for name in ['fine-h4096-after','fine-keep25-h4096']:
        r=cases[name]['representative'];lines+=['',f'## {name}', '', '| 部分 | ms/token |', '|---|---:|']
        for backend in ['cpu','dsp']:
            for label,v in r[backend+'_ms'].items():lines.append(f'| {backend}/{label} | {v:.3f} |')
        for label,v in r['decomposition_ms'].items():
            if label not in ['cpu_nodes','dsp_ops']:lines.append(f'| {label} | {v:.3f} |')
    lines+=['', '## 测量边界', '',
        'CPU逐节点时间包含既有barrier等待，CPU融合节点作为整体。没有增加同步点。CPU执行段减节点时间包含线程池派发、线程准备、收尾与插桩；不能把差额全称为OS调度。', '',
        'DSP内部usec来自原版profile=1。外层HTP调用减DSP内部时间包含主机提交、缓存维护、等待、profiling和打印，不能全称为RPC，也不能把并发批次响应时长之和当墙钟。', '',
        'H512通过重新prefill实际前缀构建递归状态，未截断H4096的递归状态。其余输入P40保持相同。短前缀仅用于计算形状对照，不作任务质量结论。H4096的KV扫描被补齐到4352，H512到768；输入逻辑量为136/24 MiB。按节点形状计算，非实测DRAM字节。原始metadata的logical_bytes字段来自ggml_nbytes，非连续view可能表示地址跨度，分析KV时已按ne元素数重算。', '',
        '首个H4096组318→217→181 ms/token，控制组前后296→230，存在明显漂移，无法干净估计插桩百分比，也不能由两组墙钟差精确推导H效应。KV核心在H512为2.55–2.67 ms，末次H4096为11.03–11.74 ms；方向及量级一致。', '',
        '未测DDR带宽、cache miss或权重访存stall计数。矩阵乘耗时包含读取、解量化和计算，不能直接认定为纯访存瓶颈。正常响应和完整历史恢复通过，正式窗口major fault为0；未做本轮自由生成质量或投机接受率验收。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({n:dict(ms=c['stats']['ms_per_token'],**({k:c['representative'][k] for k in ['decomposition_ms','cpu_ms','dsp_ms','kv_logical_input_MiB','dsp_batches_per_token']} if 'representative' in c else {})) for n,c in cases.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
