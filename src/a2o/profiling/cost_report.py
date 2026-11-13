"""验收固定配置的原始配对请求，导出可追溯统计和P/D图。"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import math

from a2o.profiling.model_cost_campaign import campaign_jobs, paired_order, save
from a2o.profiling.pd_sweep import validate


def write_csv(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input.resolve()
    out=root/'report';out.mkdir(exist_ok=True)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    cfg=json.loads((root/'plan.json').read_text())['config']
    completion=json.loads((root/'completion.json').read_text())
    assert all(completion[k] for k in ['completed','service_stopped','lock_released'])
    schedule=json.loads((root/'schedule.json').read_text())
    expected=[dict(round=r,stage=s,P=p,D=max(d,default=0),decode_points=d,
                   backend_order=paired_order(cfg,r,s,p)) for r,s,p,d in campaign_jobs(cfg)]
    assert schedule==expected and len(schedule)==473
    pairs={};raw_rows=[];hashes={};numeric=[];resource=collections.defaultdict(list)
    for job in schedule:
        name=f"r{job['round']:02d}-{job['stage']}-p{job['P']:03d}.json"
        path=root/'measurements'/name;data=json.loads(path.read_text())
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        request,response=data['request'],data['response'];validate(request,response)
        assert request['backend_order']==job['backend_order']
        assert data['round']==job['round'] and data['stage']==job['stage']
        assert len(request['prefix_ids'])==cfg['H'] and len(request['input_ids'])==job['P']
        assert request['decode_points']==job['decode_points']
        numeric.append(response['logit_comparison'])
        for row in response['rows']:
            assert row['single_resident'] and row['n_threads']==(6 if row['backend']=='cpu' else 4)
            a,b,c=[row[k] for k in ['resources_before','resources_after_prefill','resources_after_decode']]
            resource[row['backend']+'-'+job['stage']].append(dict(
                prefill_faults=b['major_faults']-a['major_faults'],
                decode_faults=c['major_faults']-b['major_faults'],swap_kib=c['VmSwap_kib']))
            points=[(0,row['prefill_ms'])] if job['stage']=='prefill' else [(x['D'],x['decode_ms']) for x in row['decode_checkpoints']]
            for d,ms in points:
                key=(job['round'],job['stage'],row['backend'],job['P'],d)
                assert key not in pairs;pairs[key]=ms
                raw_rows.append(dict(round=key[0],stage=key[1],backend=key[2],P=key[3],D=d,
                                     elapsed_s=ms/1000,source=name))
    actual_files=[p for p in (root/'measurements').glob('r*.json') if '.raw.' not in p.name]
    assert len(actual_files)==473 and len(raw_rows)==1006
    csv_rows=list(csv.DictReader((root/'stage-times.csv').open()))
    assert len(csv_rows)==1006
    for row in csv_rows:
        key=(int(row['round']),row['stage'],row['backend'],int(row['P']),int(row['D']))
        assert math.isclose(float(row['elapsed_ms']),pairs.pop(key),abs_tol=1e-8)
    assert not pairs
    for p in cfg['P']:
        group=[j for j in schedule if j['stage']=='prefill' and j['P']==p]
        assert len(group)==10 and sum(j['backend_order'][0]=='cpu' for j in group)==5
    states=[]
    for h in cfg['state_H']:
        value=json.loads((root/'measurements'/f'state-h{h}.json').read_text())
        group=value['response']['rows'];assert len(group)==6
        assert sum(r['warmup'] for r in group)==1
        assert all(r['state_verified'] and r['H']==h and r['P']==40 for r in group)
        assert all(math.isfinite(r[k]) and r[k]>=0 for r in group
                   for k in ['snapshot_ms','prepare_ms','prepare_import_work_ms',
                             'prepare_verification_ms','htp_to_cpu_ms','cpu_restore_ms'])
        states.extend(group)
    state_csv=list(csv.DictReader((root/'state-times.csv').open()))
    assert len(state_csv)==18
    for row in state_csv:
        original=next(r for r in states if r['H']==int(row['H']) and r['repetition']==int(row['repetition']))
        for key in ['snapshot_ms','prepare_ms','htp_to_cpu_ms','cpu_restore_ms','snapshot_bytes','handoff_bytes']:
            assert math.isclose(float(row[key]),original[key],abs_tol=1e-8)
    for row in csv.DictReader((root/'state-cost-table.csv').open()):
        group=[r for r in states if r['H']==int(row['H']) and not r['warmup']]
        assert len(group)==int(row['n'])==5
        for key in ['snapshot_ms','prepare_ms','htp_to_cpu_ms','cpu_restore_ms']:
            assert math.isclose(float(row[key+'_mean']),statistics.mean(r[key] for r in group),abs_tol=1e-8)
            assert math.isclose(float(row[key+'_median']),statistics.median(r[key] for r in group),abs_tol=1e-8)
    for path in ['cc','hh']:
        assert json.loads((root/f'preflight-{path}.json').read_text())['passed']
    for path in ['hc','ch']:
        assert json.loads((root/f'dual-state-preflight-{path}.json').read_text())['passed']
    audit=dict(passed=True,paired_requests=473,prefill_rows=940,decode_rows=66,state_rows=len(states),
               balanced_order=True,csv_matches_raw=True,
               numerical_fail_pairs=sum(not x['passed'] for x in numeric),
               min_cosine=min(x['cosine'] for x in numeric),
               resources={k:{field:dict(median=statistics.median(v[field] for v in vs),
                            maximum=max(v[field] for v in vs),nonzero=sum(v[field]>0 for v in vs))
                            for field in ['prefill_faults','decode_faults','swap_kib']} for k,vs in resource.items()},
               completion=completion)
    assert audit['numerical_fail_pairs']==0
    save(out/'audit.json',audit);save(out/'raw-identities.json',hashes)
    write_csv(out/'validated-samples.csv',raw_rows)

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    sys.path.insert(0,str(Path.home()/'.codex/skills/figure-style'))
    from kernel import apply_figure_style
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    apply_figure_style(font='Arial Unicode MS',sizes=(14,12,11))
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['savefig.bbox']=None
    colors={'cpu':'#2379AE','htp':'#C47524'}
    labels={'cpu':'CPU（6线程）','htp':'HTP offload（CPU attention，主机4线程）'}
    aggregates=[];exclusions=[];qa=[]

    def export(fig,name):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
        bounds=[(t,t.get_window_extent(renderer)) for t in texts]
        outside=[t.get_text() for t,b in bounds if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]
        overlaps=[(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(bounds)
                  for b,bb in bounds[i+1:] if ba.overlaps(bb)]
        qa.append(dict(figure=name,outside=outside,text_overlaps=overlaps))
        assert not outside and not overlaps,(name,outside,overlaps)
        for ext in ['png','pdf','svg']:fig.savefig(figs/(name+'.'+ext),dpi=300)
        # 单独保存面板，便于逐面板像素检查。
        for i,ax in enumerate(fig.axes):
            fig.savefig(figs/(name+f'-panel-{i+1}.png'),dpi=160,
                        bbox_inches=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.04))
        plt.close(fig)

    for clean in [False,True]:
        fig,axes=plt.subplots(1,2,figsize=(15,7))
        fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.70,wspace=.23)
        fig.text(.075,.94,'Prefill长度扫描'+('：上尾异常值敏感性分析' if clean else '：全部原始样本'),fontsize=14)
        fig.text(.075,.885,'H=4096 · Qwen3-8B Q4_0 · context4608 · batch128 · F16 KV',fontsize=12)
        for backend in ['cpu','htp']:
            series=[]
            for p in cfg['P']:
                rows=[r for r in raw_rows if r['stage']=='prefill' and r['backend']==backend and r['P']==p]
                vals=[r['elapsed_s'] for r in rows];assert len(vals)==10
                q1,q3=np.quantile(vals,[.25,.75]);threshold=float(q3+1.5*(q3-q1))
                included=[r for r in rows if not clean or r['elapsed_s']<=threshold]
                excluded=[r for r in rows if clean and r['elapsed_s']>threshold]
                if clean:exclusions.extend(dict(r,threshold_s=threshold) for r in excluded)
                v=[r['elapsed_s'] for r in included]
                stat=dict(version='cleaned' if clean else 'all',backend=backend,P=p,n=len(v),
                          excluded=len(excluded),mean_s=statistics.mean(v),median_s=statistics.median(v))
                aggregates.append(stat);series.append(stat)
                for i,ax in enumerate(axes):
                    if i==1 and p>64:continue
                    ax.scatter([p]*len(v),v,color=colors[backend],alpha=.28,s=20,linewidths=0,zorder=4)
            for i,ax in enumerate(axes):
                selected=[r for r in series if i==0 or r['P']<=64]
                ax.plot([r['P'] for r in selected],[r['mean_s'] for r in selected],'-s',
                        color=colors[backend],lw=1.8,ms=4,label=labels[backend],zorder=3)
        for i,ax in enumerate(axes):
            ax.set_title(['a  全范围：P24–128','b  逐整数区域：P24–64'][i],loc='left',pad=14)
            ax.set_xlabel('新增token数 P');ax.set_ylabel('Prefill耗时（秒）')
            ax.set_xticks([24,40,64,80,96,128] if i==0 else list(range(24,65,4)))
            ax.grid(axis='y',color='#E5E7EB',lw=.7);ax.margins(x=.04,y=.10)
        handles,legend_labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='upper left',bbox_to_anchor=(.075,.835),ncol=2,frameon=False)
        removed=len(exclusions) if clean else 0
        note=(f'各P、各后端以Q3+1.5×IQR为上界；940个样本中排除{removed}个，各组保留8–10次。' if clean
              else '每个P、每个后端n=10；全部940个样本进入均值，不删除慢点或按换页标记筛选。')
        fig.text(.075,.13,note,fontsize=12)
        fig.text(.075,.075,'方点连线：均值；浅色圆点：样本。单模型驻留；加载、预热、恢复与迁移不计入阶段耗时。',fontsize=12)
        export(fig,'prefill-cleaned' if clean else 'prefill-all')

    fig,ax=plt.subplots(figsize=(9,6.8));fig.subplots_adjust(left=.13,right=.97,bottom=.25,top=.72)
    fig.text(.13,.94,'Decode：连续逐token执行',fontsize=14)
    fig.text(.13,.885,'H=4096 · P40 · Qwen3-8B Q4_0 · 每后端n=3',fontsize=12)
    decode_stats=[]
    for backend in ['cpu','htp']:
        means=[]
        for d in cfg['D']:
            values=[r['elapsed_s'] for r in raw_rows if r['stage']=='decode' and r['backend']==backend and r['D']==d]
            assert len(values)==3
            means.append(statistics.mean(values))
            ax.scatter([d]*3,values,color=colors[backend],s=28,alpha=.35,linewidths=0,zorder=4)
            decode_stats.append(dict(backend=backend,D=d,n=3,mean_s=means[-1],median_s=statistics.median(values)))
        ax.plot(cfg['D'],means,'-s',color=colors[backend],ms=4,lw=1.8,label=labels[backend],zorder=3)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.13,.84),frameon=False,fontsize=12)
    ax.set_xlabel('输出token数 D');ax.set_ylabel('累计Decode耗时（秒）')
    ax.set_xticks([1,64,128,192,256,320]);ax.grid(axis='y',color='#E5E7EB',lw=.7);ax.margins(x=.04,y=.08)
    fig.text(.13,.14,'均值连线＋三次原始观测；每轮连续执行到D320，保留全部样本。',fontsize=12)
    fig.text(.13,.085,'固定token前向计算，不含采样；初始化、状态恢复和检查点验证另计。',fontsize=12)
    export(fig,'decode')
    write_csv(out/'prefill-summary.csv',aggregates)
    write_csv(out/'decode-summary.csv',decode_stats)
    if exclusions:write_csv(out/'excluded-prefill.csv',exclusions)
    else:(out/'excluded-prefill.csv').write_text('round,stage,backend,P,D,elapsed_s,source,threshold_s\n')
    save(out/'figure-qa.json',qa)
    save(out/'plot-summary.json',dict(prefill_original_n=940,cleaned_excluded_n=len(exclusions),
         cleaned_retained_min=min(x['n'] for x in aggregates if x['version']=='cleaned'),decode_original_n=66))
    lines=['# Qwen3-8B固定配置验收与图表','',
           '验收通过：473个配对请求；940条prefill、66条decode检查点、18条状态成本（含3条预热）。',
           '逐点十次、CPU/HTP先测5/5、固定seed轮内乱序均核对；CSV与全部原始响应逐行一致。',
           f"全部配对的prefill logits对照通过；最低cosine={audit['min_cosine']:.8f}。这不等于完整任务质量验收。",'',
           '当前固定配置：HTP offload，CPU attention/KV与Q6_K输出头，FA关闭；不外推为后端默认或纯NPU性能。',
           'P/D为单模型驻留；逐点重载时的初始化、32-token预热和恢复单列。状态交接另为双驻留，helper包含字节核验。','',
           '主图保留全部样本；清理图仅按各P/后端Q3+1.5IQR上界排除慢点，不删除原始记录。',
           f'清理图排除{len(exclusions)}/940点，完整清单见excluded-prefill.csv。Decode三轮全部保留。','',
           '图形导出PNG/PDF/SVG；图中文字边界和重叠检查通过，像素审阅另行执行。','',
           '| P | CPU原始均值 s | HTP offload原始均值 s |','|---|---:|---:|']
    for p in [24,31,32,40,64,128]:
        vals={x['backend']:x['mean_s'] for x in aggregates if x['version']=='all' and x['P']==p}
        lines.append(f"| {p} | {vals['cpu']:.3f} | {vals['htp']:.3f} |")
    lines.extend(['','| D320 | CPU均值 s | HTP offload均值 s |','|---|---:|---:|'])
    vals={x['backend']:x['mean_s'] for x in decode_stats if x['D']==320}
    lines.append(f"| 累计 | {vals['cpu']:.3f} | {vals['htp']:.3f} |")
    (out/'验收报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit=audit['passed'],excluded=len(exclusions),report=str(out)),ensure_ascii=False))
