#!/usr/bin/env python3
"""保留图1旧数据，增加独立复测的最佳CPU线程曲线及同二进制4线程对照。"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.cpu_threads import load_thread_run,select_threads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
spec=importlib.util.spec_from_file_location('observation_plot',ROOT/'scripts/plot/observations.py');obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);a=ap.parse_args();base=a.campaign;out=base/'figures';out.mkdir(exist_ok=True)
    state=json.loads((base/'campaign.json').read_text());selection=json.loads((base/'selection.json').read_text());best=selection['best_threads']
    old=ROOT/'results/20261105-xiaomi14pro-observation-main/figures/plot-data.json';baseline=json.loads(old.read_text());reference=json.loads((Path(baseline['sources'][0]['directory'])/'manifest.json').read_text())
    verified_old=[]
    for source in baseline['sources']:verified_old+=obs.load_run(Path(source['directory']))[1]
    if verified_old!=baseline['rows']:raise ValueError('图1数据与原始验收记录不同')
    sweeps=[];curves=[];provenance=[];constants=None
    for kind in ['sweep','curve']:
        for path in state[kind+'_runs']:
            m,rows=load_thread_run(path);c=m['config']
            for key in ['context_size','batch_size','htp_queue','htp_opbatch','kv_policy','pre_move','cpu_repack','release_calibration_state','input_fixture','numeric_acceptance']:
                if c[key]!=reference['config'][key]:raise ValueError('与图1配置不一致：'+key)
            for key in ['device_identity','device_files_sha256','clock_id','model']:
                if m[key]!=reference[key]:raise ValueError('与图1设备/模型/时钟不一致：'+key)
            # 新二进制仅增加线程参数、setter与记录；其余原生源文件必须与图1一致。
            changes=[k for k,v in reference['source_files_sha256'].items() if m['source_files_sha256'].get(k)!=v]
            if changes!=['src/a2o/native/profiling/oracle.h']:raise ValueError('存在未解释的原生源码变化：'+str(changes))
            if constants is not None and constants!=m['binary_sha256']:raise ValueError('本次线程扫描/复测不是同一二进制')
            constants=m['binary_sha256'];provenance.append(dict(kind=kind,path=path,manifest=m))
            (sweeps if kind=='sweep' else curves).extend(rows)
    recalculated=select_threads(sweeps,list(range(1,9)))
    if json.loads(json.dumps(recalculated))!=selection:raise ValueError('线程选择与原始数据不一致')
    skill=Path.home()/'.codex/skills/figure-style/kernel.py';sp=importlib.util.spec_from_file_location('figure_style',skill);style=importlib.util.module_from_spec(sp);sp.loader.exec_module(style);style.apply_figure_style(sizes=(10,9,8))
    qa={};colors={2048:'#0072B2',4096:'#D55E00'};ps=[16,24,31,32,33,40,48,56,64]
    def group(rows,h,p,t):return [r for r in rows if (r['phase'],r['H'],r['P'],r['cpu_prefill_threads'])==('measurement',h,p,t)]
    def med(rows,key):return st.median(r[key] for r in rows)/1000
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,h in zip(axs,[2048,4096]):
        for p,color in zip([16,31,32,64],['#0072B2','#D55E00','#009E73','#CC79A7']):
            groups=[group(sweeps,h,p,t) for t in range(1,9)]
            ax.plot(range(1,9),[med(g,'prefill_ms') for g in groups],marker='D',color=color,label=f'P={p}')
            for t,g in enumerate(groups,1):ax.scatter([t]*len(g),[r['prefill_ms']/1000 for r in g],s=8,alpha=.3,color=color)
        ax.axvline(best,color='#888888',ls=':',lw=1);ax.set_title(f'H={h}; selected {best} threads');ax.set_xlabel('CPU prefill threads');ax.set_xticks(range(1,9));ax.margins(.05)
    axs[0].set_ylabel('CPU prefill time (s)');axs[0].legend();obs.save(fig,out,'01_thread_sweep',qa)
    old_rows=[r for r in baseline['rows'] if r['phase']=='measurement']
    def original(h,p,route):return [r for r in old_rows if (r['H'],r['P'],r['gap_requested_ms'],r['strategy'],r['requested_route'])==(h,p,200,'after_return' if route=='CC' else 'always',route)]
    fig,axs=plt.subplots(1,2,figsize=(12,4.2),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
    for ax,points in zip(axs,[ps,[31,32,33]]):
        for h in [2048,4096]:
            for route,ls,marker in [('CC','-','D'),('HC','--','s')]:
                samples=[original(h,p,route) for p in points]
                ax.plot(points,[med(g,'response_after_return_ms') for g in samples],ls,marker=marker,markersize=3.5,color=colors[h],label=f'H={h}, original '+('CPU 4t' if route=='CC' else 'HTP/CPU'))
            samples=[group(curves,h,p,best) for p in points]
            ax.plot(points,[med(g,'response_after_return_ms') for g in samples],'-.',marker='^',lw=1.8,color=colors[h],label=f'H={h}, CPU prefill {best}t')
            for p,g in zip(points,samples):ax.scatter([p]*len(g),[r['response_after_return_ms']/1000 for r in g],s=9,color=colors[h],alpha=.4)
        ax.set_xlabel('New input P (tokens)');ax.margins(.06)
    axs[0].set_xticks([16,24,32,40,48,56,64]);axs[1].set_xticks([31,32,33]);axs[0].set_ylabel('Full response after Tool return (s)')
    axs[0].set_title('Best-thread CPU overlay, P≤64');axs[1].set_title('Detail around P=32')
    handles,labels=axs[0].get_legend_handles_labels();order=[0,3,1,4,2,5]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='outside upper center',ncol=3);obs.save(fig,out,'02_backend_boundary_best_threads_p64',qa)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');comparison=[]
    for ax,h in zip(axs,[2048,4096]):
        for t,ls in [(4,'-'),(best,'--')] if best!=4 else [(4,'-')]:
            ax.plot(ps,[med(group(curves,h,p,t),'response_after_return_ms') for p in ps],ls,marker='D',label=f'CPU {t}t',color='#777777' if t==4 else '#0072B2')
        ax.plot(ps,[med(original(h,p,'CC'),'response_after_return_ms') for p in ps],':',color='#BBBBBB',label='Original CPU 4t')
        ax.set_title(f'H={h}, same-binary controls');ax.set_xlabel('New input P (tokens)');ax.set_xticks([16,24,32,40,48,56,64]);ax.margins(.05)
        for p in ps:
            control=group(curves,h,p,4);opt=group(curves,h,p,best)
            comparison.append(dict(H=h,P=p,best_threads=best,control_prefill_ms=med(control,'prefill_ms')*1000,best_prefill_ms=med(opt,'prefill_ms')*1000,control_response_ms=med(control,'response_after_return_ms')*1000,best_response_ms=med(opt,'response_after_return_ms')*1000,control_trial_ids=[r['id'] for r in control],best_trial_ids=[r['id'] for r in opt]))
    axs[0].set_ylabel('Full response after Tool return (s)');axs[0].legend();obs.save(fig,out,'03_same_binary_controls',qa)
    (out/'geometry-qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    (out/'plot-data.json').write_text(json.dumps(dict(selection=selection,provenance=provenance,sweep_rows=sweeps,curve_rows=curves,comparisons=comparison,original_source=str(old),
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_code_sha256=hashlib.sha256((ROOT/'src/a2o/profiling/cpu_threads.py').read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    with (out/'best-thread-curve.csv').open('w') as f:
        names=list(comparison[0]);writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(comparison)
    text=f'''# CPU最优线程补充图（P≤64）

选择结果：本次扫描范围内全局最优为 **{best} 个prefill线程**。decode始终4线程。选择只用独立扫描数据，对H2048/4096和P16/31/32/64八个条件等权，比较三次prefill中位数相对4线程的几何平均；不是逐点挑最快的Oracle包络，也不是普适最优线程的证明。

- [线程扫描](01_thread_sweep.png) · [PDF](01_thread_sweep.pdf)
- [图1补充：最佳线程CPU曲线，P≤64](02_backend_boundary_best_threads_p64.png) · [PDF](02_backend_boundary_best_threads_p64.pdf)
- [同一新二进制的4线程复测对照](03_same_binary_controls.png) · [PDF](03_same_binary_controls.pdf)

模型、输入正文、历史前缀、设备、F16 KV、ctx6144、batch128、CPU/HTP双模型内存配置、SIMD和repack关闭均与图1一致。唯一有意改变的是CPU prefill线程数。CPU历史建立与decode均固定4线程；G目标200ms，完整延迟仍用设备CLOCK_BOOTTIME实际返回后到完整JSON发布。每个线程/形状先有独立logits与完整输出校准，然后随机顺序测量3次；校准不进入延迟汇总。最大P=64，没有补跑更大P。

图1原CPU4线程和HTP曲线来自已验收的旧批次，原文件不改。新CPU最佳曲线是选择结束后的独立复测；同时复测4线程，用于检查新旧轮次漂移。线程变化使新二进制与图1二进制不同，因此线程收益应以同二进制新4线程对照判定；图中旧曲线保留为背景参考。原生其他文件、运行时库和模型hash都核对一致。未修改手机设置、频率、亲和性或执行模型动作。

只有3次重复，接近的线程差异不足以证明统计显著。完整原始记录和所选trial ID见plot-data.json；汇总数据见best-thread-curve.csv。几何检查见geometry-qa.json，视觉审阅另记。
'''
    (out/'README.md').write_text(text)
    print(json.dumps(dict(best_threads=best,geometry=qa,output=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
