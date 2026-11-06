#!/usr/bin/env python3
"""原第一张曲线追加OpenCL/CPU；独立会话用明确图例标注，保留全部原始点。"""
import argparse
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import observations as obs

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',type=Path,nargs='+',required=True)
    ap.add_argument('--opencl',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    style_path=Path.home()/'.codex/skills/figure-style/kernel.py'
    spec=importlib.util.spec_from_file_location('figure_style',style_path)
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(10,9,8))
    groups={}; sources=[]; rows=[]; shared=None; response_signature=None
    for candidate,paths in [(False,a.original),(True,a.opencl)]:
        for path in paths:
            m,r,q,_=obs.load_run(path,candidate_scan=candidate)
            c=m['config'];h=c['history_tokens']
            signatures={(x['output'],tuple(x['generated_ids']),x['D']) for x in r if x['route']=='CC' or q[x['P']]}
            if len(signatures)!=1:raise ValueError('输出不一致')
            signature_response=next(iter(signatures))
            if response_signature is not None and response_signature!=signature_response:raise ValueError('跨后端/跨H完整响应不一致')
            response_signature=signature_response
            signature={k:c.get(k) for k in ['context_size','batch_size','kv_policy','pre_move','cpu_repack','numeric_acceptance','input_fixture','release_calibration_state']}
            signature.update(model=m['model'],device_identity=m['device_identity'],runtime=m.get('loaded_runtime_files_sha256'))
            if shared is not None and shared!=signature:raise ValueError('模型/设备/运行库/预算不一致')
            shared=signature
            if candidate and not all(q.values()):raise ValueError('OpenCL未全部准入，不筛选成功子集')
            if (candidate,h) in groups:raise ValueError('重复H，不按性能选择运行')
            groups[candidate,h]=(m,r,q)
            sources.append(dict(run=str(path.resolve()),opencl=candidate,binary_sha256=m['binary_sha256'],
                hashes={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('manifest.json','checked-rows.jsonl','http-server.jsonl','native-events.jsonl')}))
    hs=sorted(h for candidate,h in groups if not candidate)
    if set(h for candidate,h in groups if candidate)!=set(hs):raise ValueError('H覆盖不一致')
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.3),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
    colors={2048:'#0072B2',4096:'#D55E00'}
    for ax,detail in zip(axs,[False,True]):
        for h in hs:
            for candidate,strategy,route,label,ls,marker in [
                (False,'after_return','CC','CPU/CPU','-','D'),
                (False,'always','HC','HTP/CPU','--','s'),
                (True,'always','HC','OpenCL/CPU*','-.','o')]:
                m,r,q=groups[candidate,h]
                ps=[31,32,33] if detail else m['config']['new_tokens']
                ys=[]
                for p in ps:
                    sample=obs.sample(r,h,p,200,strategy,route)
                    if len(sample)!=3:raise ValueError('要求每点3次正式测量')
                    accepted=route=='CC' or q[p]
                    y=statistics.median(x['response_after_return_ms'] for x in sample)/1000 if accepted else float('nan')
                    ys.append(y)
                    if accepted:ax.scatter([p]*3,[x['response_after_return_ms']/1000 for x in sample],s=9,alpha=.25,color=colors[h])
                    if not detail:rows.append(dict(H=h,P=p,backend=label,accepted=accepted,median_response_s=y,run=str(next(path for path in (a.opencl if candidate else a.original) if json.loads((path/'manifest.json').read_text())['config']['history_tokens']==h))))
                ax.plot(ps,ys,ls,marker=marker,markersize=4,color=colors[h],label=f'H={h}, {label}')
        ax.set_xlabel('New input P (tokens)');ax.margins(.06)
    ps=groups[False,hs[0]][0]['config']['new_tokens']
    ticks=[16,24,32,40,48,56,64,128,256,512,1024]
    axs[0].set_xscale('log',base=2);axs[0].set_xticks(ticks,[str(p) for p in ticks])
    axs[1].set_xticks([31,32,33]);axs[0].set_ylabel('Full response after Tool return (s)')
    axs[0].set_title('CPU / HTP / OpenCL across P=16–1024');axs[1].set_title('Detail around P=32')
    handles,labels=axs[0].get_legend_handles_labels()
    order = [0,3,1,4,2,5] if len(handles)==6 else list(range(len(handles)))
    fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='outside upper center',ncol=3)
    fig.supxlabel('n=3 replays/point; Q4_0, F16 KV, batch128; all routes decode on CPU.\n* OpenCL measured in a separate session; CPU/HTP retain the original measurements.',fontsize=9)
    qa={};obs.save(fig,a.output,'01_backend_boundary',qa)
    (a.output/'geometry-qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    if any(v['text_collisions'] or v['outside_canvas'] for v in qa.values()):raise ValueError('图形几何检查失败')
    (a.output/'plot-data.json').write_text(json.dumps(dict(sources=sources,shared=shared,points=rows,clock_diagnostic_read_skew_tolerance_ms={'original':0.1,'opencl':1.0},response_signature=response_signature,plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figure_style_sha256=hashlib.sha256(style_path.read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__),indent=2)+'\n')
    with (a.output/'boundary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(str(a.output/'01_backend_boundary.png'))

if __name__=='__main__':main()
