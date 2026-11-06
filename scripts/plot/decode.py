#!/usr/bin/env python3
"""三后端decode独立图：完整JSON的单token评估与固定序列回放。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from matplotlib.lines import Line2D
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import observations as obs

ROOT=Path(__file__).resolve().parents[2]


def load_run(path):
    m=json.loads((path/'manifest.json').read_text());c=m['config']
    if m['clock_id']!='CLOCK_BOOTTIME' or m['transport']!='ssh':raise ValueError('decode requires SSH/BOOTTIME')
    completion=json.loads((path/'completion.json').read_text())
    if not completion['passed'] and not (completion.get('completed') and completion.get('response_and_state_passed')):raise ValueError('decode run incomplete')
    raw=[json.loads(x) for x in (path/'device/trials.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (path/'checked-rows.jsonl').read_text().splitlines()]
    if len(raw)!=len(rows) or len(rows)!=m['expected_trials']:raise ValueError('incomplete decode matrix')
    if any(any(r[k]!=v for k,v in original.items()) for original,r in zip(raw,rows)):raise ValueError('raw records changed')
    expected={(phase,b,int(h),int(p),rep) for h in c['history_fixtures'] for p in c['input_fixtures']
              for phase in ['reference','warmup','measurement']
              for b in (['cpu'] if phase=='reference' else c['backends'])
              for rep in (range(c['repetitions']) if phase=='measurement' else [-1])}
    keys=[(r['phase'],r['backend'],r['H'],r['P'],r['repetition']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('missing or duplicate conditions')
    if [r['id'] for r in rows]!=list(range(len(rows))):raise ValueError('trial IDs mismatch')
    signatures={(r['output'],tuple(r['generated_ids'])) for r in rows}
    if len(signatures)!=1 or any(len(r['generated_ids'])!=6 for r in rows):raise ValueError('natural output/token count mismatch')
    for r in rows:
        if not all(r[k] for k in ['state_transfer_passed','eog','generated_ids_equal']) or not r.get('natural_passed',r['passed']) or not r['response_check']['passed'] or r.get('natural_min_cosine',r['min_cosine'])<.99:raise ValueError('response/state/numeric failure')
        if 'natural_step_cosine' in r:
            if len(r['natural_step_cosine'])!=6 or len(r['replay_step_cosine'])!=c['horizon']:raise ValueError('missing numerical steps')
            if any(not math.isfinite(v) for k in ['natural_step_cosine','replay_step_cosine'] for v in r[k]):raise ValueError('invalid numerical audit')
            if abs(min(r['natural_step_cosine'])-r['natural_min_cosine'])>1e-6 or abs(min(r['replay_step_cosine'])-r['replay_min_cosine'])>1e-6:raise ValueError('numeric minima inconsistent')
            if r['passed'] != (r['natural_passed'] and r['replay_passed']):raise ValueError('aggregate admission mismatch')
        if 'replay_passed' in r and r['replay_passed'] != (r['replay_min_cosine']>=.99):raise ValueError('replay admission flag mismatch')
        if len(r['natural_step_ms'])!=6 or len(r['replay_step_ms'])!=c['horizon']:raise ValueError('decode length mismatch')
        if c['horizon']>=64 and 'replay_input_ids' not in r:raise ValueError('missing actual replay tokens')
        if 'replay_input_ids' in r and r['replay_input_ids'] != [r['generated_ids'][i%5] for i in range(c['horizon'])]:raise ValueError('replay token stream mismatch')
        for field in ['natural_step_ms','replay_step_ms','natural_active_step_ms','replay_active_step_ms']:
            if any(not math.isfinite(x) or x<0 for x in r[field]):raise ValueError('invalid decode timing')
        if any(sum(r[f'{kind}_step_ms'])>r[f'{kind}_wall_with_audit_ms']+1 for kind in ['natural','replay']):raise ValueError('decode timing identity invalid')
    if hashlib.sha256((path/'device/agent').read_bytes()).hexdigest()!=m['binary_sha256']:raise ValueError('binary hash mismatch')
    for name,digest in m['fixture_sha256'].items():
        if hashlib.sha256((path/'device/fixtures'/name).read_bytes()).hexdigest()!=digest:raise ValueError('fixture hash mismatch')
    return m,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    m,rows=load_run(a.run);out=a.output;out.mkdir(parents=True,exist_ok=True)
    c=m['config'];measured=[r for r in rows if r['phase']=='measurement']
    style_path=Path.home()/'.codex/skills/figure-style/kernel.py';spec=importlib.util.spec_from_file_location('decode_style',style_path)
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style);style.apply_figure_style(sizes=(10,9,8))
    shapes=sorted({(r['H']+r['P'],r['H'],r['P']) for r in measured})
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    colors={'cpu':'#666666','htp':'#D55E00','opencl':'#0072B2'};labels={'cpu':'CPU','htp':'HTP','opencl':'OpenCL GPU'}
    summary=[];has_rejected=False
    for backend in c['backends']:
        xs=[];natural=[];replay=[]
        for kv,h,p in shapes:
            group=[r for r in measured if (r['backend'],r['H'],r['P'])==(backend,h,p)]
            if len(group)!=3:raise ValueError('plot requires exactly three trials')
            ns=[sum(r['natural_step_ms']) for r in group];rs=[statistics.mean(r['replay_step_ms']) for r in group]
            qualification=[r for r in rows if (r['backend'],r['H'],r['P'])==(backend,h,p)]
            qualified=all(r.get('replay_passed',r['passed']) for r in qualification)
            has_rejected |= not qualified
            xs.append(kv);natural.append(statistics.median(ns));replay.append(statistics.median(rs) if qualified else float('nan'))
            axs[0].scatter([kv]*len(ns),ns,s=12,alpha=.3,color=colors[backend])
            if qualified:axs[1].scatter([kv]*len(rs),rs,s=12,alpha=.3,color=colors[backend])
            else:axs[1].scatter([kv]*len(rs),rs,s=28,marker='^',facecolors='none',edgecolors=colors[backend])
            summary.append(dict(H=h,P=p,kv_tokens=kv,backend=backend,natural_decode_median_ms=statistics.median(ns),replay_qualified=qualified,replay_min_cosine=min(r.get('replay_min_cosine',r['min_cosine']) for r in qualification),replay_ms_per_token=statistics.median(rs) if qualified else None,replay_tokens_per_s=1000/statistics.median(rs) if qualified else None))
        for ax,ys in zip(axs,[natural,replay]):ax.plot(xs,ys,'-D',markersize=4,color=colors[backend],label=labels[backend])
    for ax in axs:
        ax.set_xticks([x[0] for x in shapes],[f'{kv}\nH{h}+P{p}' for kv,h,p in shapes]);ax.set_xlabel('Resident KV before decode (tokens)');ax.margins(.1)
    axs[0].set_title('Natural JSON decode');axs[0].set_ylabel('6 single-token evaluations (ms)')
    axs[1].set_title(f"Fixed {c['horizon']}-step replay");axs[1].set_ylabel('Mean evaluation latency (ms/token)')
    handles,legend=axs[0].get_legend_handles_labels()
    if has_rejected:
        handles.append(Line2D([],[],marker='^',linestyle='none',markerfacecolor='none',color='#666666'));legend.append('Replay numeric gate failed')
    fig.legend(handles,legend,loc='outside upper center',ncol=4 if has_rejected else 3)
    fig.supxlabel(f"n=3 trials/backend/condition; Q4_0, F16 KV, batch128, FA off; shared CPU-prefill state.\nLeft: JSON including EOG evaluation. Right: fixed token replay, not a natural response.\nTransfers, initialization and numerical audit excluded; all {c['horizon']} replay steps included.\nOpen triangles: individual rejected replay trials; no admissible median shown.",fontsize=9)
    qa={};obs.save(fig,out,'decode_backends',qa)
    (out/'geometry-qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    if any(x['text_collisions'] or x['outside_canvas'] for x in qa.values()):raise ValueError('decode figure geometry failed')
    (out/'summary.json').write_text(json.dumps(dict(run=str(a.run.resolve()),points=summary,checks_passed=True),indent=2)+'\n')
    with (out/'decode-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    admission={(x['backend'],x['H'],x['P']):x['replay_qualified'] for x in summary}
    traces=[dict(id=r['id'],backend=r['backend'],H=r['H'],P=r['P'],step=i+1,ms=ms,
        trial_replay_passed=r.get('replay_passed',r['passed']),condition_replay_qualified=admission[r['backend'],r['H'],r['P']],
        cosine=r.get('replay_step_cosine',[None]*c['horizon'])[i]) for r in measured for i,ms in enumerate(r['replay_step_ms'])]
    with (out/'decode-traces.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(traces[0]));w.writeheader();w.writerows(traces)
    (out/'plot-data.json').write_text(json.dumps(dict(manifest=m,rows=rows,plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figure_style_sha256=hashlib.sha256(style_path.read_bytes()).hexdigest()),indent=2)+'\n')
    print(out/'decode_backends.png')

if __name__=='__main__':main()
