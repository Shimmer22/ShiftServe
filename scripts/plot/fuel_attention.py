"""绘制 fuel 单例真实注意力；保留全上下文归一化。"""
import argparse
import json
from pathlib import Path
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

p = argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--style',type=Path,required=True)
a = p.parse_args()
runpy.run_path(str(a.style))['apply_figure_style'](sizes=(11,10,9))
m = json.loads((a.root/'metadata.json').read_text())
d = np.load(a.root/'attention.npz')
w,positions = d['weights'],d['positions']
avg = w.mean(0)
h,n = m['H'],len(m['input_ids'])
span = m['tool_spans'][-1]
s,e = span['start'],span['end']
tokens = m['tokens']

def label(i):
    return f'{i}: '+tokens[i].replace('\n','↵').replace(' ','·')

# 最后一个 P query 同时预测第一个输出，因此在 decode 图中再次展示。
decode_pos = [n-1+i for i in m['decode_sample_indices']]
di = [list(positions).index(x) for x in decode_pos]
pi = [j for j,x in enumerate(positions) if h<=x<n]
local = avg[pi,h:n]
fig,axes = plt.subplots(1,2,figsize=(16,9),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
im = axes[0].imshow(local,norm=LogNorm(vmin=1e-5,vmax=max(.01,local.max())),cmap='magma',aspect='auto')
axes[0].set(xticks=range(n-h),xticklabels=[label(i) for i in range(h,n)],
    yticks=range(n-h),yticklabels=[label(i) for i in range(h,n)],
    xlabel='Key token (absolute position)',ylabel='Query token (absolute position)',
    title='A  Tool-return prefill: attention within the 27 appended tokens')
plt.setp(axes[0].get_xticklabels(),rotation=90)
for ax in axes[:1]:
    for b in [s-h-.5,e-h-.5]:
        ax.axvline(b,color='#00bcd4',lw=1)
        ax.axhline(b,color='#00bcd4',lw=1)
fig.colorbar(im,ax=axes[0],shrink=.65,label='Mean attention probability (log scale)')
v=avg[di,s:e]
im2=axes[1].imshow(v,cmap='magma',aspect='auto',vmin=0,vmax=v.max())
axes[1].set(xticks=range(e-s),xticklabels=[label(i) for i in range(s,e)],
    yticks=range(len(di)),yticklabels=[f'Predict {i}: '+tokens[n+i].replace('\n','↵') if n+i<len(tokens) else f'Predict {i}: [last output]' for i in m['decode_sample_indices']],
    xlabel='Key token in fuel result',ylabel='Sampled output prediction',
    title='B  Decode: attention to the fuel result')
plt.setp(axes[1].get_xticklabels(),rotation=90)
fig.colorbar(im2,ax=axes[1],shrink=.65,label='Mean attention probability')
fig.suptitle('BFCL base 55 · fuelLevel = 22.5 · head/layer mean over 8 full-attention layers')
fig.savefig(a.root/'fuel-local.png');fig.savefig(a.root/'fuel-local.pdf')
plt.close(fig)

# 上下文分块求和，不将历史裁掉后重新 softmax。
bins=list(range(0,h,64))+[h,s,e,n,len(tokens)]
bins=sorted(set(bins))
binned=np.stack([avg[:,x:y].sum(-1) for x,y in zip(bins[:-1],bins[1:])],-1)
fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained')
im=axes[0].imshow(binned,norm=LogNorm(vmin=1e-5,vmax=1),aspect='auto',cmap='magma')
ticks=list(range(0,len(bins)-1,8))+[len(bins)-2]
axes[0].set(xticks=ticks,xticklabels=[str(bins[i]) for i in ticks],xlabel='Key position (history bins of 64 tokens)',
    ylabel='Query row',title='A  Attention mass across the full visible context')
axes[0].axhline(len(pi)-1.5,color='#00bcd4',lw=1)
fig.colorbar(im,ax=axes[0],label='Attention mass per bin (log scale)')
mass=w[:,di,s:e].sum(-1)
im=axes[1].imshow(mass,aspect='auto',cmap='magma',vmin=0,vmax=mass.max())
axes[1].set(yticks=range(len(d['layers'])),yticklabels=d['layers'],
    xticks=range(len(di)),xticklabels=m['decode_sample_indices'],
    xlabel='Output token being predicted (zero based)',ylabel='Full-attention layer (zero based)',
    title='B  Decode attention mass on the 10-token fuel result, by layer')
fig.colorbar(im,ax=axes[1],label='Attention mass on fuel result')
fig.savefig(a.root/'fuel-context.png');fig.savefig(a.root/'fuel-context.pdf')
summary=dict(decode_fuel_mass_mean=float(mass.mean()),decode_fuel_mass_max_layer_query=float(mass.max()),
    prefill_fuel_mass_mean=float(w[:,pi,s:e].sum(-1).mean()),
    greedy_matches=m['greedy_matches'],greedy_checked=m['greedy_checked'])
(a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
