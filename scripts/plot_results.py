#!/usr/bin/env python3
"""Render the recorded phone diagnostics; no device access or new measurements."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.text as mtext
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures'
STYLE = Path.home() / '.codex/skills/figure-style/kernel.py'
spec = importlib.util.spec_from_file_location('figure_style', STYLE)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
fm.fontManager.addfont(FONT)
style.apply_figure_style(font=fm.FontProperties(fname=FONT).get_name(), sizes=(14, 12, 11))
plt.rcParams.update({'axes.unicode_minus': False, 'text.color': '#233444',
                     'axes.labelcolor': '#233444', 'xtick.color': '#526271',
                     'ytick.color': '#233444', 'figure.facecolor': 'white'})

BLUE, GREY, GOLD, PURPLE = '#287EA8', '#A9B3BE', '#DEA54A', '#9180A7'
TEAL = '#168A8A'
qa, sources = {}, {}


def load(name):
    path = ROOT / 'results' / name / 'result.json'
    result = json.loads(path.read_text())
    sources[name] = str(path.relative_to(ROOT))
    return result


def title(fig, headline, subtitle):
    fig.text(.05, .95, headline, va='top', fontsize=14)
    fig.text(.05, .875, subtitle, va='top', fontsize=12, color='#526271')


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip()]
    collisions = []
    for i, (a, box_a) in enumerate(texts):
        for b, box_b in texts[i+1:]:
            if box_a.overlaps(box_b):
                collisions.append([a.get_text(), b.get_text()])
    outside = [t.get_text() for t, box in texts
               if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1)]
    qa[name] = {'text_collisions': collisions, 'text_outside_figure': outside}
    fig.savefig(OUT / (name+'.png'), dpi=300, bbox_inches=None)
    fig.savefig(OUT / (name+'.pdf'), bbox_inches=None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [
        '20261103-145229-319495-hybrid-none-screenshot-history',
        '20261103-145315-264640-hybrid-none-pre-screenshot-history',
        '20261103-145358-148081-hybrid-recent-screenshot-history',
        '20261103-145440-670886-hybrid-recent-pre-screenshot-history',
    ]
    rows = [load(name) for name in runs]
    assert all(r['response_check']['passed'] and r['transfer_checked'] for r in rows)
    assert len({tuple(r['generated_ids']) for r in rows}) == 1
    labels = ['不整理 · 不预搬移', '不整理 · 预搬移', 'recent 整理 · 不预搬移', 'recent 整理 · 预搬移']

    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    fig.subplots_adjust(left=.27, right=.94, bottom=.29, top=.73)
    title(fig, '观察到达后的耗时，仍主要在视觉编码与 prefill',
          '同一截图与历史 · HTP prefill → CPU decode · 四条件响应和 KV 检查均通过')
    t = [r['timing_ms'] for r in rows]
    total = np.array([x['observation_to_response_ms'] for x in t])/1000
    init = np.array([x['vision_init_ms'] for x in t])/1000
    obs = np.array([x['observation_eval_ms'] for x in t])/1000
    decode = np.array([x['decode_ms'] for x in t])/1000
    other = total-init-obs-decode
    assert np.all(other >= 0)
    left = np.zeros(4)
    for values, color, label in [(init,GREY,'视觉模型初始化'), (obs,BLUE,'视觉编码 + prefill'),
                                  (other,PURPLE,'搬移 / 释放等'), (decode,GOLD,'CPU decode')]:
        ax.barh(np.arange(4), values, left=left, height=.49, color=color, label=label)
        left += values
    for y, value in enumerate(total):
        ax.text(value+.3, y, f'{value:.1f} s', va='center', fontsize=12)
    ax.set(yticks=np.arange(4), yticklabels=labels, xlim=(0,25), xlabel='观察到达 → 响应完成（秒）')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.14); ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(-.31,1.0), ncol=2, columnspacing=2)
    fig.text(.05,.12,'每条件 1 次；耗时组成图，非执行时序。包含视觉初始化，不含模型加载和观察前的 A2O。',fontsize=12)
    fig.text(.05,.055,'单次差异受负载与缓存影响，不能据此认定预搬移带来秒级加速。',fontsize=12,color='#526271')
    save(fig,'01_latency_breakdown')

    fig, axes = plt.subplots(1,2,figsize=(10.8,5.3))
    fig.subplots_adjust(left=.17, right=.94, bottom=.27, top=.69, wspace=.43)
    title(fig,'历史 KV 压缩后，上行变小；回传仍包含本轮新增 KV',
          '历史 141 → 46 token；本轮观察另有 600 token，其中图像 560 token')
    for ax, field, heading, limit in zip(axes,['upload_bytes','download_bytes'],
                                         ['CPU → HTP：历史状态','HTP → CPU：更新后状态'],[6,31]):
        values=np.array([rows[0][field],rows[2][field]])/2**20
        assert rows[0][field]==rows[1][field] and rows[2][field]==rows[3][field]
        ax.hlines([1,0],0,values,color='#D4DBE1',lw=2)
        ax.scatter(values,[1,0],s=90,c=[GREY,TEAL],zorder=3)
        for y, value in zip([1,0],values):
            ax.annotate(f'{value:.2f}',(value,y),xytext=(9,0),textcoords='offset points',va='center',fontsize=12)
        ax.set(yticks=[1,0],yticklabels=['不整理','recent 整理'],xlim=(0,limit),ylim=(-.65,1.65),xlabel='序列化状态（MiB）')
        ax.set_title(heading,pad=17)
        ax.grid(axis='x',alpha=.14); ax.set_axisbelow(True)
    fig.text(.05,.115,'有无预搬移的状态字节数相同。MiB = 2²⁰ bytes；图中使用保存的完整序列化状态大小。',fontsize=12)
    fig.text(.05,.05,'这不是实际分配的 KV buffer 容量；不能把上行压缩比例直接套到回传。',fontsize=12,color='#526271')
    save(fig,'02_kv_transfer_volume')

    before=load('20261103-144959-909906-hybrid-serial-screen-history')
    after=rows[0]
    assert before['generated_ids']==after['generated_ids']
    values=[before,after]
    fig, axes=plt.subplots(1,2,figsize=(10.8,5.3),sharey=True)
    fig.subplots_adjust(left=.23,right=.93,bottom=.27,top=.69,wspace=.22)
    title(fig,'本次对照：释放视觉编码器后，CPU decode 换页减少',
          '同一历史、截图及 17 个输出 token · batch128 · CPU mmap · HTP 队列深度 1')
    durations=[r['timing_ms']['decode_ms']/1000 for r in values]
    faults=[r['resources']['decode_end']['major_faults']-r['resources']['decode_start']['major_faults'] for r in values]
    axes[0].hlines([1,0],0,durations,color='#D4DBE1',lw=2)
    for ax, x, heading, xlabel in [(axes[0],durations,'CPU decode 耗时','秒'),
                                      (axes[1],faults,'decode 期间 major fault','次数（对数刻度）')]:
        ax.scatter(x,[1,0],s=90,c=[GREY,TEAL],zorder=3)
        ax.set(yticks=[1,0],ylim=(-.65,1.65),xlabel=xlabel)
        ax.set_title(heading,pad=17)
        ax.grid(axis='x',alpha=.14); ax.set_axisbelow(True)
    axes[0].set(yticklabels=['保留编码器至 decode 后','prefill 后释放编码器'],xlim=(0,34))
    for y,value in zip([1,0],durations):
        axes[0].annotate(f'{value:.1f}',(value,y),xytext=(9,0),textcoords='offset points',va='center',fontsize=12)
    axes[1].set_xscale('log'); axes[1].set_xlim(100,700000)
    axes[1].set_xticks([100,1000,10000,100000],['100','1k','10k','100k'])
    for y,value in zip([1,0],faults):
        axes[1].annotate(f'{value:,}',(value,y),xytext=(9,0),textcoords='offset points',va='center',fontsize=12)
    fig.text(.05,.115,'每配置 1 次诊断测量；系统负载与缓存未控制，不作为稳定加速比。两组输出内容相同。',fontsize=12)
    fig.text(.05,.05,'major fault 为该进程 decode 结束减开始的累计计数；未完成 / 主动终止的运行未纳入。',fontsize=12,color='#526271')
    save(fig,'03_encoder_lifetime')
    (OUT/'sources.json').write_text(json.dumps({'runs':sources,'figure_qa':qa},indent=2,ensure_ascii=False))
    (OUT/'plot_data.json').write_text(json.dumps({'final_four':rows,'encoder_retained':before},indent=2,ensure_ascii=False))
    print(json.dumps(qa,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
