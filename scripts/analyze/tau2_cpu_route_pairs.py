#!/usr/bin/env python3
"""CPU/Route同边界同轮配对；以边界为独立单位，保留统计方法敏感性。"""
import itertools
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]


def quantile(values,p):
    values=sorted(values);index=(len(values)-1)*p;low=int(index)
    return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(index-low)


def main():
    out=ROOT/'results/20261112-tau2-phone-pilot'
    groups={b:[] for b in range(6)}
    for path in sorted((out/'measurements').glob('r*-b*.json')):
        if '.raw.' in path.name:continue
        repetition,boundary=map(int,re.fullmatch(r'r(\d+)-b(\d+)',path.stem).groups())
        d=json.loads(path.read_text());r={row['condition']:row for row in d['response']['results']}
        cpu,route=r['cpu']['post_return_ms']/1000,r['route']['post_return_ms']/1000
        assert r['route']['route']=='cpu'
        groups[boundary].append(dict(round=repetition,cpu_s=cpu,route_s=route,saved_s=cpu-route,
            prefill_saved_s=(r['cpu']['prefill_ms']-r['route']['prefill_ms'])/1000,
            decode_saved_s=(r['cpu']['decode_ms']-r['route']['decode_ms'])/1000))
    per_boundary=[];means=[]
    for b,g in groups.items():
        assert len(g)==3
        ds=[r['saved_s'] for r in g];mean=statistics.mean(ds);se=statistics.stdev(ds)/math.sqrt(3)
        # t(.975,df=2)；每个边界仅3轮，属于探索性区间。
        ci=[mean-4.3026527299*se,mean+4.3026527299*se];means.append(mean)
        per_boundary.append(dict(boundary=b,route_faster=sum(x>0 for x in ds),route_slower=sum(x<0 for x in ds),
                                 saved_s=ds,mean_saved_s=mean,mean_saved_s_t95=ci,crosses_zero=ci[0]<=0<=ci[1]))
    rows=[r for g in groups.values() for r in g];deltas=[r['saved_s'] for r in rows]
    mean=statistics.mean(means);se=statistics.stdev(means)/math.sqrt(6)
    cluster_ci=[mean-2.5705818356*se,mean+2.5705818356*se]  # df=5
    naive_se=statistics.stdev(deltas)/math.sqrt(18)
    naive_ci=[mean-2.1098155778*naive_se,mean+2.1098155778*naive_se]  # df=17
    flips=[statistics.mean(x*s for x,s in zip(means,signs)) for signs in itertools.product([-1,1],repeat=6)]
    signflip_p=sum(abs(v)>=abs(mean)-1e-12 for v in flips)/len(flips)
    boot=[];ratios=[]
    cpu=[statistics.mean(r['cpu_s'] for r in groups[b]) for b in range(6)]
    route=[statistics.mean(r['route_s'] for r in groups[b]) for b in range(6)]
    # 穷举6^6种按边界重采样；同一边界的3轮作为一簇保留。
    for ids in itertools.product(range(6),repeat=6):
        boot.append(statistics.mean(means[i] for i in ids))
        ratios.append(sum(cpu[i] for i in ids)/sum(route[i] for i in ids))
    result=dict(definition='saved_s = CPU post_return_s - Route post_return_s; positive means Route faster',
        pairs=rows,per_boundary=per_boundary,route_faster=sum(x>0 for x in deltas),route_slower=sum(x<0 for x in deltas),
        mean_saved_s=mean,median_saved_s=statistics.median(deltas),saved_s_range=[min(deltas),max(deltas)],
        aggregate_elapsed_reduction_percent=100*sum(deltas)/sum(r['cpu_s'] for r in rows),
        boundary_mean_t95=cluster_ci,boundary_mean_t95_crosses_zero=cluster_ci[0]<=0<=cluster_ci[1],
        exploratory_boundary_signflip_p=signflip_p,
        naive_18_independent_t95=naive_ci,
        boundary_percentile_bootstrap95=[quantile(boot,.025),quantile(boot,.975)],
        boundary_percentile_bootstrap_ratio95=[quantile(ratios,.025),quantile(ratios,.975)],
        top_three_positive_differences_share_of_net_saved=sum(sorted(deltas,reverse=True)[:3])/sum(deltas),
        prefill_share_of_net_saved=sum(r['prefill_saved_s'] for r in rows)/sum(deltas),
        limitations='仅6个预选边界，每边界3次顺序平衡观测；无CPU重复空白对照，无随机化设备状态干预；统计结果对方法敏感，不作因果或全benchmark推断。')
    (out/'cpu-route-paired.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# CPU 与 Route 配对差异','',
        '正差定义为CPU耗时−Route耗时，正数表示Route更快。按同边界同轮配对。',
        f"18对中Route快{result['route_faster']}对、慢{result['route_slower']}对。累计耗时减少{result['aggregate_elapsed_reduction_percent']:.2f}%，对应1.082×比值。",'',
        '| 边界 | 快/慢次数 | 每次平均节省 s | 95% t区间 s | 跨零 |','|---|---:|---:|---:|---|']
    for r in per_boundary:
        lo,hi=r['mean_saved_s_t95'];lines.append(f"| {r['boundary']} | {r['route_faster']}/{r['route_slower']} | {r['mean_saved_s']:.2f} | [{lo:.2f}, {hi:.2f}] | 是 |")
    lines.extend(['',f'以6个边界的三轮均差为单位，平均每次节省{mean:.2f}s，95% t区间[{cluster_ci[0]:.2f}, {cluster_ci[1]:.2f}]s，跨零。探索性边界符号翻转检验双侧p={signflip_p:.4f}。',
        f"单次差值中位数{statistics.median(deltas):.2f}s，范围[{min(deltas):.2f}, {max(deltas):.2f}]s；三个最大的正差占净节省{100*result['top_three_positive_differences_share_of_net_saved']:.1f}%。",'',
        '## 统计方法敏感性','',
        f'若误把18次重复当成18个独立边界，t区间约[{naive_ci[0]:.3f}, {naive_ci[1]:.3f}]s，刚好不跨零；不能据此忽略组内重复。',
        f"按边界穷举重采样的百分位bootstrap区间为[{result['boundary_percentile_bootstrap95'][0]:.2f}, {result['boundary_percentile_bootstrap95'][1]:.2f}]s，同样不跨零；该方法与小样本t区间结论不同。只有6个簇时，不能挑选有利区间宣称显著。",'',
        '## 代码与解释','',
        '两条CPU路径使用同一个cpu->ctx、相同6线程、输入token、完整锚点恢复、evaluate prefill和逐token decode循环。没有CPU算子、padding、KV整理或线程数优化。Route额外启动HTP状态恢复、检查ready并取消未完成准备。',
        '正式18次Route均pending并选择CPU，无一次异构prefill。差异可能包含运行波动与额外后台工作的间接影响；当前记录无法分离，不能断言全是随机噪声，也不能将其写成已证明优化。',
        '下一步可用完全相同CPU路径的重复空白对照估计波动，再随机交错准备开/关；本分析不启动新设备实验。'])
    (out/'cpu-route-paired.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','per_boundary']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
