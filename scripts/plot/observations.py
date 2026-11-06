#!/usr/bin/env python3
"""从完整设备重放绘制五张主图；不补值、不模拟重叠、不混合不一致配置。"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.validation.observation import measurement_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# BOOTTIME 与 steady_clock 分开采样，差值含两次读取之间的调度抖动。
# 此容差只用于双时钟诊断；主时钟计时恒等式仍使用 0.01ms，原值不裁剪。
DUAL_CLOCK_READ_SKEW_TOLERANCE_MS = 0.1


def check_dual_clock(row, read_skew_tolerance_ms=DUAL_CLOCK_READ_SKEW_TOLERANCE_MS):
    if (row['actual_gap_ms']+read_skew_tolerance_ms<row['active_actual_gap_ms'] or
            row['suspend_during_tool_to_response_ms']<-read_skew_tolerance_ms or
            abs(row['tool_to_response_ms']-row['active_tool_to_response_ms']-row['suspend_during_tool_to_response_ms'])>.01):
        raise ValueError('双时钟或挂起诊断不一致')


def load_run(path, *, candidate_scan=False):
    """全组校验后再筛选；失败 HC 整个 P 排除，不选成功子集。"""
    read = lambda name: [json.loads(x) for x in (path / name).read_text().splitlines()]
    manifest = json.loads((path / 'manifest.json').read_text())
    cfg = manifest['config']
    if manifest.get('clock_id') != 'CLOCK_BOOTTIME':
        raise ValueError('完整用户等待延迟必须使用包含挂起的 CLOCK_BOOTTIME')
    if candidate_scan:
        if not cfg.get('candidate_scan_only') or cfg.get('candidate_backend') != 'opencl':
            raise ValueError('要求显式OpenCL独立扫描')
    if (not cfg.get('observation_scan') and not candidate_scan) or cfg.get('prepack_comparison') or cfg.get('htp_kv_prepack_dir'):
        raise ValueError('主图要求统一的 observation_scan，关闭额外 KV 表示')
    if not json.loads((path / 'completion.json').read_text())['passed']:
        raise ValueError('实验未完成')
    raw, rows, http, events = map(read, ['device/trials.jsonl', 'checked-rows.jsonl', 'http-server.jsonl', 'native-events.jsonl'])
    expected = {('calibration', p, 0, 'after_return', route, rep)
                for p in cfg['new_tokens'] for route in ['CC', 'HC'] for rep in range(3)}
    grid = [(p,200,'always','HC') for p in cfg['new_tokens']] if candidate_scan else measurement_grid(cfg)
    expected |= {('measurement', p, g, s, route, rep) for p, g, s, route in grid
                 for rep in range(cfg['repetitions'])}
    keys = [(r['phase'], r['P'], r['gap_requested_ms'], r['strategy'], r['requested_route'], r['repetition']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('测量矩阵缺失或重复，拒绝汇总')
    if sorted(r['id'] for r in rows) != list(range(len(expected))):
        raise ValueError('trial ID 缺失或重复')
    if len(raw) != len(rows) or any(any(r[k] != v for k, v in a.items()) for a, r in zip(raw, rows)):
        raise ValueError('设备原始行与主机记录不同')
    if candidate_scan and any(r.get('candidate_backend') != 'opencl' for r in rows):
        raise ValueError('设备实际候选后端不符')
    by_http = {h['id']: h for h in http}
    if len(http) != len(rows) or set(by_http) != {r['id'] for r in rows}:
        raise ValueError('HTTP 记录不完整')
    qualified = {}
    lim = cfg['numeric_acceptance']
    for p in cfg['new_tokens']:
        cal = [r for r in rows if r['phase'] == 'calibration' and r['P'] == p]
        for r in cal:
            ok = r['boundary_min_cosine'] >= lim['min_cosine'] and (lim['max_nrmse'] is None or r['boundary_max_nrmse'] <= lim['max_nrmse'])
            if bool(r['numerical_passed']) != ok:
                raise ValueError('数值准入与 manifest 不一致')
            if r['route']=='CC' and not ok:
                raise ValueError('CPU 参考自身数值不一致')
        qualified[p] = all(r['numerical_passed'] and r['passed'] and r['response_check']['passed'] for r in cal)
    accepted = [r for r in rows if r['route'] == 'CC' or qualified[r['P']]]
    if not all(r['passed'] and r['response_check']['passed'] and r['eog'] and r['generated_ids_equal'] for r in accepted):
        raise ValueError('准入路径响应/状态未通过')
    if len({(r['output'], tuple(r['generated_ids']), r['D']) for r in accepted}) != 1:
        raise ValueError('输入条件间完整输出不一致')
    pubs = {e['turn']: e for e in events if e['event'] == 'response_published'}
    ends = {e['turn']: e for e in events if e['event'] == 'trial_completed'}
    for r in rows:
        h = by_http[r['id']]
        if (h['P'], h['gap_requested_ms']) != (r['P'], r['gap_requested_ms']):
            raise ValueError('HTTP 条件不匹配')
        if h['body_sha256'] != hashlib.sha256((path / f'P{r["P"]}.txt').read_bytes()).hexdigest():
            raise ValueError('HTTP 正文 hash 不一致')
        if not (r['http_issued_ms'] <= r['http_returned_ms'] <= r['response_compute_completed_ms'] <= pubs[r['id']]['monotonic_ms'] <= r['response_completed_ms'] <= ends[r['id']]['monotonic_ms']):
            raise ValueError('设备事件顺序不成立')
        if abs(r['tool_to_response_ms'] - r['actual_gap_ms'] - r['response_after_return_ms']) > .01:
            raise ValueError('设备同一时钟的计时恒等式不成立')
        # 独立时钟读取可被调度打断；候选扫描保留原值并容许1ms诊断偏差。
        # 主CLOCK_BOOTTIME恒等式/事件顺序仍使用原严格检查。
        check_dual_clock(r, read_skew_tolerance_ms=1.0 if candidate_scan else DUAL_CLOCK_READ_SKEW_TOLERANCE_MS)
        if pubs[r['id']]['output'] != r['output']:
            raise ValueError('发布输出不一致')
    if hashlib.sha256((path / 'device/agent').read_bytes()).hexdigest() != manifest['binary_sha256']:
        raise ValueError('运行二进制 hash 不一致')
    return manifest, rows, qualified, events


def sample(rows, h, p, g, s, route):
    return [r for r in rows if r['phase'] == 'measurement' and
            (r['H'], r['P'], r['gap_requested_ms'], r['strategy'], r['requested_route']) == (h, p, g, s, route)]


def median(rows, field):
    return st.median(r[field] for r in rows)


def oracle_regret(rows, h, p, gap, qualified):
    """严格选路比较返回后成本；联合候选保留完整重放及原始trial ID。"""
    actual=sample(rows,h,p,gap,'cost_aware','router')
    cc=sample(rows,h,p,gap,'cost_aware','CC');hc=sample(rows,h,p,gap,'cost_aware','HC')
    matched=qualified;replays=[]
    for a in actual:
        c=next(r for r in cc if r['repetition']==a['repetition'])
        t=next((r for r in hc if r['repetition']==a['repetition']),None)
        candidates=[a];same=True
        if a['route']=='HC':
            if (c['ready_at_return'],c['prepared_early'])==(a['ready_at_return'],a['prepared_early']):candidates.append(c)
            else:same=False
        elif qualified and a['ready_at_return']:
            if t and t['ready_at_return'] and t['prepared_early']==a['prepared_early']:candidates.append(t)
            else:same=False
        matched=matched and same
        best=min(candidates,key=lambda r:r['response_after_return_ms'])
        replays.append(dict(repetition=a['repetition'],actual_trial_id=a['id'],candidate_trial_ids=[r['id'] for r in candidates],
                            selected_trial_id=best['id'],after_return_ms=best['response_after_return_ms'],state_matched=same))
    joint=[]
    for s in ['after_return','always','cost_aware']:
        for route in ['CC','HC','router']:
            group=sample(rows,h,p,gap,s,route)
            if not group or any(r['route']=='HC' for r in group) and not qualified:continue
            joint.append(dict(strategy=s,requested_route=route,trial_ids=[r['id'] for r in group],
                              median_total_ms=median(group,'tool_to_response_ms'),median_actual_gap_ms=median(group,'actual_gap_ms'),
                              ready_count=sum(r['ready_at_return'] for r in group)))
    best=min(joint,key=lambda r:r['median_total_ms'])
    return dict(H=h,P=p,gap_target_ms=gap,
        routing_regret_ms=median(actual,'response_after_return_ms')-st.median(r['after_return_ms'] for r in replays) if matched else None,
        joint_empirical_gap_ms=median(actual,'tool_to_response_ms')-best['median_total_ms'] if qualified else None,
        actual_gap_ms=median(actual,'actual_gap_ms'),matched=matched,routing_replays=replays,
        joint_candidates=joint,joint_oracle=best)


def audit_geometry(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Locator 会保留视窗之外的 tick Text；这些字实际上不绘制，不能当成越界。
    hidden_ticks = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo <= tick.get_loc() <= hi:
                    hidden_ticks.update([tick.label1, tick.label2])
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t not in hidden_ticks and t.get_visible() and t.get_text().strip() and t.get_window_extent(renderer).width > 0]
    collisions = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts)
                  for b, bb in texts[i+1:] if ba.overlaps(bb)]
    outside = [t.get_text() for t, box in texts if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1]
    return dict(text_collisions=collisions, outside_canvas=outside)


def save(fig, out, name, qa):
    qa[name] = audit_geometry(fig)
    fig.savefig(out / (name + '.png'), dpi=300)
    fig.savefig(out / (name + '.pdf'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--legacy-preview', action='store_true', help='只画已有三张 Observation，系统对照仍需新批次')
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    skill = Path.home() / '.codex/skills/figure-style/kernel.py'
    spec = importlib.util.spec_from_file_location('figure_style', skill)
    style = importlib.util.module_from_spec(spec); spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(10, 9, 8))
    manifests = []; rows = []; sources = []; q = {}; qa = {}; initializations = []
    constants = None
    for path in args.runs:
        if args.legacy_preview:
            from a2o.validation.oracle import verify_oracle_run
            audit = verify_oracle_run(path)
            m = json.loads((path/'manifest.json').read_text())
            r = [json.loads(x) for x in (path/'checked-rows.jsonl').read_text().splitlines()]
            events = [json.loads(x) for x in (path/'native-events.jsonl').read_text().splitlines()]
            qualified = {int(k):v for k,v in audit['HC_qualified'].items()}
        else:
            m, r, qualified, events = load_run(path)
        c = m['config']
        signature = {k: c.get(k) for k in ['context_size', 'batch_size', 'htp_queue', 'htp_opbatch', 'numeric_acceptance', 'kv_policy', 'pre_move', 'cpu_repack', 'release_calibration_state', 'input_fixture', 'post_tool_routing', 'system_inputs', 'gap_ms', 'new_tokens']}
        signature.update(binary_sha256=m['binary_sha256'], model=m['model'], history_backend=m.get('history_backend', 'cpu'))
        signature['clock_id']=m.get('clock_id','legacy_steady_clock')
        signature['device_identity']=m.get('device_identity')
        signature['device_files_sha256']=m.get('device_files_sha256')
        if args.legacy_preview:
            for key in ['new_tokens', 'gap_ms']:
                signature.pop(key)
        if constants is not None and constants != signature:
            raise ValueError('跨 H 的模型、二进制或内存预算/协议不一致')
        constants = signature
        if c['history_tokens'] in q and not args.legacy_preview:
            raise ValueError('同 H 重复运行必须显式选择，禁止按性能择优')
        q.setdefault(c['history_tokens'], {}).update(qualified)
        if args.legacy_preview:
            rows = [old for old in rows if not (old['H']==c['history_tokens'] and old['P'] in c['new_tokens'])]
        manifests.append(m); rows.extend(r)
        sources.append(dict(directory=str(path.resolve()), qualified=qualified,
                            hashes={f: hashlib.sha256((path/f).read_bytes()).hexdigest() for f in ['manifest.json', 'checked-rows.jsonl', 'http-server.jsonl', 'native-events.jsonl']}))
        initializations.append(dict(H=c['history_tokens'], **next(e for e in events if e['event']=='session_ready')))
    hs = sorted(q); cfg = manifests[0]['config']
    ps = sorted({p for m in manifests for p in m['config']['new_tokens']})
    g = cfg.get('scan_gap_ms', 200)
    system = cfg.get('system_inputs', [31,32,33]); gaps = cfg['gap_ms']
    valid = [r for r in rows if r['route'] == 'CC' or q[r['H']][r['P']]]
    if len({(r['output'], tuple(r['generated_ids']), r['D']) for r in valid}) != 1:
        raise ValueError('跨 H 输出不一致')
    data = dict(sources=sources, constants=constants, initialization=initializations, rows=rows,
                plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                figure_style_sha256=hashlib.sha256(skill.read_bytes()).hexdigest(),
                matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                dual_clock_read_skew_tolerance_ms=DUAL_CLOCK_READ_SKEW_TOLERANCE_MS)
    (out/'plot-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    suspension=[]
    for key in sorted({(r['H'],r['P'],r['gap_requested_ms'],r['strategy'],r['requested_route']) for r in valid if r['phase']=='measurement' and not args.legacy_preview}):
        group=sample(valid,*key)
        active=[r['active_tool_to_response_ms']-r['active_actual_gap_ms'] for r in group]
        full=[r['response_after_return_ms'] for r in group]
        suspension.append(dict(H=key[0],P=key[1],gap_target_ms=key[2],strategy=key[3],requested_route=key[4],
            mean_full_response_ms=st.mean(full),mean_active_response_ms=st.mean(active),
            mean_suspend_response_ms=st.mean(f-a for f,a in zip(full,active)),trial_ids=[r['id'] for r in group]))
    (out/'suspension-summary.json').write_text(json.dumps(suspension,ensure_ascii=False,indent=2)+'\n')
    colors = {h: c for h, c in zip(hs, ['#0072B2', '#D55E00', '#009E73'])}
    # 1. 全扫描与 31/32/33 原始细节；HC 未准入点留缺口。
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.8), layout='constrained', gridspec_kw={'width_ratios':[1.7,1]})
    for ax, scan in zip(axs, [ps, [31,32,33]]):
        for h in hs:
            for s, route, ls in [('after_return','CC','-'), ('always','HC','--')]:
                points = [sample(valid,h,p,g,s,route) for p in scan]
                y = [median(x,'response_after_return_ms')/1000 if x else np.nan for x in points]
                route_name='CPU/CPU' if route=='CC' else 'HTP/CPU'
                ax.plot(scan, y, ls, marker='D' if route=='CC' else 's', markersize=3.5, color=colors[h], label=f'H={h}, {route_name}')
                for p, group in zip(scan,points):
                    ax.scatter([p]*len(group),[r['response_after_return_ms']/1000 for r in group],s=8,alpha=.25,color=colors[h])
        ax.set_xlabel('New input P (tokens)'); ax.margins(.06)
    ticks = [p for p in [16,24,32,40,48,56,64,128,256,512,1024] if min(ps)<=p<=max(ps)]
    axs[0].set_xscale('log',base=2); axs[0].set_xticks(ticks,[str(p) for p in ticks])
    axs[1].set_xticks([31,32,33]); axs[0].set_ylabel('Active-clock response after Tool (s)' if args.legacy_preview else 'Full response after Tool return (s)')
    axs[0].set_title('Available CPU / HTP scan' if args.legacy_preview else 'CPU / HTP across P=16–1024'); axs[1].set_title('Detail around P=32')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2*len(hs))
    save(fig,out,'01_backend_boundary',qa)
    # 2. 用均值相加保持每根堆叠柱与实际总时间守恒；原始总时长点保留。
    fields = ['observation_processing_ms','wait_or_prepare_after_return_ms','prefill_ms','handoff_checked_ms','decode_ms','other_ms']
    labels = ['Observation','Blocking preparation','Prefill','State handoff','Decode','Other']
    palette = ['#BBBBBB','#CC79A7','#56B4E9','#E69F00','#009E73','#666666']
    fig, axs = plt.subplots(1,len(hs),figsize=(12,4.1),layout='constrained',squeeze=False)
    prepare_table = []
    for ax,h in zip(axs[0],hs):
        xlabels=[]; groups=[]
        for p in system:
            for s,route,name in [('after_return','CC','CC'),('after_return','HC','Late HC'),('always','HC','Early HC')]:
                group=sample(valid,h,p,g,s,route); groups.append(group); xlabels.append(f'P{p}\n{name}')
                prepare_table.append(dict(H=h,P=p,policy=name,prepare_ms=median(group,'prepare_ms') if group else None))
        bottoms=np.zeros(len(groups))
        for field,label,color in zip(fields,labels,palette):
            values=[]
            for group in groups:
                if not group: values.append(np.nan); continue
                if field=='other_ms':
                    vals=[r['response_after_return_ms']-sum(r[f] for f in fields[:-1]) for r in group]
                    if min(vals)<-.01: raise ValueError('阶段分解超出完整响应')
                else: vals=[r[field] for r in group]
                values.append(st.mean(vals)/1000)
            ax.bar(np.arange(len(groups)),values,bottom=bottoms,color=color,width=.72,label=label)
            bottoms+=np.nan_to_num(values)
        for i,group in enumerate(groups):
            ax.scatter([i]*len(group),[r['response_after_return_ms']/1000 for r in group],s=10,color='black',zorder=5)
            if not group: ax.text(i,0,'n.d.',ha='center')
        ax.set_xticks(range(len(groups)),xlabels,rotation=45,ha='right'); ax.set_title(f'H={h}'); ax.set_ylim(bottom=0)
    axs[0,0].set_ylabel('Active-clock response after Tool (s)' if args.legacy_preview else 'Full response after Tool return (s)')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(palette,labels)],loc='outside upper center',ncol=6)
    save(fig,out,'02_cost_breakdown',qa)
    # 3. 横坐标严格使用每次设备测得的 G；线连接组内中位数，不按服务器延迟作图。
    fig,axs=plt.subplots(1,len(hs),figsize=(9.5,3.7),layout='constrained',squeeze=False)
    for ax,h in zip(axs[0],hs):
        for s,label,color in [('after_return','Prepare after Tool','#777777'),('always','Prepare during Tool','#0072B2')]:
            points=[]
            for gap in gaps:
                group=sample(valid,h,32,gap,s,'HC')
                if not group: continue
                ax.scatter([r['actual_gap_ms'] for r in group],[r['response_after_return_ms'] for r in group],alpha=.45,s=18,color=color)
                points.append((median(group,'actual_gap_ms'),median(group,'response_after_return_ms')))
            if points:
                x,y=zip(*sorted(points)); ax.plot(x,y,marker='D',color=color,label=label)
        ax.set_title(f'H={h}, P=32'); ax.set_xlabel('Active-clock Tool gap (ms)' if args.legacy_preview else 'Actual device Tool gap G (ms)'); ax.margins(.08)
    axs[0,0].set_ylabel('Active-clock response after Tool (ms)' if args.legacy_preview else 'Full response after Tool return (ms)'); axs[0,0].legend()
    save(fig,out,'03_actual_gap_overlap',qa)
    if args.legacy_preview:
        (out/'geometry-qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
        (out/'README.md').write_text('# 已有数据 Observation 草图\n\n复核后绘制三张图，H4096 覆盖 P24/31/32/33/40/48/56，H2048 只有31/32/33。H2048缺测位置保留缺口，不外推。每条件n=3。这里只是已有覆盖，P16–1024 的统一主图正在补跑。\n\n旧曲线和 Gap 均使用排除系统挂起的 steady_clock，图中已标记 Active-clock；不能作为完整用户等待延迟。2026-11-04只读探针5.00秒steady对应14.34秒同机BOOTTIME，新主图需统一复测。成本柱为阶段均值，黑点为各次完整响应；提前准备只把返回后阻塞计入堆叠。当前样本不能证明稳定的重叠增益。\n\n不绘制旧版“返回后Router”作为Post-tool策略图；它只允许ready HTP，恒选CPU，不满足对照语义。\n')
        print(json.dumps(dict(output=str(out),preview=True,trials=len(rows),geometry=qa),ensure_ascii=False))
        return
    # 4. 五策略统一从设备发起 Tool 到发布完整响应，禁止使用返回后指标冒充端到端。
    policies=[('after_return','CC','CC'),('after_return','HC','HC'),('after_return','router','Post-tool'),('always','router','Always prep.'),('cost_aware','router','Cost-aware')]
    fig,axs=plt.subplots(len(hs),len(system),figsize=(11,3.5*len(hs)),layout='constrained',squeeze=False)
    comparisons=[]
    for row,h in enumerate(hs):
        for ax,p in zip(axs[row],system):
            for i,(s,route,label) in enumerate(policies):
                group=sample(valid,h,p,g,s,route)
                color='#0072B2' if s=='cost_aware' else '#888888'
                if group:
                    ax.scatter(np.linspace(i-.09,i+.09,len(group)),[r['tool_to_response_ms']/1000 for r in group],s=19,color=color,alpha=.55)
                    ax.hlines(median(group,'tool_to_response_ms')/1000,i-.22,i+.22,color=color,lw=2)
                    comparisons.append(dict(H=h,P=p,gap_target_ms=g,policy=label,median_total_ms=median(group,'tool_to_response_ms'),
                        median_actual_gap_ms=median(group,'actual_gap_ms'),median_after_return_ms=median(group,'response_after_return_ms'),
                        trial_ids=[r['id'] for r in group],actual_routes=[r['route'] for r in group]))
                else: ax.text(i,0,'n.d.',ha='center')
            ax.set_xticks(range(5),['CPU/CPU','HTP/CPU','Post-tool\nrouting','Always\nprepare','Cost-aware']); ax.set_title(f'H={h}, P={p}'); ax.margins(x=.08,y=.12)
            if ax is axs[row,0]: ax.set_ylabel('Tool issue to full response (s)')
    save(fig,out,'04_end_to_end_policies',qa)
    # 5. 同准备状态的严格选路 Oracle，与允许等待的经验联合 Oracle 分开。
    regrets=[]
    for h in hs:
        for p in system:
            for gap in gaps:
                regrets.append(oracle_regret(valid,h,p,gap,q[h][p]))
    fig,axs=plt.subplots(1,2,figsize=(11,3.9),layout='constrained')
    workloads=[(h,p) for h in hs for p in system]
    for ax,field,title in zip(axs,['routing_regret_ms','joint_empirical_gap_ms'],['State-matched routing Oracle','Joint replay Oracle (waiting allowed)']):
        for i,gap in enumerate(gaps):
            values=[next(r[field] for r in regrets if (r['H'],r['P'],r['gap_target_ms'])==(h,p,gap)) for h,p in workloads]
            xx=np.arange(len(workloads))+(i-1)*.2
            yy=[v if v is not None else np.nan for v in values]
            ax.scatter(xx,yy,label=f'Tool target {gap} ms',marker=['o','s','D'][i],s=24)
            for x,v in zip(xx,values):
                if v is None: ax.text(x,0,'×',ha='center',va='bottom',color='#888888')
        ax.axhline(0,color='#BBBBBB',lw=.7); ax.set_xticks(range(len(workloads)),[f'H{h}\nP{p}' for h,p in workloads],rotation=35,ha='right')
        ax.set_title(title); ax.set_ylabel('Regret after Tool return (ms)' if field=='routing_regret_ms' else 'Full-latency regret (ms)'); ax.margins(.08)
    axs[0].scatter([],[],marker='x',color='#888888',label='State unmatched')
    axs[0].legend(); save(fig,out,'05_oracle_gap',qa)
    (out/'comparisons.json').write_text(json.dumps(dict(policies=comparisons,regret=regrets,preparation=prepare_table),ensure_ascii=False,indent=2)+'\n')
    (out/'geometry-qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    with (out/'measurements.csv').open('w') as f:
        names=['H','P','gap_requested_ms','actual_gap_ms','strategy','requested_route','route','repetition','response_after_return_ms','tool_to_response_ms','active_actual_gap_ms','active_tool_to_response_ms','suspend_during_tool_to_response_ms','prepare_ms','observation_processing_ms','prefill_ms','handoff_checked_ms','decode_ms','ready_at_return']
        w=csv.DictWriter(f,fieldnames=names,extrasaction='ignore');w.writeheader();w.writerows(r for r in rows if r['phase']=='measurement')
    lines=['# Observation 与系统收益主图','',f'来源 {len(manifests)} 个串行批次，共 {len(rows)} 条响应记录。数值校准与正式测量分开；每条件 n=3 次独立重放。所有准入路径完整输出相同，D=6（含 EOG）。',
        '', '统一 Qwen2.5-VL-3B Q4_0、F16 KV、ctx6144、batch128，CPU 权威历史、CPU decode；同一进程内 CPU/HTP 双上下文。无额外 tiles、预搬移或 KV 整理。各策略预算相同，初始化及共同历史建立不在 Tool 发起到响应的计时范围内，原始时间单列于 plot-data.json。',
        '', '## 图与读法','',
        '1. `01_backend_boundary`：P16–1024 全扫描与 P31/32/33 细节；实测点及中位数，CPU 为实线，HTP prefill＋CPU decode 为虚线。比较固定服务器目标200ms下实际返回后的完整响应，不把核阶跃推广成连续曲面。',
        '2. `02_cost_breakdown`：堆叠采用阶段墙钟均值，包含阶段内系统挂起，黑点是各次完整响应；准备只计返回后阻塞部分，后台总准备时间见 comparisons.json；交接包含状态检查；Other 为同次完整时间减已列阶段的余额。初始化/共同历史不被虚构成每轮成本。双时钟拆出的响应内总挂起单列 suspension-summary.json，不能把含挂起的阶段差异全部归因于算子或机制。',
        '3. `03_actual_gap_overlap`：横轴是每次设备 CLOCK_BOOTTIME 实测 G（包含挂起）；线连接各目标组的实测中位数，不使用服务端目标作为横坐标。三个 G 条件只能判断已有跨度，无法据此宣称精准隐藏阈值。',
        '4. `04_end_to_end_policies`：统一 Tool 发起到完整 JSON 发布；CC 不提前准备，HC 在返回后按需准备；Post-tool 按包含准备成本的独立校准选路；Always Prepare 和 Cost-aware 均为实际在线 Router。服务器目标200ms，实际 G 各次记录，不能逐次视为完全配对。',
        '5. `05_oracle_gap`：左为同准备状态、从Tool返回计时的选路反事实，状态无法匹配时 × 表示不可识别；右为全部九路完整重放、允许等待的联合经验最小值，从Tool发起计时。联合候选包含实际策略自身，也包含固定提前HC等强制路径；候选和所选trial ID均保留。串行重放实际G不同，三次中位数取最小有选择偏差，只能用于筛查空间。Oracle不拼接阶段最小值。',
        '', '## 结论边界','',
        '小样本用于首轮筛查；三次重放不足以支撑几十毫秒的稳定收益。没有图像或真实外部 Tool 工作负载，只有真实 HTTP 固定文本微基准。独立历史记忆与实际图像验收沿用独立入口，本次不宣称其通过。P16–1024 是13点稀疏扫描，不是每个整数。未准入 HC 整组排除，不以成功子集作图。',
        '', '初步统计：','']
    for h in hs:
        lines.append(f'- H={h}，HC 准入 {sum(q[h].values())}/{len(q[h])} 个 P 条件。')
    for h in hs:
        for p in system:
            cmp={r['policy']:r['median_total_ms'] for r in comparisons if r['H']==h and r['P']==p}
            if 'Cost-aware' in cmp:
                lines.append(f'- H{h}/P{p}：Cost-aware {cmp["Cost-aware"]:.1f}ms；CC {cmp["CC"]:.1f}ms；Post-tool {cmp["Post-tool"]:.1f}ms；Always Prepare {cmp["Always prep."]:.1f}ms。')
    lines += ['', '来源：']+['- '+s['directory'] for s in sources]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),trials=len(rows),qualified=q,geometry=qa),ensure_ascii=False))


if __name__=='__main__': main()
