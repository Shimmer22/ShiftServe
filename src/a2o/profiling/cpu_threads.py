"""CPU线程试验完整性检查及预先定义的全局线程选择。"""
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def load_thread_run(path):
    path=Path(path)
    read=lambda name:[json.loads(x) for x in (path/name).read_text().splitlines()]
    m=json.loads((path/'manifest.json').read_text());c=m['config'];threads=c['cpu_thread_scan']
    completion=json.loads((path/'completion.json').read_text())
    if not completion['passed'] or not completion['all_CPU_thread_candidates_numerically_qualified']:raise ValueError('线程试验未完整通过')
    rows=read('checked-rows.jsonl');raw=read('device/trials.jsonl');http=read('http-server.jsonl');events=read('native-events.jsonl')
    expected={('calibration',p,t,0,0) for p in c['new_tokens'] for t in threads}
    expected|={('measurement',p,t,rep,200) for p in c['new_tokens'] for t in threads for rep in range(c['repetitions'])}
    keys=[(r['phase'],r['P'],r['cpu_prefill_threads'],r['repetition'],r['gap_requested_ms']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected or sorted(r['id'] for r in rows)!=list(range(len(rows))):raise ValueError('线程采样矩阵缺失或重复')
    if len(raw)!=len(rows) or any(any(r[k]!=v for k,v in a.items()) for a,r in zip(raw,rows)):raise ValueError('设备原始记录不一致')
    if not all(r['passed'] and r['response_check']['passed'] and r['eog'] and r['generated_ids_equal'] and r['route']=='CC' and r['cpu_decode_threads']==4 for r in rows):raise ValueError('完整响应或配置检查失败')
    if len({(r['output'],tuple(r['generated_ids']),r['D']) for r in rows})!=1:raise ValueError('完整输出不同')
    for p in c['new_tokens']:
        cal=[r for r in rows if r['phase']=='calibration' and r['P']==p]
        if min(cal,key=lambda r:r['id'])['cpu_prefill_threads']!=4:raise ValueError('参考不是4线程')
        for r in cal:
            lim=c['numeric_acceptance'];ok=r['boundary_min_cosine']>=lim['min_cosine'] and (lim['max_nrmse'] is None or r['boundary_max_nrmse']<=lim['max_nrmse'])
            if not ok or not r['numerical_passed']:raise ValueError('线程数值校准失败')
    hs={r['id']:r for r in http};pub={e['turn']:e for e in events if e['event']=='response_published'};end={e['turn']:e for e in events if e['event']=='trial_completed'}
    if len(http)!=len(rows) or len(hs)!=len(rows):raise ValueError('HTTP次数不同')
    for r in rows:
        event=hs[r['id']]
        if (event['P'],event['gap_requested_ms'])!=(r['P'],r['gap_requested_ms']):raise ValueError('HTTP条件不一致')
        if event['body_sha256']!=hashlib.sha256((path/f'P{r["P"]}.txt').read_bytes()).hexdigest():raise ValueError('正文不同')
        if not r['http_issued_ms']<=r['http_returned_ms']<=r['response_compute_completed_ms']<=pub[r['id']]['monotonic_ms']<=r['response_completed_ms']<=end[r['id']]['monotonic_ms']:raise ValueError('计时顺序错误')
        if abs(r['tool_to_response_ms']-r['actual_gap_ms']-r['response_after_return_ms'])>.01:raise ValueError('主时钟恒等式错误')
    if hashlib.sha256((path/'device/agent').read_bytes()).hexdigest()!=m['binary_sha256']:raise ValueError('二进制不同')
    return m,rows


def select_threads(rows,threads):
    """每个H/P的prefill中位数相对4线程归一化，再对各条件等权几何平均。"""
    groups=sorted({(r['H'],r['P']) for r in rows if r['phase']=='measurement'})
    scores={};cells=[]
    for t in threads:
        ratios=[]
        for h,p in groups:
            sample=lambda count:[r['prefill_ms'] for r in rows if (r['phase'],r['H'],r['P'],r['cpu_prefill_threads'])==('measurement',h,p,count)]
            a,b=sample(t),sample(4)
            if len(a)!=3 or len(b)!=3:raise ValueError('线程选择要求每条件完整3次，不选择成功子集')
            value=st.median(a);base=st.median(b);ratios.append(value/base)
            cells.append(dict(H=h,P=p,threads=t,prefill_median_ms=value,ratio_to_4=value/base))
        scores[t]=math.exp(st.mean(math.log(x) for x in ratios))
    best=min(threads,key=lambda t:(scores[t],t))
    return dict(best_threads=best,selection_metric='equal-weight geometric mean of per-H/P median prefill ratios to 4 threads; lower is better',scores=scores,cells=cells)
