#!/usr/bin/env python3
"""独立复核 τ² 手机 pilot 的完整性、原始计时和结论边界。"""
import collections
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.bfcl_attention import save
from a2o.profiling.pd_sweep import validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'results/20261112-tau2-phone-pilot'
    source=ROOT/'results/20261112-tau2-always-prepare'
    plan=json.loads((out/'plan.json').read_text())
    requests=json.loads((source/'pilot-six/requests.json').read_text())
    assert digest(source/'pilot-six/requests.json')==plan['requests_sha256']
    assert digest(source/'pilot-six/tokenizer-fixtures.json')==plan['fixtures_sha256']
    assert digest(source/'build-gap-v3/agent')==plan['binary_sha256']
    build=json.loads((source/'build-gap-v3/manifest.json').read_text())
    assert all(digest(source/'build-gap-v3/source'/path)==sha for path,sha in build['source_sha256'].items())
    assert len(requests)==6 and len({r['task_key'] for r in requests})==6
    assert collections.Counter(r['domain'] for r in requests)==dict(retail=2,airline=2,telecom=2)
    assert plan['preparation_policy']=='always' and plan['return_policy']=='ready-only'
    assert (plan['context_size'],plan['cpu_threads'],plan['htp_host_threads'])==(16384,6,4)
    assert plan['kv_policy']=='none' and not plan['pre_move']
    seen=set();rows=[];bootstrap=0;cache=collections.Counter();positions=collections.defaultdict(collections.Counter)
    for path in sorted((out/'measurements').glob('*.json')):
        if '.raw.' in path.name:continue
        match=re.fullmatch(r'r(\d+)-b(\d+)\.json',path.name);assert match,path
        repetition,boundary=map(int,match.groups());assert (repetition,boundary) not in seen
        seen.add((repetition,boundary));data=json.loads(path.read_text());body=data['request'];response=data['response']
        assert json.loads(path.with_name(path.stem+'.raw.json').read_text())==data
        r=requests[boundary]
        for key in ['prefix_ids','input_ids','generated_ids']:assert body[key]==r[key]
        assert (len(body['prefix_ids']),len(body['input_ids']),len(body['generated_ids']))==(r['H'],r['P'],r['D'])
        assert r['anchor_fully_reusable'] and r['gap_kind']=='tool'
        assert sum(len(body[k]) for k in ['prefix_ids','input_ids','generated_ids'])<16384
        assert digest(Path(r['source_response']))==r['source_sha256']
        assert body['gap_ms']==r['gap_ms']
        assert response['mode']=='always_prepare_gap_replay_v1'
        assert [row['condition'] for row in response['results']]==body['order']
        assert sorted(body['order'])==['cpu','htp','route']
        assert not body['prediction']['formal_results_used']
        for index,row in enumerate(response['results']):
            positions[(boundary,row['condition'])][index]+=1
            assert (row['H'],row['P'],row['D'])==(r['H'],r['P'],r['D'])
            for key in ['fixture_restore_ms','actual_gap_ms','prepare_ms','restore_ms','prefill_ms','handoff_ms','decode_ms','post_return_ms','retirement_tail_ms','serialized_total_ms']:
                assert math.isfinite(row[key]) and row[key]>=0,(path,key)
            assert row['actual_gap_ms']>=body['gap_ms']
            residual=row['post_return_ms']-sum(row[k] for k in ['restore_ms','prefill_ms','handoff_ms','decode_ms'])
            assert -1e-6<=residual<20,(path,residual)
            assert math.isclose(row['serialized_total_ms'],row['actual_gap_ms']+row['post_return_ms']+row['retirement_tail_ms'],abs_tol=1e-6)
            if row['condition']=='route':
                assert row['always_prepare_started'] and row['availability']=='pending' and row['route']=='cpu'
                assert row['route_reason']=='preparation_pending'
            else:assert row['route']==row['condition'] and not row['always_prepare_started']
            rows.append(dict(boundary=boundary,round=repetition,**row))
        bootstrap+=response['bootstrap_ms'];cache[response['cache']['source']]+=1
    assert seen=={(r,b) for r in range(3) for b in range(6)} and len(rows)==54
    assert all(count==dict({0:1,1:1,2:1}) for count in positions.values())
    checks=json.loads((out/'tokenizer-checks.json').read_text())
    assert len(checks)==12 and all(row['passed'] for row in checks)
    normal=json.loads((out/'normal-response.raw.json').read_text())
    assert normal['choices'][0]['finish_reason']!='length' and normal['choices'][0]['message']['content'].strip().lower()=='ready.'
    for gap,expected in [(0,'cpu'),(200,'htp')]:
        d=json.loads((out/f'validation/gap-{gap}.json').read_text())
        row=next(r for r in d['response']['results'] if r['condition']=='route')
        assert row['route']==expected and row['always_prepare_started']
    calibration=[]
    for path in (out/'calibration').glob('*.json'):
        if '.raw.' in path.name:continue
        d=json.loads(path.read_text());calibration.append(d)
        assert json.loads(path.with_name(path.stem+'.raw.json').read_text())==d
        if d['response']['mode']=='stage_sweep_v1':
            validate(d['request'],d['response'])
            for row in d['response']['rows']:assert row['n_threads']==(6 if row['backend']=='cpu' else 4)
        else:
            assert d['response']['mode']=='state_cost_sweep_v1'
            assert len(d['response']['rows'])==4 and all(row['state_verified'] for row in d['response']['rows'])
    assert len(calibration)==51
    summary=json.loads((out/'summary.json').read_text())
    assert json.loads((out/'completion.json').read_text())==summary
    assert json.loads((out/'status.json').read_text())['phase']=='complete'
    totals={mode:{key:sum(r[key] for r in rows if r['condition']==mode)/1000 for key in
        ['restore_ms','prefill_ms','handoff_ms','decode_ms','post_return_ms','prepare_ms','actual_gap_ms','retirement_tail_ms','serialized_total_ms']} for mode in ['cpu','htp','route']}
    for mode in totals:assert math.isclose(totals[mode]['post_return_ms'],summary['post_return_s'][mode],rel_tol=1e-12)
    pairs=[]
    for b in range(6):
        group=[r for r in rows if r['boundary']==b]
        pairs.append(dict(boundary=b,domain=requests[b]['domain'],H=requests[b]['H'],P=requests[b]['P'],D=requests[b]['D'],
            **{mode:[r['post_return_ms']/1000 for r in sorted(group,key=lambda r:r['round']) if r['condition']==mode] for mode in ['cpu','htp','route']}))
    costs=json.loads((out/'cost-model.json').read_text())
    assert not costs['validated'] and not summary['cost_model_validated']
    observed=[r['prepare_ms'] for r in rows if r['condition']=='route']
    result=dict(data_checks_passed=True,checked_evaluations=len(rows),independent_calibration_requests=len(calibration),
        mechanisms_passed=['normal_response','exact_token_ids_12_sequences','pending_to_cpu','ready_to_htp_smoke','state_byte_verification','balanced_condition_position'],
        native_source_and_binary_hashes_verified=True,source_trajectory_hashes_verified=True,
        totals_seconds=totals,per_boundary_seconds=pairs,cache_sources=dict(cache),bootstrap_seconds=bootstrap/1000,
        route_prepare_ms=dict(min=min(observed),median=statistics.median(observed),max=max(observed)),
        raw_speedup_vs_cpu=summary['speedup_vs_cpu'],raw_speedup_vs_htp=summary['speedup_vs_htp'],
        accepted_as='frozen-token asynchronous-preparation mechanism pilot',
        not_established=['causal routing speedup','validated general H/P cost model','phone task quality','live task E2E'],
        model_error_by_backend={backend:max(r['relative_error'] for r in costs['validation'] if r['backend']==backend) for backend in ['cpu','htp']})
    save(out/'acceptance.json',result)
    completed=datetime.fromtimestamp(json.loads((out/'status.json').read_text())['updated_unix'],timezone(timedelta(hours=8)))
    lines=['# τ²-bench 手机 pilot 验收','',
        '**结论：采集完整性和状态机制通过；可归因的选路加速、通用 H/P 成本模型及手机任务质量未通过验收。**','',
        f'实验完成时间：{completed:%Y-%m-%d %H:%M:%S}（北京时间）。本次重新从原始 JSON 校验，不以 status=complete 代替验收。','',
        '## 完整性与机制','',
        '- 6个不同任务来源边界，三域各2个；3轮×CPU/HTP/Route，54/54条件测量完整。51个独立成本请求完整。',
        '- 原始/校验后JSON一致；精确H/P/D、轨迹SHA、二进制SHA及全部原生源码SHA通过；条件顺序轮换，每个条件在每个位置出现一次。',
        '- 12个HF/GGUF输入和输出序列token核对通过；正常响应为Ready.；短机制检查覆盖pending→CPU、ready→HTP，均不进入正式耗时。',
        '- 独立阶段的状态恢复与有限logits检查通过；迁移/恢复使用原有完整字节核验助手。该检查不等于完整自由生成或任务正确率通过。','',
        '## 原始累计耗时','',
        '| 条件 | 恢复 s | Prefill s | Decode s | 返回后总耗时 s |',
        '|---|---:|---:|---:|---:|']
    for mode in ['cpu','htp','route']:
        t=totals[mode];lines.append(f"| {mode} | {t['restore_ms']:.3f} | {t['prefill_ms']:.3f} | {t['decode_ms']:.3f} | {t['post_return_ms']:.3f} |")
    lines.extend(['',f"累计比值为HTP/Route={summary['speedup_vs_htp']:.4f}，CPU/Route={summary['speedup_vs_cpu']:.4f}。这是原始耗时比值，不作为因果加速结论。",'',
        'Route的18/18次均启动准备，返回时18/18为pending，最终18/18走CPU，正式HTP→CPU切换为0次。',
        '因此相对HTP的优势主要体现CPU decode更快，未证明异构路由收益。CPU与Route执行相同token的CPU路径；累计差异主要来自prefill，其因果来源没有CPU-vs-CPU控制或设备频率/温度观测支持，不能归因于选路。',
        f"正式准备用时中位数{statistics.median(observed):.1f} ms，范围{min(observed):.1f}–{max(observed):.1f} ms；所选工具等待约2–3 ms。并发准备成本不能用孤立M(H)的中位数代替。",
        f"Route准备累计{totals['route']['prepare_ms']:.3f} s，与返回后CPU执行重叠，不能重复加到响应耗时；清理尾部累计{totals['route']['retirement_tail_ms']:.3f} s，已单列。",
        f"历史bootstrap累计{bootstrap/1000:.3f} s，另有夹具恢复与缓存读写；不混入上述目标耗时。",'',
        '## M(H) 与成本模型','',
        '| H | CPU→HTP准备 ms | HTP→CPU迁移 ms |',
        '|---:|---:|---:|'])
    for r in costs['state']:lines.append(f"| {r['H']} | {r['prepare_ms']:.2f} | {r['htp_to_cpu_ms']:.2f} |")
    lines.extend(['','context=16384，CPU6线程、HTP主机4线程；状态成本为正式3次中位数，预热另记。返回迁移的状态长度为H+40，包含源导出及完整字节核验。',
        '失败的是以P64测到的H比例缩放其他P的可分离模型：CPU最大留出误差43.47%，HTP最大2.46%。这不表示M(H)原始点失效，也不能据此宣称误差全由H造成。',
        '运行器保留validated=false，并将预测HTP成本设为极大值以阻止未验证模型选HTP。因此这批没有独立验证成本路由器；返回时全pending本身也已决定走CPU。',
        'H4096原47点表仍保留；本轮16K的代表P实测另存。不能把单一H的表或P64比例直接当作通用二维成本曲线。','',
        '## 工作负载与结论范围','',
        '主机30题全部尝试，20题完整结束，14题得到官方reward=1；另外10题不完整，不能只在完整子集上算官方准确率。本地9B用户模拟器属于自定义配置。',
        '全量335个后续边界中，170个工具边界完整锚点匹配，165个用户边界不匹配。手机pilot只覆盖6个可恢复工具边界，并非6个完整任务。',
        '生成token冻结，工具按实测G定时等待；真实执行异步状态准备。不验证手机自由生成质量、历史记忆语义或完整任务E2E。','',
        '设备收尾状态单独记录在device-release-check.json；手机快照保留，日志已回收。原始失败日志与8K停止恢复记录保留。',
        '后续若继续研究收益，应先校验可恢复的等待窗口和CPU二维H/P成本，增加CPU-vs-CPU控制；本次验收不自动启动新实验。'])
    (out/'acceptance.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,evaluations=len(rows),raw_speedup_cpu=summary['speedup_vs_cpu'],raw_speedup_htp=summary['speedup_vs_htp']),indent=2))


if __name__=='__main__':
    main()
