#!/usr/bin/env python3
"""三条件逐题E2E与质量并列；失败和路由覆盖缺口显式报告。"""
import argparse,json,csv
from pathlib import Path
from statistics import median
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();conditions={};rows=[]
for condition in ['cpu','htp','route']:
 root=a.run/('20261107-route-e2e-'+condition+'-formal-ten')
 if not (root/'summary.json').exists():continue
 summary=json.loads((root/'summary.json').read_text());conditions[condition]=summary
 for result in summary['results']:
  metrics=json.loads((root/result['name']/'metrics.json').read_text())
  decisions=[s.get('backend_timings') or {} for s in metrics['inference']]
  rows.append(dict(condition=condition,scenario=result['name'],trace_complete=result['trace_complete'],score=(result.get('evaluation') or {}).get('similarity'),e2e_ms=result.get('agent_e2e_ms'),inference_calls=result['inference_calls'],tool_calls=result['tool_calls'],htp_requests=sum(d.get('route')=='htp' for d in decisions),covered_requests=sum(d.get('route_reason')=='independent_calibration' for d in decisions),responses=len(decisions)))
if rows:
 with (a.run/'e2e-comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=['# Qwen3.5-9B ToolSandbox 三条件端到端比较','', '原5题保留并新增5题。HTP为显式offload、CPU KV和不支持算子的CPU执行；不是所有算子纯NPU。所有条件同一服务/context4096/batch128。KV整理/预搬移关闭。', '', '| 题目 | CPU 秒 / 分数 | HTP 秒 / 分数 | route 秒 / 分数 | route/CPU | route/HTP |','|---|---:|---:|---:|---:|---:|']
by={(r['condition'],r['scenario']):r for r in rows}
names=list(dict.fromkeys(r['scenario'] for r in rows))
for name in names:
 cells=[]
 for c in ['cpu','htp','route']:
  r=by.get((c,name));cells.append('未完成' if not r or not r['trace_complete'] else f"{r['e2e_ms']/1000:.2f} / {r['score']:.4f}")
 route=by.get(('route',name));ratios=[]
 for c in ['cpu','htp']:
  baseline=by.get((c,name));ratios.append(f"{route['e2e_ms']/baseline['e2e_ms']:.3f}" if route and baseline and route['e2e_ms'] and baseline['e2e_ms'] else '—')
 report.append('| '+name+' | '+' | '.join(cells+ratios)+' |')
report+=['','小于1表示route耗时较少。质量、完整性不同的轨迹不能直接解释为同等任务加速。E2E从首次模板计量准备到最终模型HTTP响应，包含真实工具、网络、host处理和状态快照/迁移，排除模型初始化及官方评分。','']
for c in conditions:
 rr=[r for r in rows if r['condition']==c];valid=[r['e2e_ms'] for r in rr if r['trace_complete'] and r['e2e_ms'] is not None]
 report.append(f"- {c}：完成{len(rr)}/10，轨迹完整{sum(r['trace_complete'] for r in rr)}/10；HTP请求{sum(r['htp_requests'] for r in rr)}，独立成本覆盖请求{sum(r['covered_requests'] for r in rr)}；逐题E2E中位数{median(valid)/1000:.2f}秒。" if valid else f'- {c}：无有效轨迹。')
report+=['','Router使用独立实测精确H/P、D=4计算预测范围，仅ready候选准入。校准失败或无覆盖时记录原因；不把覆盖缺口解释为策略成功。旧预检及CPU bootstrap不纳入正式统计。']
(a.run/'report.md').write_text('\n'.join(report)+'\n');print(a.run/'report.md')
