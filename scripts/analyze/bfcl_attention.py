"""汇总试跑：工具结果在后续 decode 中获得的注意力，绝不据此自动删除 KV。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(root):
    groups = defaultdict(list)
    status = []
    rows = []
    outputs = []
    for folder in sorted(root.glob('multi_turn_long_context_*')):
        if not (folder/'result.json').exists():
            continue
        result = json.loads((folder/'result.json').read_text())
        status.append(result)
        outputs.extend(json.loads((folder/'tools.json').read_text()))
        events = {r['step']: r for r in map(json.loads,(folder/'events.jsonl').read_text().splitlines())}
        for row in json.loads((folder/'observations.json').read_text()):
            row.update(task=folder.name, quality_pass=bool(result['quality'].get('valid')),
                       layer_query_count=row.get('layer_query_count',events[row['step']].get('attention_layer_query_count',0)))
            rows.append(row)
            groups[(folder.name,row['tool_id'])].append(row)
    tools = []
    for (task,tid), values in groups.items():
        for scope in ['all_subsequent_calls','later_user_turns']:
            selected = [r for r in values if scope != 'later_user_turns' or r['turn']>r['tool_turn']]
            if not selected:
                continue
            weight = sum(r['layer_query_count'] for r in selected)
            tools.append({'task':task, 'tool_id':tid, 'scope':scope,
                'quality_pass':selected[0]['quality_pass'], 'tokens':selected[0]['tokens'],
                'observed_calls':len(selected),
                'mean_attention_mass':sum(r['attention_mass']*r['layer_query_count'] for r in selected)/weight,
                'max_call_attention_mass':max(r['attention_mass'] for r in selected)})
    tables = []
    for scope in ['all_subsequent_calls','later_user_turns']:
        for quality in ['all','passed','not_passed']:
            selected = [r for r in tools if r['scope']==scope and
                (quality=='all' or r['quality_pass']==(quality=='passed'))]
            total_tokens = sum(r['tokens'] for r in selected)
            tables.append({'scope':scope,'quality':quality,'observed_tools':len(selected),
                'thresholds':[{'attention_mass_below':t,
                    'tool_fraction':sum(r['mean_attention_mass']<t for r in selected)/len(selected) if selected else None,
                    'tool_token_fraction':sum(r['tokens'] for r in selected if r['mean_attention_mass']<t)/total_tokens if total_tokens else None,
                    'always_below_across_observed_call_means_fraction':sum(r['max_call_attention_mass']<t for r in selected)/len(selected) if selected else None}
                    for t in [.001,.005,.01,.05]]})
    lengths = sorted(r['result_tokens'] for r in outputs)
    step_groups = defaultdict(list)
    for r in rows:
        step_groups[(r['task'],r['step'])].append(r)
    step_stats = [{'task':task,'step':step,'tool_history_token_fraction':sum(r['tokens'] for r in group)/group[0]['prompt_tokens'],
                   'tool_history_attention_mass':sum(r['attention_mass'] for r in group)}
                  for (task,step),group in step_groups.items()]
    return {'finished_tasks':len(status),'quality_pass':sum(bool(r['quality'].get('valid')) for r in status),
        'task_errors':sum(r['error'] is not None for r in status),
        'tool_output_count':len(outputs),
        'tool_output_token_quantiles':{str(q):lengths[round(q*(len(lengths)-1))] if lengths else None for q in [.5,.9,.95,.99,1]},
        'tool_outputs_at_least_1024_tokens':sum(n>=1024 for n in lengths),
        'observation_count':len(rows), 'tables':tables,
        'interpretation':'full-attention decode mass only; low attention does not establish safe retirement; unobserved outputs excluded',
        'tool_summaries':tools,'step_summaries':step_stats}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    report = summarize(args.root)
    (args.root/'attention-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tool_summaries','step_summaries')},ensure_ascii=False,indent=2))
