"""仅离线读取已存 token/注意力向量，分析 >1024 token 工具结果。"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


def ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i+j-1)/2
        i = j
    return result


def corr(x, y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def spearman(x,y):
    return corr(ranks(x),ranks(y))


def partial(x,y,controls):
    if len(x)<6:
        return None
    design = np.column_stack([np.ones(len(x))]+[ranks(c) for c in controls])
    a,b = ranks(x),ranks(y)
    return corr(a-design@np.linalg.lstsq(design,a,rcond=None)[0],
                b-design@np.linalg.lstsq(design,b,rcond=None)[0])


def stats(values):
    if not values:
        return {'n':0}
    return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
                p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))


def blocks(vector,size):
    starts = np.arange(0,len(vector),size)
    lengths = np.minimum(size,len(vector)-starts)
    mass = np.add.reduceat(vector,starts)
    return mass,lengths


def concentration(vector,size):
    mass,lengths = blocks(vector,size)
    order = np.argsort(-mass,kind='stable')
    total = float(mass.sum())
    result = {}
    for fraction in [.1,.2,.5]:
        count = math.ceil(len(mass)*fraction)
        selected = order[:count]
        result[str(fraction)] = {'share':float(mass[selected].sum()/total) if total else 0.,
            'token_budget':float(lengths[selected].sum()/len(vector)), 'blocks':count}
    return result


def stability(series,size,landmark):
    if landmark=='first_observation':
        index = 0
    else:
        first_turn = series[0]['turn']
        index = max(i for i,s in enumerate(series) if s['turn']==first_turn)
    current = series[index]
    future = [r for r in series[index+1:] if r['turn']>current['turn']]
    if not future:
        return None
    mass,lengths = blocks(current['vector'],size)
    density = mass/lengths
    count = len(mass)
    low = np.argsort(density,kind='stable')[:count//2]
    important = np.zeros(count,dtype=bool)
    strong = np.zeros(count,dtype=bool)
    jump = np.zeros(count,dtype=bool)
    future_peak = np.zeros(count)
    future_share = np.zeros(count)
    for row in future:
        fm,_ = blocks(row['vector'],size)
        fd = fm/lengths
        top = np.argsort(-fd,kind='stable')[:math.ceil(.1*count)]
        important[top] = True
        lift = (fm/max(float(fm.sum()),1e-30))/(lengths/lengths.sum())
        strong[top] |= lift[top]>=2
        jump |= (fm>=.001)&(fd>=4*np.maximum(density,1e-30))
        future_peak = np.maximum(future_peak,fm)
        future_share = np.maximum(future_share,fm/max(float(fm.sum()),1e-30))
    return {'current_step':current['step'],'current_turn':current['turn'],
        'future_calls':len(future),'future_turns':len({r['turn'] for r in future}),
        'low_blocks':len(low),'low_tokens':int(lengths[low].sum()),
        'future_top10_blocks':int(important[low].sum()),
        'future_top10_tokens':int(lengths[low][important[low]].sum()),
        'future_top10_and_lift2_blocks':int(strong[low].sum()),
        'future_top10_and_lift2_tokens':int(lengths[low][strong[low]].sum()),
        'future_mass001_and_density4x_blocks':int(jump[low].sum()),
        'future_mass001_and_density4x_tokens':int(lengths[low][jump[low]].sum()),
        'low_set_current_result_attention_share':float(mass[low].sum()/max(float(mass.sum()),1e-30)),
        'max_future_low_block_absolute_mass':float(future_peak[low].max()),
        'max_future_low_block_result_share':float(future_share[low].max())}


def csv_write(path,rows):
    if rows:
        with path.open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    tool_rows,conc_rows,stable_rows = [],[],[]
    counts = defaultdict(int)
    token_mismatch = []
    for folder in sorted(args.root.glob('multi_turn_long_context_*')):
        passed = bool(json.loads((folder/'result.json').read_text())['quality'].get('valid'))
        tools = {r['id']:r for r in json.loads((folder/'tools.json').read_text()) if r['result_tokens']>1024}
        if not tools:
            continue
        counts['long_results_total'] += len(tools)
        counts['tasks_with_long_results'] += 1
        series = defaultdict(list)
        canonical_ids = {}
        events = list(map(json.loads,(folder/'events.jsonl').read_text().splitlines()))
        checks = {r['step']:r for r in json.loads((folder/'replay-checks.json').read_text())}
        observed = json.loads((folder/'observations.json').read_text())
        weights = {(r['tool_id'],r['step']):r['layer_query_count'] for r in observed}
        for call_index,event in enumerate(events):
            key = event['step']
            file = folder/(key+'.attention.json')
            if not file.exists():
                continue
            inputs = json.loads((folder/(key+'.input.json')).read_text())
            spans = [s for s in inputs['tool_spans'] if s['id'] in tools]
            if not spans:
                continue
            vector = np.asarray(json.loads(file.read_text()),dtype=float)
            if len(vector)!=len(inputs['input_ids']) or np.any(vector<0) or not np.isfinite(vector).all():
                raise ValueError('invalid attention vector: '+str(file))
            for span in spans:
                tid,a,b = span['id'],span['start'],span['end']
                body_ids = inputs['input_ids'][a:b]
                if tid in canonical_ids and body_ids!=canonical_ids[tid]:
                    token_mismatch.append({'task':folder.name,'tool':tid,'step':key})
                    continue
                canonical_ids[tid] = body_ids
                series[tid].append({'step':key,'turn':event['turn'],'call_index':call_index,
                    'prompt_tokens':len(vector),'age_turns':event['turn']-span['turn'],
                    'uniform_density':float(np.mean(1/(len(vector)+np.asarray(checks[key]['sample_indices'],dtype=float)))),
                    'weight':weights[(tid,key)],'vector':vector[a:b].copy()})
        for tid, rows in series.items():
            counts['long_results_observed'] += 1
            total_weight = sum(r['weight'] for r in rows)
            n = len(rows[0]['vector'])
            density = sum(r['weight']*float(r['vector'].mean()) for r in rows)/total_weight
            lift = sum(r['weight']*float(r['vector'].mean())/r['uniform_density'] for r in rows)/total_weight
            base = {'task':folder.name,'tool_id':tid,'passed':passed,
                    'function':tools[tid]['call']['name'],'result_tokens':tools[tid]['result_tokens'],
                    'span_tokens':n,'observed_calls':len(rows)}
            tool_rows.append(dict(base,mean_density=density,mean_uniform_lift=lift,
                mean_prompt_tokens=sum(r['weight']*r['prompt_tokens'] for r in rows)/total_weight,
                mean_turn_age=sum(r['weight']*r['age_turns'] for r in rows)/total_weight))
            for size in [32,64,128]:
                for scope in ['all_observations','later_user_turns']:
                    chosen = rows if scope=='all_observations' else [r for r in rows if r['age_turns']>0]
                    if not chosen:
                        continue
                    w = sum(r['weight'] for r in chosen)
                    cs = [(r,concentration(r['vector'],size)) for r in chosen]
                    cr = dict(base,block_size=size,scope=scope)
                    for fraction in [.1,.2,.5]:
                        label = str(int(fraction*100))
                        cr['top'+label+'_share'] = sum(r['weight']*c[str(fraction)]['share'] for r,c in cs)/w
                        cr['top'+label+'_token_budget'] = sum(r['weight']*c[str(fraction)]['token_budget'] for r,c in cs)/w
                    conc_rows.append(cr)
                for landmark in ['first_observation','end_of_first_observed_turn']:
                    sr = stability(rows,size,landmark)
                    if sr:
                        stable_rows.append(dict(base,block_size=size,landmark=landmark,**sr))
    summary = {'counts':dict(counts),'token_alignment_mismatches':token_mismatch,
        'method':{'filter':'tool.result_tokens > 1024, strict; no inference or new model replay',
            'density':'mean attention per result token, query-weighted within each result, equal-result summaries',
            'uniform_lift':'density / mean_query(1/visible_keys); visible_keys = prompt_tokens + sampled generation index',
            'blocks':'32/64/128 contiguous tokens anchored at result start; last block may be shorter',
            'concentration':'rank blocks by attention mass separately for each call; top ceil(fraction*block_count); average shares within result, then across results',
            'stability':'current bottom floor(50% of blocks) by density; future calls strictly later user turns; first-observation and end-of-first-observed-turn landmarks',
            'strong_importance':'future top10% block density AND >=2x within-result uniform density',
            'absolute_jump':'future block >=0.1% all-key attention AND >=4x current density',
            'censoring':'no later-user-turn observations excluded from temporal denominators; low-ranked ties use stable order'},
        'length_density':[],'concentration':[],'stability':[]}
    for quality in ['all','passed','not_passed']:
        select = lambda rows:[r for r in rows if quality=='all' or r['passed']==(quality=='passed')]
        rows = select(tool_rows)
        x = [r['span_tokens'] for r in rows]
        y = [r['mean_density'] for r in rows]
        z = [r['mean_uniform_lift'] for r in rows]
        controls = [[r[k] for r in rows] for k in ['mean_prompt_tokens','mean_turn_age']]
        functions = sorted({r['function'] for r in rows})
        function_controls = controls+[[int(r['function']==f) for r in rows] for f in functions[1:]]
        bins = []
        for lo,hi in [(1024,2048),(2048,4096),(4096,8192),(8192,16384),(16384,1000000)]:
            selected = [r for r in rows if lo<r['result_tokens']<=hi]
            bins.append({'min_exclusive':lo,'max_inclusive':hi,'results':len(selected),
                'density':stats([r['mean_density'] for r in selected]),
                'uniform_lift':stats([r['mean_uniform_lift'] for r in selected])})
        summary['length_density'].append({'quality':quality,'results':len(rows),'tasks':len({r['task'] for r in rows}),
            'spearman_length_density':spearman(x,y),'spearman_length_uniform_lift':spearman(x,z),
            'partial_spearman_length_density_controlling_prompt_and_age':partial(x,y,controls),
            'partial_spearman_length_density_controlling_prompt_age_and_function':partial(x,y,function_controls),
            'functions':[{'function':f,'results':sum(r['function']==f for r in rows),
                'min_tokens':min(r['result_tokens'] for r in rows if r['function']==f),
                'max_tokens':max(r['result_tokens'] for r in rows if r['function']==f)} for f in functions],
            'bins':bins})
        for size in [32,64,128]:
            for scope in ['all_observations','later_user_turns']:
                chosen = [r for r in select(conc_rows) if r['block_size']==size and r['scope']==scope]
                summary['concentration'].append({'quality':quality,'block_size':size,'scope':scope,
                    'results':len(chosen),'tasks':len({r['task'] for r in chosen}),
                    **{k:stats([r[k] for r in chosen]) for k in ['top10_share','top20_share','top50_share','top10_token_budget','top20_token_budget','top50_token_budget']}})
            for landmark in ['first_observation','end_of_first_observed_turn']:
                chosen = [r for r in select(stable_rows) if r['block_size']==size and r['landmark']==landmark]
                denominator = sum(r['low_blocks'] for r in chosen)
                tokens = sum(r['low_tokens'] for r in chosen)
                summary['stability'].append({'quality':quality,'block_size':size,'landmark':landmark,
                    'results':len(chosen),'tasks':len({r['task'] for r in chosen}),'low_blocks':denominator,'low_tokens':tokens,
                    'future_calls':stats([r['future_calls'] for r in chosen]),
                    **{key:{'blocks':sum(r[key+'_blocks'] for r in chosen),
                        'block_fraction':sum(r[key+'_blocks'] for r in chosen)/denominator if denominator else None,
                        'token_fraction':sum(r[key+'_tokens'] for r in chosen)/tokens if tokens else None,
                        'equal_result_fraction':stats([r[key+'_blocks']/r['low_blocks'] for r in chosen]),
                        'results_with_any':sum(r[key+'_blocks']>0 for r in chosen)}
                       for key in ['future_top10','future_top10_and_lift2','future_mass001_and_density4x']}})
    for name,rows in [('tool-density',tool_rows),('tool-concentration',conc_rows),('tool-stability',stable_rows)]:
        csv_write(args.output/(name+'.csv'),rows)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary['counts']),flush=True)
    print('token_alignment_mismatches',len(token_mismatch),flush=True)


if __name__=='__main__':
    main()
