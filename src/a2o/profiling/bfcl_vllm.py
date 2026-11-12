"""vLLM 并发采集；完整历史、原生工具格式，不执行 KV retire。"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import re
import sys
import time
import traceback
import urllib.request

from a2o.profiling.bfcl_attention import native_calls, save


def schema(value):
    if isinstance(value, dict):
        return {k:({'dict':'object','float':'number','int':'integer','list':'array','bool':'boolean'}.get(v,v)
                   if k=='type' and isinstance(v,str) else schema(v)) for k,v in value.items()}
    return [schema(v) for v in value] if isinstance(value,list) else value


def post(url, data):
    req = urllib.request.Request(url,json.dumps(data).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as response:
        return json.load(response)


def run_case(entry, cfg, out, tokenizer, truth):
    from a2o.workloads.bfcl import sandbox_child, parse_calls, common_prefix
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    folder = out/entry['id']
    folder.mkdir(exist_ok=False)
    save(folder/'entry.json',entry)
    started = time.time()
    docs = {d['name']:d for d in entry['function']}
    tools_schema = [{'type':'function','function':schema({k:d[k] for k in ['name','description','parameters']})}
                    for d in entry['function']]
    messages = [{'role':'system','content':'You are a helpful assistant. Use the provided tools to fulfill the user requests.'}]
    decoded, tools = [], []
    previous_ids, pending = [], []
    error = None
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=sandbox_child,args=(child,entry))
    proc.start()
    quality = {'valid':False,'not_run':True}
    events = (folder/'events.jsonl').open('w',buffering=1)
    try:
        if not parent.poll(30) or not parent.recv().get('ready'):
            raise RuntimeError('sandbox initialization failed')
        for turn, user in enumerate(entry['question']):
            if str(turn) in entry.get('held_docs', {}):
                added = entry['held_docs'][str(turn)]
                docs.update({d['name']: d for d in added})
                tools_schema.extend({'type':'function','function':schema({k:d[k] for k in ['name','description','parameters']})} for d in added)
                user = [{'role':'user','content':json.dumps(added) + '\nI have updated some more functions you can choose from. What about now?'}]
            messages.extend(copy.deepcopy(user))
            decoded.append([])
            for step in range(cfg['max_steps']):
                key = f't{turn:02d}-s{step:02d}'
                prompt = tokenizer.apply_chat_template(messages,tools=tools_schema,tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                encoded = tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
                ids = encoded['input_ids']
                h = common_prefix(previous_ids, ids)
                for tool in pending:
                    tool.update(next_step=key, H_common_prefix=h, P_append=len(ids)-h)
                preceding_tools, pending = pending, []
                (folder/(key+'.prompt.txt')).write_text(prompt)
                if len(ids)+cfg['max_new_tokens']>cfg['context_limit']:
                    raise RuntimeError(f'context_overflow:{len(ids)}; no truncation')
                matches = list(re.finditer(r'\n<tool_response>\n(.*?)\n</tool_response>',prompt,re.S))
                if len(matches)!=len(tools):
                    raise RuntimeError('tool response span count mismatch')
                spans = []
                for match, tool in zip(matches,tools):
                    if match.group(1)!=tool['content'].strip():
                        raise RuntimeError('tool response body mismatch')
                    begin,end = match.span(1)
                    positions = [i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>begin and a<end]
                    if not positions:
                        raise RuntimeError('empty tool body span')
                    spans.append({'id':tool['id'],'turn':tool['turn'],
                                  'start':positions[0],'end':positions[-1]+1})
                save(folder/(key+'.input.json'),{'input_ids':ids,'tool_spans':spans,'turn':turn})
                request = {'model':'qwen35-9b','prompt':ids,'max_tokens':cfg['max_new_tokens'],
                    'temperature':0,'seed':cfg['seed'],'return_token_ids':True,
                    'skip_special_tokens':False,'stop_token_ids':cfg['stop_ids']}
                save(folder/(key+'.request.json'),request)
                t0 = time.time()
                response = post(cfg['endpoint']+'/v1/completions',request)
                save(folder/(key+'.response.json'),response)
                choice = response['choices'][0]
                content = choice['text']
                sampled = choice.get('token_ids')
                if not sampled:
                    raise RuntimeError('vLLM did not return sampled token IDs')
                previous_ids = ids + sampled
                for tool in preceding_tools:
                    tool['D_next'] = len(sampled)
                row = {'H_common_prefix':h, 'P_append':len(ids)-h, 'image_count':0, 'step':key,'turn':turn,'prompt_tokens':len(ids),'generated_ids':sampled,
                       'response':content,'seconds':time.time()-t0,'finish_reason':choice['finish_reason']}
                events.write(json.dumps(row,ensure_ascii=False)+'\n')
                if choice['finish_reason']=='length':
                    raise RuntimeError('generation_limit; raw output retained')
                calls = native_calls(content,docs,parse_calls)
                decoded[-1].append([c['canonical'] for c in calls])
                messages.append({'role':'assistant','content':content})
                if not calls:
                    break
                issued_ns = time.monotonic_ns()
                parent.send(calls)
                if not parent.poll(30):
                    raise TimeoutError('sandbox tool timeout')
                values = parent.recv()
                available_ns = time.monotonic_ns()
                for i,(call,value) in enumerate(zip(calls,values)):
                    record = {'issued_ns':issued_ns,'result_available_ns':available_ns,'G_ms':(available_ns-issued_ns)/1e6,'H_issue':len(previous_ids),'image_count':0,'id':key+f'-tool{i}','turn':turn,'call':call,'content':value['result'],
                              'error':value['error'],'result_tokens':len(tokenizer.encode(value['result'],add_special_tokens=False))}
                    tools.append(record)
                    pending.append(record)
                    messages.append({'role':'tool','content':value['result']})
            else:
                raise RuntimeError('maximum steps exceeded')
        quality = multi_turn_checker(decoded,truth,copy.deepcopy(entry),entry['id'].rsplit('_',1)[0],'a2o_vllm_qwen35')
    except Exception:
        error = traceback.format_exc()
    finally:
        if proc.is_alive():
            try:
                parent.send(None)
            except OSError:
                pass
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        parent.close()
        child.close()
        events.close()
    result = {'id':entry['id'],'quality':quality,'error':error,'seconds':time.time()-started,'tools':len(tools)}
    save(folder/'messages.json',messages)
    save(folder/'decoded.json',decoded)
    save(folder/'tools.json',tools)
    save(folder/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    sys.path.insert(0,cfg['bfcl_root'])
    from a2o.workloads.bfcl import load_entries
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    cfg['stop_ids'] = sorted({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')})
    with urllib.request.urlopen(cfg['endpoint']+'/health',timeout=5) as response:
        assert response.status==200
    entries = load_entries(Path(cfg['bfcl_root']),cfg)
    if cfg.get('tasks_per_category'):
        selected = []
        for category in cfg['categories']:
            group = [e for e in entries if e['id'].rsplit('_',1)[0] == category]
            random.Random(cfg['seed']).shuffle(group)
            selected.extend(group[:cfg['tasks_per_category']])
        entries = selected
    else:
        random.Random(cfg['seed']).shuffle(entries)
    out = Path(args.output)
    out.mkdir(exist_ok=True,parents=True)
    save(out/'manifest.json',dict(cfg,engine='vLLM',prompt='Qwen native XML tools',thinking=False,
        data_sha256={c:hashlib.sha256(Path(cfg['bfcl_root'],'bfcl_eval/data/BFCL_v4_'+c+'.json').read_bytes()).hexdigest() for c in cfg['categories']},
        selected_ids=[e['id'] for e in entries], image_count=0,
        scope='host workload characterization only; no GPU/OpenCL phone backend comparison',
        attention='separate replay stage; not collected by vLLM',kv_policy='none'))
    truths = {}
    for c in cfg['categories']:
        truths.update({r['id']:r['ground_truth'] for r in map(json.loads,Path(cfg['bfcl_root'],
            'bfcl_eval/data/possible_answer/BFCL_v4_'+c+'.json').read_text().splitlines())})
    started = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
        futures = [pool.submit(run_case,e,cfg,out,tokenizer,truths[e['id']]) for e in entries]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            elapsed = time.time()-started
            status = {'stage':'collection','finished':len(results),'total':len(entries),'elapsed_s':elapsed,
                'eta_seconds':elapsed/len(results)*(len(entries)-len(results)),
                'quality_pass':sum(bool(r['quality'].get('valid')) for r in results),
                'errors':sum(r['error'] is not None for r in results),'latest':result}
            save(out/'status.json',status)
            print(json.dumps({k:v for k,v in status.items() if k!='latest'},ensure_ascii=False,default=str),flush=True)
    save(out/'collection-completion.json',{'results':results,'elapsed_s':time.time()-started})


if __name__=='__main__':
    main()
