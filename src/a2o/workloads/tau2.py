"""使用官方领域、编排和评分；仅替换本地模型传输并记录精确 token。"""
import argparse
import concurrent.futures
import contextvars
import hashlib
import json
from pathlib import Path
import random
import threading
import time
import traceback
import urllib.error
import urllib.request

from a2o.profiling.bfcl_attention import save

CURRENT = contextvars.ContextVar('tau2_recording', default=None)


def install_recording(cfg):
    import litellm
    from tau2.utils import llm_utils
    from tau2.environment.environment import Environment
    original_tool = Environment.get_response

    def complete(**kwargs):
        ctx = CURRENT.get()
        assert ctx is not None, 'LLM call outside recorded task'
        kind = 'agent' if kwargs['model'] == 'a2o-agent' else 'user' if kwargs['model'] == 'a2o-user' else 'judge'
        index = ctx['calls']
        ctx['calls'] += 1
        body = {k: v for k, v in kwargs.items() if k in (
            'messages', 'tools', 'tool_choice', 'temperature', 'seed', 'response_format', 'max_tokens') and v is not None}
        if not body.get('tools'):
            body.pop('tools', None)
            body.pop('tool_choice', None)
        body.update(model=cfg['model'], return_token_ids=True,
                    chat_template_kwargs={'enable_thinking':False})
        body.setdefault('max_tokens', cfg['max_tokens_agent'])
        stem = ctx['folder'] / f'{index:03d}-{kind}'
        save(stem.with_suffix('.request.json'), body)
        request_ns = time.monotonic_ns()
        ctx['emit']('llm_request', index=index, role=kind, request_ns=request_ns)
        req = urllib.request.Request(cfg['endpoint']+'/v1/chat/completions',json.dumps(body).encode(),
                                     {'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=900) as response:
                raw = response.read()
            stem.with_suffix('.response.raw.json').write_bytes(raw)
            response = json.loads(raw)
        except urllib.error.HTTPError as exc:
            stem.with_suffix('.error.txt').write_bytes(exc.read())
            raise
        response_ns = time.monotonic_ns()
        save(stem.with_suffix('.response.json'),response)
        choice = response['choices'][0]
        ids = response.get('prompt_token_ids')
        generated = choice.get('token_ids')
        assert ids and generated, 'server did not return exact prompt/output IDs'
        assert len(ids)==response['usage']['prompt_tokens']
        assert len(generated)==response['usage']['completion_tokens']
        ctx['emit']('llm_response',index=index,role=kind,request_ns=request_ns,response_ns=response_ns,
                    prompt_tokens=len(ids),D=len(generated),finish_reason=choice['finish_reason'])
        if choice['finish_reason']=='length':
            raise RuntimeError('output limit; raw response retained')
        # 官方 generate 保持其 schema 转换、消息构造和 JSON 参数解析。
        return litellm.ModelResponse(**response)

    def get_response(environment, tool_call):
        ctx = CURRENT.get()
        if ctx is None or ctx.get('evaluating'):
            return original_tool(environment,tool_call)
        start = time.monotonic_ns()
        try:
            result = original_tool(environment,tool_call)
        except BaseException as exc:
            ctx['emit']('tool_error',call=tool_call.model_dump(),start_ns=start,end_ns=time.monotonic_ns(),error=repr(exc))
            raise
        ctx['emit']('tool_result',call=tool_call.model_dump(),start_ns=start,end_ns=time.monotonic_ns(),
                    result=result.model_dump())
        return result

    llm_utils.completion = complete
    llm_utils.get_response_cost = lambda response:0.0  # 本地推理无 API 费用，不修改官方 reward。
    Environment.get_response = get_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    import sys
    sys.path.insert(0,str(Path(cfg['benchmark_root'])/'src'))
    from tau2.run import get_tasks,run_task
    from tau2.orchestrator.orchestrator import Orchestrator
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr,level='ERROR')
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    install_recording(cfg)
    original_run = Orchestrator.run
    def recorded_run(orchestrator):
        result = original_run(orchestrator)
        ctx = CURRENT.get()
        save(ctx['folder']/'simulation-before-evaluation.json',result.model_dump(mode='json'))
        ctx['evaluating'] = True
        return result
    Orchestrator.run = recorded_run
    jobs = []
    manifest = dict(config=cfg,config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),tasks=[])
    for domain in cfg['domains']:
        tasks = sorted(get_tasks(domain),key=lambda task:task.id)
        random.Random(str(cfg['seed'])+domain).shuffle(tasks)
        selected = tasks[:1 if args.smoke else cfg['tasks_per_domain']]
        for task in selected:
            key = f'{domain}-{hashlib.sha256(task.id.encode()).hexdigest()[:12]}'
            jobs.append((domain,task,key))
            manifest['tasks'].append(dict(domain=domain,id=task.id,key=key))
    manifest['scope']='官方任务及评分；本地9B用户模拟器；非参考榜单配置；主机延迟不用于手机性能结论'
    manifest['source_sha256'] = {str(p.relative_to(Path(__file__).parents[3])):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__),Path(__file__).parents[3]/'scripts/collect/tau2_host.py']}
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())==manifest,'refusing mixed configuration'
    else:
        save(out/'manifest.json',manifest)
    mutex = threading.Lock()
    results = []
    def run(job):
        domain,task,key = job
        folder = out/key
        folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            return json.loads((folder/'result.json').read_text())
        attempts = sorted(folder.glob('attempt-*'))
        attempt = folder/f'attempt-{len(attempts):02d}'
        attempt.mkdir()
        save(attempt/'task.json',task.model_dump(mode='json'))
        def emit(kind,**kw):
            with (attempt/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind=kind,monotonic_ns=time.monotonic_ns(),**kw),ensure_ascii=False)+'\n')
        token = CURRENT.set(dict(folder=attempt,calls=0,emit=emit,evaluating=False))
        begin = time.monotonic()
        try:
            seed = cfg['seed']+int(hashlib.sha256(key.encode()).hexdigest()[:6],16)
            simulation = run_task(domain=domain,task=task,agent='llm_agent',user='user_simulator',
                llm_agent='a2o-agent',llm_user='a2o-user',
                llm_args_agent=dict(temperature=cfg['temperature'],seed=seed,max_tokens=cfg['max_tokens_agent']),
                llm_args_user=dict(temperature=cfg['temperature'],seed=seed,max_tokens=cfg['max_tokens_user']),
                max_steps=cfg['max_steps'],max_errors=cfg['max_errors'],seed=seed)
            save(attempt/'simulation.json',simulation.model_dump(mode='json'))
            result = dict(domain=domain,id=task.id,key=key,attempt=attempt.name,complete=True,
                          reward=simulation.reward_info.reward,termination_reason=simulation.termination_reason,
                          llm_calls=CURRENT.get()['calls'],seconds=time.monotonic()-begin)
        except BaseException:
            (attempt/'failure.txt').write_text(traceback.format_exc())
            result = dict(domain=domain,id=task.id,key=key,attempt=attempt.name,complete=False,
                          error=traceback.format_exc(),llm_calls=CURRENT.get()['calls'],seconds=time.monotonic()-begin)
        finally:
            CURRENT.reset(token)
        save(attempt/'result.json',result)
        save(folder/'result.json',result)
        with mutex:
            results.append(result)
            save(out/'status.json',dict(phase='collecting',finished=len(results),total=len(jobs),
                                      updated_unix=time.time(),results=results))
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return result
    save(out/'status.json',dict(phase='collecting',finished=0,total=len(jobs),updated_unix=time.time()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
        all_results=list(pool.map(run,jobs))
    save(out/'summary.json',dict(attempted=len(jobs),complete=sum(r['complete'] for r in all_results),
                               results=all_results,scope=manifest['scope']))
    save(out/'status.json',dict(phase='complete',finished=len(jobs),total=len(jobs),updated_unix=time.time()))


if __name__=='__main__':
    main()
