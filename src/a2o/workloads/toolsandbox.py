"""wins 上运行官方 ToolSandbox；模型请求只发送到手机 HTTP 服务。"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import time
import traceback
import urllib.request


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def post(base, route, body, timeout=1800):
    req = urllib.request.Request(base + route, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'{route}: HTTP {exc.code}: {exc.read().decode()}') from exc


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def run_case(name, scenario, cfg, out):
    import attrs
    from openai.types.chat import ChatCompletion
    from tool_sandbox.common.execution_context import RoleType, get_current_context
    from tool_sandbox.roles.base_role import BaseRole
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
    import tool_sandbox.roles.execution_environment as environment
    from tool_sandbox.tools.user_tools import end_conversation

    folder = out / name
    folder.mkdir(parents=True, exist_ok=False)
    events, steps, tool_rows = [], [], []
    pending, executions = {}, {}
    state_ids = []
    task_started_ns, final_response_ns = None, None
    base = cfg['endpoint']
    if cfg.get('a2o_service'):
        save(folder / 'reset.json', post(base, '/a2o/reset', {}))

    def emit(kind, **row):
        event = dict(kind=kind, monotonic_ns=time.monotonic_ns(), **row)
        events.append(event)
        with (folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')

    def tokenize(text):
        return post(base, '/tokenize', dict(content=text, add_special=False, parse_special=True))['tokens']

    class PhoneAgent(OpenAIAPIAgent):
        def __init__(self):
            # 不初始化官方默认的云端客户端。
            pass

        def model_inference(self, openai_messages, openai_tools):
            nonlocal state_ids, task_started_ns, final_response_ns
            idx = len(steps)
            body = dict(model=cfg.get('model_id', 'qwen35-9b-q4_0'), messages=openai_messages,
                        tools=list(openai_tools), temperature=0, seed=cfg['seed'],
                        max_tokens=cfg['max_tokens'], stream=False, cache_prompt=True,
                        parallel_tool_calls=False, return_tokens=True,
                        chat_template_kwargs={'enable_thinking': cfg.get('enable_thinking', False)})
            save(folder / f'{idx:03d}.request.json', body)
            prep_start = time.monotonic_ns()
            if task_started_ns is None:
                task_started_ns = prep_start
            prompt = post(base, '/apply-template', body)['prompt']
            ids = tokenize(prompt)
            save(folder / f'{idx:03d}.prompt-ids.json', ids)
            (folder / f'{idx:03d}.prompt.txt').write_text(prompt)
            h = common_prefix(state_ids, ids)
            if len(ids) + cfg['max_tokens'] > cfg['context_size']:
                raise RuntimeError('context overflow; refusing truncation')
            preceding_tools = []
            for message in openai_messages:
                if message['role'] != 'tool' or message['tool_call_id'] not in pending:
                    continue
                call = pending.pop(message['tool_call_id'])
                row = dict(**call, next_step=idx, P_body=len(tokenize(message['content'])),
                           result_bytes=len(message['content'].encode()), result=message['content'],
                           H_common_prefix=h, P_append=len(ids)-h, next_prompt_tokens=len(ids))
                tool_rows.append(row)
                preceding_tools.append(row)
            request_ns = time.monotonic_ns()
            for row in preceding_tools:
                row.update(next_request_ns=request_ns,
                           issue_to_next_request_ms=(request_ns-row['issued_ns'])/1e6,
                           result_to_next_request_ms=(request_ns-row['result_available_ns'])/1e6)
                emit('tool_return', **row)
            response = post(base, '/v1/chat/completions', body, cfg['request_timeout_s'])
            response_ns = time.monotonic_ns()
            save(folder / f'{idx:03d}.response.json', response)
            if not response['choices'][0]['message'].get('tool_calls'):
                final_response_ns = response_ns
            native = response.get('__verbose')
            if not native or not native.get('tokens'):
                raise RuntimeError('missing raw server token ids; start server with --verbosity 10')
            if native.get('prompt') != prompt:
                raise RuntimeError('apply-template differs from actual inference prompt')
            if native.get('truncated') or response['choices'][0]['finish_reason'] == 'length':
                raise RuntimeError('context/output truncated; raw response retained')
            usage = response['usage']
            if usage['prompt_tokens'] != len(ids):
                raise RuntimeError(f'prompt token accounting mismatch: {usage["prompt_tokens"]} != {len(ids)}')
            state_ids = ids + native['tokens']
            row = dict(step=idx, H_common_prefix=h, P_append=len(ids)-h,
                       prompt_tokens=len(ids), D=usage['completion_tokens'],
                       D_raw_ids=len(native['tokens']), generated_ids=native['tokens'],
                       finish_reason=response['choices'][0]['finish_reason'],
                       server_stop_type=native['stop_type'],
                       host_request_ms=(response_ns-request_ns)/1e6,
                       measurement_preparation_ms=(request_ns-prep_start)/1e6,
                       backend_timings=response.get('timings'),
                       cache_tokens_reported=usage.get('prompt_tokens_details', {}).get('cached_tokens'))
            steps.append(row)
            for tool in preceding_tools:
                tool['D_next'] = row['D']
                tool['issue_to_response_ms'] = (response_ns-tool['issued_ns'])/1e6
                tool['result_to_response_ms'] = (response_ns-tool['result_available_ns'])/1e6
            emit('inference', **row)
            mappings = []
            for call_index, call in enumerate(response['choices'][0]['message'].get('tool_calls') or []):
                # llama.cpp 的 opaque ID 可用数字开头；官方环境将 ID 拼入 Python 变量名。
                # 仅适配关联 ID，原始服务响应已在上面原样落盘。
                original_id = call['id']
                call['id'] = f'call_{idx}_{call_index}'
                mappings.append(dict(raw_id=original_id, execution_id=call['id']))
                json.loads(call['function']['arguments'])
            save(folder / f'{idx:03d}.tool-id-map.json', mappings)
            return ChatCompletion.model_validate(response)

        def respond(self, ending_index=None):
            super().respond(ending_index)
            issued = time.monotonic_ns()
            for message in self.get_messages():
                if (message.sender == RoleType.AGENT and
                    message.recipient == RoleType.EXECUTION_ENVIRONMENT and
                    message.openai_tool_call_id not in executions and
                    message.openai_tool_call_id not in pending):
                    pending[message.openai_tool_call_id] = dict(
                        tool_id=message.openai_tool_call_id, tool=message.openai_function_name,
                        issued_ns=issued, H=len(state_ids), inference_step=len(steps)-1)
            if pending and cfg.get('a2o_service'):
                emit('preparation_requested', response=post(base, '/a2o/tool-start', {}))
                # 服务准备确认之后才将工具交给官方环境，G 不包含控制 RPC。
                issued = time.monotonic_ns()
                for record in pending.values():
                    record['issued_ns'] = issued

    class SingleTurnUser(BaseRole):
        """只用于预先选择的单用户轮场景；首个最终回答后结束，不注入正确答案。"""
        role_type = RoleType.USER

        def respond(self, ending_index=None):
            end_conversation()

    original_single = environment.respond_to_single_message

    def measured_single(interactive_console, message, role_type):
        t0 = time.monotonic_ns()
        result = original_single(interactive_console, message, role_type)
        t1 = time.monotonic_ns()
        if message.sender == RoleType.AGENT and result is not None:
            executions[message.openai_tool_call_id] = dict(
                execution_start_ns=t0, execution_end_ns=t1,
                execution_ms=(t1-t0)/1e6, tool_error=result.tool_call_exception)
        return result

    class MeasuredEnvironment(environment.ExecutionEnvironment):
        def respond(self, ending_index=None):
            super().respond(ending_index)
            available = time.monotonic_ns()
            for key, record in pending.items():
                if key in executions:
                    record.update(executions[key], result_available_ns=available,
                                  G_ms=(available-record['issued_ns'])/1e6)

    # 官方环境仍执行白名单工具并维护数据库，探针只包围单次执行和返回边界。
    environment.respond_to_single_message = measured_single
    roles = {RoleType.AGENT: PhoneAgent(), RoleType.USER: SingleTurnUser(),
             RoleType.EXECUTION_ENVIRONMENT: MeasuredEnvironment()}
    error, evaluation = None, None
    try:
        result = scenario.play_and_evaluate(roles=roles, output_directory=folder, scenario_name=name)
        evaluation = attrs.asdict(result.evaluation_result)
        if pending:
            raise RuntimeError('unconsumed tool result events')
        if not tool_rows or not all('G_ms' in row for row in tool_rows):
            raise RuntimeError('incomplete H/P/G/D trace')
    except Exception:
        error = traceback.format_exc()
        emit('failure', error=error)
    finally:
        environment.respond_to_single_message = original_single
    result = dict(name=name, trace_complete=error is None, error=error, evaluation=evaluation,
                  full_score=evaluation is not None and evaluation['similarity'] == 1.0,
                  inference_calls=len(steps), tool_calls=len(tool_rows))
    result['agent_e2e_ms'] = ((final_response_ns-task_started_ns)/1e6
                              if error is None and final_response_ns is not None else None)
    result['e2e_definition'] = ('first request measurement/template preparation to final model HTTP response; '
                                'includes real tools, network and host orchestration; excludes model init and evaluator')
    save(folder / 'metrics.json', dict(inference=steps, tools=tool_rows))
    save(folder / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['backend'] not in ('CPU', 'HTP', 'route') or cfg['concurrency'] != 1:
        raise ValueError('本入口要求显式 CPU/HTP/route、单任务串行配置')
    if cfg['backend'] != 'CPU' and not cfg.get('a2o_service'):
        raise ValueError('HTP/route 必须显式接入 a2o_service')
    collector_lock = Path(cfg['collector_lock']).open('a')
    fcntl.flock(collector_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    random.seed(cfg['seed'])
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios.single_tool_call_scenarios import named_single_tool_call_scenarios
    from tool_sandbox.scenarios.multiple_tool_call_scenarios import named_multiple_tool_call_scenarios
    scenarios = named_single_tool_call_scenarios(ToolBackend.DEFAULT)
    scenarios.update(named_multiple_tool_call_scenarios(ToolBackend.DEFAULT))
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output / 'collector-source.py').write_bytes(source)
    save(args.output / 'manifest.json', dict(config=cfg, engine='phone llama.cpp HTTP server',
        collector_sha256=hashlib.sha256(source).hexdigest(),
        user='deterministic close after first final answer; single-user-turn bootstrap',
        evaluator='unmodified official ToolSandbox milestone/minefield evaluator',
        metrics={'H':'previous phone prompt + sampled output token count at tool issue',
                 'P_body':'each tool result content token count using phone tokenizer',
                 'H_common_prefix':'LCP of previous prompt+raw generation and next prompt',
                 'P_append':'next prompt tokens minus H_common_prefix; includes template overhead',
                 'G_ms':'host tool issue to result available after official execution environment',
                 'D':'server usage.completion_tokens; raw sampled IDs and count also saved'},
        timing='host CLOCK_MONOTONIC; G excludes tokenizer/profiling preparation before next request',
        backend=cfg['backend'], kv_policy=cfg.get('kv_policy', 'none'), pre_move=False, vision='text-only; not exercised'))
    results = []
    for name in cfg['scenarios']:
        result = run_case(name, copy.deepcopy(scenarios[name]), cfg, args.output)
        results.append(result)
        save(args.output / 'summary.json', dict(results=results, completed=len(results),
            trace_complete=sum(r['trace_complete'] for r in results),
            full_score=sum(r['full_score'] for r in results)))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    save(args.output / 'completion.json', dict(completed=True, groups=len(results)))


if __name__ == '__main__':
    main()
