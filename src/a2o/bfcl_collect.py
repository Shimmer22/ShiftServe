"""记录真实调用轨迹和工具时间线；CPU/HTP 状态迁移不在远端模拟。"""
import argparse
import ast
import concurrent.futures
import copy
import hashlib
import importlib
import inspect
import json
import multiprocessing
import queue
import random
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path


def clock_ns():
    # CLOCK_MONOTONIC is shared across processes, including older macOS Python.
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def post(base, route, data, timeout=300):
    req = urllib.request.Request(base + route, json.dumps(data).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def tokenize(base, text):
    return post(base, '/tokenize', {'content': text, 'add_special': False,
                                  'parse_special': True})['tokens']


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def parse_calls(text, allowed):
    """只接受顶层函数和 literal 参数；拒绝属性、嵌套调用及任意 Python。"""
    clean = text.strip()
    if clean.startswith('```'):
        clean = re.sub(r'^```(?:python)?\s*|\s*```$', '', clean)
    try:
        node = ast.parse(clean, mode='eval').body
    except SyntaxError:
        if clean.startswith('['):
            raise ValueError('malformed function-call list')
        return []
    if not isinstance(node, (ast.List, ast.Tuple, ast.Call)):
        if clean.startswith('['):
            raise ValueError('not a literal function-call list')
        return []
    nodes = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    calls = []
    for call in nodes:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError('only direct sandbox function calls are allowed')
        name = call.func.id
        if name not in allowed or any(k.arg is None for k in call.keywords):
            raise ValueError('unknown function or expanded arguments: ' + name)
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        calls.append({'name': name, 'args': args, 'kwargs': kwargs,
                      'canonical': ast.unparse(call)})
    return calls


def sandbox_child(conn, entry):
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES
    methods = {}
    for clsname in entry['involved_classes']:
        if clsname not in {'GorillaFileSystem', 'MathAPI', 'MessageAPI', 'TwitterAPI',
                           'TicketAPI', 'TradingBot', 'TravelAPI', 'VehicleControlAPI'}:
            raise ValueError('not a permitted in-memory BFCL class: ' + clsname)
        obj = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[clsname]), clsname)()
        if clsname not in STATELESS_CLASSES:
            obj._load_scenario(copy.deepcopy(entry.get('initial_config', {}).get(clsname, {})),
                               long_context='long_context' in entry['id'])
        for name, method in inspect.getmembers(obj, inspect.ismethod):
            if not name.startswith('_'):
                methods[name] = (clsname, method)
    conn.send({'ready': True})
    while True:
        calls = conn.recv()
        if calls is None:
            return
        results = []
        for call in calls:
            started = clock_ns()
            error = None
            try:
                clsname, method = methods[call['name']]
                value = method(*call['args'], **call['kwargs'])
                if isinstance(value, str):
                    result = value
                elif isinstance(value, dict):
                    try:
                        result = json.dumps(value)
                    except TypeError:
                        result = str(value)
                else:
                    result = str(value)
            except Exception as exc:
                clsname = methods.get(call['name'], ('unknown',))[0]
                error = repr(exc)
                result = 'Error during execution: ' + str(exc)
            ended = clock_ns()
            results.append({'class': clsname, 'result': result, 'error': error,
                            'execution_start_ns': started, 'first_return_ns': ended,
                            'execution_complete_ns': ended, 'streaming': False})
        conn.send(results)


def load_entries(root, config):
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    entries = []
    for category in config['categories']:
        file = root / 'bfcl_eval/data' / ('BFCL_v4_' + category + '.json')
        for line in file.read_text().splitlines():
            entry = json.loads(line)
            docs = []
            for cls in entry['involved_classes']:
                docpath = root / 'bfcl_eval/data/multi_turn_func_doc' / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]
                docs.extend(json.loads(s) for s in docpath.read_text().splitlines() if s.strip())
            entry['function'] = docs
            entry['held_docs'] = {}
            for turn, names in entry.get('missed_function', {}).items():
                entry['held_docs'][turn] = [d for d in docs if d['name'] in names]
                docs = [d for d in docs if d['name'] not in names]
            entry['function'] = docs
            entries.append(entry)
    return entries


def prompt_messages(entry):
    from bfcl_eval.constants.default_prompts import _DEFAULT_SYSTEM_PROMPT
    return [{'role': 'system', 'content': _DEFAULT_SYSTEM_PROMPT.replace(
        '{functions}', json.dumps(entry['function'], ensure_ascii=False))}]


def serialized(messages, generation=True):
    return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'
                   for m in messages) + ('<|im_start|>assistant\n' if generation else '')


def run_case(entry, config, phase, slot, output):
    out = output / phase / entry['id']
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'entry.json', entry)
    base = 'http://127.0.0.1:' + str(config['port'])
    events = (out / 'events.jsonl').open('w', buffering=1)
    def emit(kind, **fields):
        row = {'kind': kind, 'task_id': entry['id'], 'phase': phase, 'slot': slot,
               'wall_time_ns': time.time_ns(), 'monotonic_ns': clock_ns(), **fields}
        events.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    sandbox = ctx.Process(target=sandbox_child, args=(child, entry))
    sandbox.start()
    messages = prompt_messages(entry)
    decoded = []
    previous_ids = []
    pending = []
    allowed = {d['name'] for d in entry['function']}
    complete = False
    error = None
    try:
        if not parent.poll(config['tool_timeout_s']) or not parent.recv().get('ready'):
            raise TimeoutError('sandbox initialization')
        for turn, user_messages in enumerate(entry['question']):
            if str(turn) in entry['held_docs']:
                newdocs = entry['held_docs'][str(turn)]
                allowed.update(d['name'] for d in newdocs)
                user_messages = [{'role': 'user', 'content': json.dumps(newdocs) +
                                 '\nI have updated some more functions you can choose from. What about now?'}]
            messages.extend(user_messages)
            steps = []
            decoded.append(steps)
            for step in range(config['max_steps']):
                key = f'turn-{turn:02d}-step-{step:02d}'
                prompt = serialized(messages)
                ids = tokenize(base, prompt)
                h = common_prefix(previous_ids, ids)
                p = len(ids) - h
                (out / (key + '.prompt.txt')).write_text(prompt)
                dump(out / (key + '.prompt.ids.json'), ids)
                if len(ids) + config['max_tokens'] > config['context_per_slot']:
                    raise ValueError(f'context_overflow: {len(ids)} + output budget; no truncation')
                preceding_tools = list(pending)
                for event in pending:
                    emit('tool_next_prompt', tool_event_id=event, next_step=key,
                         H_common_prefix=h, P_append=p, prompt_tokens=len(ids),
                         history_serialization_prefix_stable=h == len(previous_ids),
                         prompt_path=key + '.prompt.txt', image_count=0, image_sizes=[])
                pending = []
                started = clock_ns()
                request = {'prompt': ids, 'id_slot': slot, 'n_predict': config['max_tokens'],
                           'temperature': 0, 'seed': config['seed'], 'return_tokens': True,
                           'cache_prompt': bool(previous_ids), 'stream': False}
                dump(out / (key + '.request.json'), request)
                response = post(base, '/completion', request, config['request_timeout_s'])
                ended = clock_ns()
                dump(out / (key + '.response.json'), response)
                content = response.get('content', '')
                output_ids = response.get('tokens', [])
                if not output_ids:
                    raise ValueError('missing generated token ids or empty response')
                timings = response.get('timings', {})
                emit('inference', step=key, request_start_ns=started, response_ns=ended,
                     host_request_ms=(ended-started)/1e6, H_common_prefix=h, P_append=p,
                     prompt_tokens=len(ids), D_generated=len(output_ids),
                     generation_token_ids=output_ids, stop_type=response.get('stop_type'),
                     stop_word=response.get('stopping_word'), thinking_tokens=None,
                     thinking_supported=False, backend_timings=timings,
                     state_migration_ms=None, vision_init_ms=None,
                     cache_tokens_reported=response.get('tokens_cached'),
                     backend_prompt_cache_tokens=timings.get('cache_n'),
                     context_truncated=response.get('truncated', False))
                if response.get('truncated') or response.get('stop_type') == 'limit':
                    raise ValueError('truncated context or output token limit')
                calls = parse_calls(content, allowed)
                call_offset = content.find('[') if calls else -1
                call_position = len(tokenize(base, content[:call_offset])) if call_offset >= 0 else (0 if calls else None)
                eog_ids = set(tokenize(base, '<|im_end|>') + tokenize(base, '<|endoftext|>'))
                messages.append({'role': 'assistant', 'content': content})
                # Logical serialized history; cache counters are recorded separately.
                previous_ids = tokenize(base, serialized(messages, generation=False))
                steps.append([c['canonical'] for c in calls])
                emit('response_validation', step=key, call_count=len(calls),
                     calls=calls, eog_stop=response.get('stop_type') == 'eos',
                     tool_call_token_position=call_position,
                     eog_token_positions=[i for i,t in enumerate(output_ids) if t in eog_ids],
                     tool_call_position_semantics='tokenized text offset to first call/list; raw sampled IDs saved')
                for event in preceding_tools:
                    emit('tool_next_decode', tool_event_id=event, next_step=key,
                         D_generated=len(output_ids), stop_type=response.get('stop_type'),
                         next_tool_types=[c['name'] for c in calls], next_turn_complete=not calls)
                if not calls:
                    break
                issued = clock_ns()
                parent.send(calls)
                if not parent.poll(config['tool_timeout_s']):
                    raise TimeoutError('tool batch exceeded timeout')
                values = parent.recv()
                available = clock_ns()
                for i, (call, value) in enumerate(zip(calls, values)):
                    eid = key + f'-tool-{i:02d}'
                    pending.append(eid)
                    emit('tool', tool_event_id=eid, call=call,
                         tool_type='BFCL_in_memory_' + value['class'],
                         timing_kind='remote_sandbox_not_real_network_or_phone_tool',
                         issued_ns=issued, result_available_ns=available, **value,
                         gap_ms=(available-issued)/1e6,
                         execution_ms=(value['execution_complete_ns']-value['execution_start_ns'])/1e6,
                         argument_bytes=len(json.dumps({'args':call['args'], 'kwargs':call['kwargs']}).encode()),
                         result_bytes=len(value['result'].encode()),
                         result_tokens=len(tokenize(base, value['result'])),
                         history_tokens_before_tool=len(previous_ids), history_rounds=turn+1,
                         image_count=0, retries=0, timeout=False,
                         execution_success=_tool_ok(value),
                         tool_response_json_valid=_is_json(value['result']))
                messages.append({'role': 'tool', 'content': json.dumps([v['result'] for v in values])})
            else:
                raise ValueError('maximum steps exceeded')
        complete = True
    except Exception:
        error = traceback.format_exc()
        emit('failure', error=error, pending_tool_events=pending)
    finally:
        if sandbox.is_alive():
            try:
                parent.send(None)
            except (BrokenPipeError, OSError):
                pass
            sandbox.join(1)
            if sandbox.is_alive():
                sandbox.terminate()
                sandbox.join()
        parent.close()
        child.close()
    dump(out / 'decoded.json', decoded)
    dump(out / 'messages.json', messages)
    quality = {'valid': False, 'not_run': True}
    if complete:
        try:
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
            root = Path(config['bfcl_root'])
            category = entry['id'].rsplit('_', 1)[0]
            truthfile = root / 'bfcl_eval/data/possible_answer' / ('BFCL_v4_' + category + '.json')
            truth = next(json.loads(s)['ground_truth'] for s in truthfile.read_text().splitlines()
                         if json.loads(s)['id'] == entry['id'])
            quality = multi_turn_checker(decoded, truth, entry, category,
                                         phase + '_' + entry['id'])
        except Exception:
            quality = {'valid': False, 'checker_error': traceback.format_exc()}
    result = {'id': entry['id'], 'trace_complete': complete,
              'quality': quality, 'error': error}
    dump(out / 'result.json', result)
    emit('task_complete', **result)
    events.close()
    return result


def _is_json(text):
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def _tool_ok(value):
    if value['error'] is not None or value['result'].startswith('Error'):
        return False
    try:
        result = json.loads(value['result'])
        return not (isinstance(result, dict) and ('error' in result or result.get('success') is False))
    except ValueError:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['bfcl_root'])
    sys.path.insert(0, str(root))
    entries = load_entries(root, config)
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted((root/'bfcl_eval/data').glob('BFCL_v4_multi_turn*.json'))
    files += sorted((root/'bfcl_eval/data/multi_turn_func_doc').glob('*.json'))
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(args.output/'corpus-manifest.json', {'bfcl_revision':config['bfcl_revision'],
         'files_sha256':hashes, 'config':config,
         'prompt_format':'BFCL default Python-call prompting in Qwen ChatML',
         'quality_checker':'official multi_turn_checker; malformed calls never executed',
         'state_migration':'not applicable to remote characterization'})
    calibration = []
    for category in config['categories']:
        choices = [e for e in entries if e['id'].rsplit('_', 1)[0] == category]
        random.Random(config['seed']).shuffle(choices)
        calibration.extend(choices[:config['calibration_cases_per_category']])
    phases = [(f'calibration-c{n}', n, calibration) for n in config['calibration_concurrency']]
    phases.append(('collection-c8', config['collection_concurrency'], entries))
    summary = {}
    for phase, concurrency, tasks in phases:
        started = time.monotonic()
        tasks_queue = queue.Queue()
        for entry in tasks:
            tasks_queue.put(entry)
        def worker(slot):
            results = []
            while True:
                try:
                    entry = tasks_queue.get_nowait()
                except queue.Empty:
                    return results
                try:
                    results.append(run_case(copy.deepcopy(entry), config, phase, slot, args.output))
                except Exception:
                    fail = {'id':entry['id'], 'trace_complete':False, 'error':traceback.format_exc()}
                    folder = args.output / phase / entry['id']
                    folder.mkdir(parents=True, exist_ok=True)
                    dump(folder / 'worker-failure.json', fail)
                    results.append(fail)
                print(json.dumps({'phase':phase, 'finished':entry['id'],
                                  'complete':results[-1]['trace_complete']}), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker, slot) for slot in range(concurrency)]
            results = [r for f in futures for r in f.result()]
        summary[phase] = {'concurrency':concurrency, 'wall_seconds':time.monotonic()-started,
                          'tasks':len(results), 'complete':sum(r['trace_complete'] for r in results),
                          'quality_pass':sum(r.get('quality', {}).get('valid', False) for r in results)}
        dump(args.output / 'summary.json', summary)
        if phase.startswith('calibration') and not any(r['trace_complete'] for r in results):
            raise RuntimeError('all calibration traces failed; refusing full collection')
    dump(args.output / 'completion.json', {'status':'completed', 'summary':summary,
                                         'failed_traces_retained':True})


if __name__ == '__main__':
    main()
