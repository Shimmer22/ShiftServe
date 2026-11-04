"""持久手机进程的 Tool 生命周期驱动；只执行显式白名单只读工具。"""
import hashlib
import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
from urllib.request import urlopen
from urllib.parse import urlparse
from datetime import datetime
from zoneinfo import ZoneInfo

from .runner import ROOT, PKG, build, check_response
from .router_table import export_table


def execute_tool(spec, device):
    kind = spec['type']
    if kind == 'device_display':
        # 仅查询显示尺寸；不设置分辨率，不执行模型生成的命令。
        proc = subprocess.run(['adb', '-s', device, 'shell', 'wm', 'size'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        if not proc.stdout.strip():
            raise RuntimeError('display query returned no observation')
        return proc.stdout
    if kind == 'http_get':
        url = spec['url']
        if urlparse(url).scheme not in {'http', 'https'}:
            raise ValueError('HTTP tool requires http/https URL')
        with urlopen(url, timeout=30) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError('HTTP tool result exceeds 64 KiB; refusing truncation')
            return raw
    raise ValueError('unknown read-only tool: ' + str(kind))


def validate_tool_config(config, scenario, mechanism):
    if (config.get('mode') != 'hybrid' or config.get('kv_policy') != 'none'
            or config.get('pre_move') is not False or config.get('cpu_repack') is not False
            or config.get('timing_valid_for_performance') is not False):
        raise ValueError('tool session requires explicit hybrid, no compression/premove/repack, diagnostic timing')
    for t in scenario['turns']:
        if t['tool']['type'] not in {'device_display', 'http_get'}:
            raise ValueError('tool is not on the read-only allowlist')
        if not mechanism and (t.get('preparation') == 'hold' or t.get('wait_ready') or t.get('router_fixture')):
            raise ValueError('barriers/fixture costs require mechanism test mode')


class Protocol:
    def __init__(self, proc, output):
        self.proc, self.output = proc, output
        self.messages = queue.Queue()
        self.events = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            with (self.output / 'protocol.log').open('w') as log:
                for line in self.proc.stdout:
                    log.write(line); log.flush()
                    if line.startswith('A2O\t'):
                        event = json.loads(line[4:])
                        self.events.append(event)
                        self.messages.put(event)
        except Exception as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(EOFError('native protocol ended'))

    def send(self, *fields):
        if any('\n' in str(f) or '\t' in str(f) for f in fields):
            raise ValueError('invalid protocol field')
        self.proc.stdin.write('\t'.join(map(str, fields)) + '\n')
        self.proc.stdin.flush()

    def wait(self, name, turn=None, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('native event timeout: ' + name)
            try:
                event = self.messages.get(timeout=left)
            except queue.Empty:
                raise TimeoutError('native event timeout: ' + name) from None
            if isinstance(event, Exception):
                raise event
            if event['event'] == name and (turn is None or event['turn'] == turn):
                return event


def validate_lifecycle(events, turns, mechanism):
    def index(name, turn):
        return next((i for i, e in enumerate(events) if e['event'] == name and e['turn'] == turn), -1)
    checks = []
    for i, turn in enumerate(turns):
        order = [index(n, i) for n in ('tool_started', 'observation_ready', 'route_selected',
                                      'prefill_started', 'action_complete', 'cpu_state_committed', 'response_published')]
        ok = min(order) >= 0 and order == sorted(order) and len(set(order)) == len(order)
        ok &= turn['committed_version'] == i + 1 and turn['authority_checked']
        checks.append({'turn': i, 'passed': bool(ok)})
    cross_round = None
    if mechanism:
        # 本套件指定 turn1 持有旧任务、turn2 继续 CPU，之后才释放。
        held = index('barrier_entered', 1)
        pub1, pub2 = index('response_published', 1), index('response_published', 2)
        released = index('barrier_released', 3)
        cancelled = index('prepare_work_cancelled', 1)
        cross_round = (0 <= held < pub1 < index('tool_started', 2) < pub2 < released
                       and cancelled > pub2 and turns[1]['background_occupied_at_publish']
                       and turns[2]['background_occupied_at_publish']
                       and turns[1]['prefill_backend'] == turns[2]['prefill_backend'] == 'cpu'
                       and turns[2]['availability'] == 'busy'
                       and turns[3]['route_reason'] == 'cpu_lower_cost'
                       and turns[4]['route_reason'] == 'htp_lower_cost')
    closed = next((e for e in events if e['event'] == 'session_closed'), None)
    shutdown = (closed is not None and closed['started_tasks'] == closed['reaped_tasks'])
    return {'passed': bool(turns) and all(c['passed'] for c in checks) and cross_round is not False and shutdown,
            'turns': checks, 'cross_round_nonblocking': cross_round, 'shutdown_reaped_all': shutdown}


def run_tool_session(config_path, scenario_path, device, *, control=False, mechanism=False):
    config = json.loads(Path(config_path).read_text())
    scenario = json.loads(Path(scenario_path).read_text())
    validate_tool_config(config, scenario, mechanism or control)
    binary = build()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / f'{stamp}-tool-session-{"control" if control else "mechanism" if mechanism else "auto"}'
    out.mkdir(parents=True)
    remote = '/data/local/tmp/a2o_tool_' + stamp
    adb = ['adb', '-s', device]
    def call(*args):
        return subprocess.run(adb + list(args), check=True, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [ROOT / 'src/a2o/agent_runtime.cpp', *sorted((ROOT / 'src/a2o').glob('*.h'))]}
    table = out / 'router.tsv'
    metadata = export_table(ROOT / config['calibration_run'], table, config['router_decode_evaluations'])
    for k in ('context_size', 'batch_size', 'htp_queue', 'htp_opbatch'):
        if config[k] != metadata['config'][k]:
            raise ValueError('runtime/calibration configuration mismatch: ' + k)
    manifest = {'config': config, 'scenario': scenario, 'device': device, 'control': control, 'mechanism_test': mechanism,
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'source_files_sha256': sources,
                'runtime_package': str(PKG), 'calibration': metadata, 'agent': 'read-only tools; no generated actions executed',
                'timing_valid_for_performance': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out / 'system.txt').write_text(scenario['system'])
    (out / 'empty.txt').write_text('')
    proc = None; protocol = None; turns = []; host_events = []
    runtime_log = (out / 'runtime.log').open('w')
    def host_event(name, turn, **fields):
        event = {'event': name, 'turn': turn, 'host_monotonic_ns': time.monotonic_ns(), **fields}
        host_events.append(event)
        with (out / 'host-events.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    try:
        call('get-state'); call('shell', 'mkdir', remote)
        for p, name in [(binary, 'agent'), (table, 'router.tsv'), (out / 'system.txt', 'system.txt'), (out / 'empty.txt', 'empty.txt')]:
            call('push', str(p), remote + '/' + name)
        call('shell', 'chmod', '755', remote + '/agent')
        args = [remote + '/agent', '--tool-session', '1', '--mode', 'hybrid', '--policy', 'none', '--pre-move', '0',
                '--model', '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf', '--system', remote + '/system.txt',
                '--history', remote + '/empty.txt', '--previous', remote + '/empty.txt', '--prompt', remote + '/empty.txt',
                '--vision-backend', 'cpu', '--cpu-repack', '0', '--ctx-size', str(config['context_size']),
                '--batch-size', str(config['batch_size']), '--htp-queue', str(config['htp_queue']),
                '--htp-opbatch', str(config['htp_opbatch']), '--router-table', remote + '/router.tsv',
                '--router-d', str(config['router_decode_evaluations']), '--max-tokens', str(config['max_tokens']),
                '--mechanism-test', str(int(mechanism)), '--out', remote + '/result.json']
        env = ('export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
               f'GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE={config["htp_queue"]} GGML_HEXAGON_OPBATCH={config["htp_opbatch"]}; ')
        # 此入口不加载视觉编码器；CPU/HTP 参数显式覆盖。
        shell = f'cd /data/local/tmp/llama.cpp; {env}echo $$ > {remote}/pid; exec ' + shlex.join(args)
        (out / 'command.txt').write_text(shell)
        proc = subprocess.Popen(adb + ['shell', '-T', shell], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=runtime_log, text=True, bufsize=1)
        protocol = Protocol(proc, out)
        protocol.wait('session_ready', timeout=config['timeout_s'])
        for i, spec in enumerate(scenario['turns']):
            mode = 'none' if control else spec.get('preparation', 'auto') if mechanism else 'auto'
            host_event('tool_dispatched', i, tool=spec['tool'])
            protocol.send('START', i, mode)
            protocol.wait('tool_started', i, config['timeout_s'])
            host_event('tool_issued', i, tool=spec['tool'])
            raw = execute_tool(spec['tool'], device)
            (out / f'tool-{i}.raw.txt').write_bytes(raw)
            host_event('tool_result_available', i, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            prompt = spec['prompt'] + '\nTool result (' + spec['tool']['type'] + '):\n' + raw.decode('utf-8')
            observation = out / f'turn-{i}.txt'; observation.write_text(prompt)
            if mechanism and spec.get('wait_ready'):
                protocol.send('WAIT', i); protocol.wait('candidate_waited', i, config['timeout_s'])
            call('push', str(observation), remote + f'/observation-{i}.txt')
            protocol.send('OBS', i, remote + f'/observation-{i}.txt')
            actual = protocol.wait('observation_ready', i, config['timeout_s'])
            run_fields = ['RUN', i]
            if mechanism and spec.get('router_fixture'):
                fixture = spec['router_fixture']
                if fixture not in {'cpu', 'htp'}:
                    raise ValueError('unknown router mechanism fixture')
                path = out / f'fixture-{i}.tsv'
                header = f'a2o-router-v1\tmechanism-fixture\t{config["context_size"]}\t{config["batch_size"]}\t{config["htp_queue"]}\t{config["htp_opbatch"]}\n'
                cc, hc = (100, 200) if fixture == 'cpu' else (200, 100)
                key = f'{actual["H"]}\t{actual["P"]}\t{actual["D"]}'
                path.write_text(header + f'{key}\tCC\t{cc}\t0\t0\n{key}\tHC\t{hc}\t999\t0\n')
                call('push', str(path), remote + '/fixture.tsv')
                run_fields.append(remote + '/fixture.tsv')
            protocol.send(*run_fields)
            result = dict(protocol.wait('response_published', i, config['timeout_s'])['result'])
            turns.append(result)
            host_event('response_received', i)
            result['response_check'] = check_response(result, spec)
            (out / f'turn-{i}.result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            if not result['response_check']['passed']:
                raise RuntimeError(f'turn {i} JSON/memory check failed')
            if mechanism and spec.get('release_after'):
                protocol.send('RELEASE'); protocol.wait('barrier_released', i + 1, config['timeout_s'])
                # 已证明两轮 CPU 推进后，结束独立测试夹具，再验证 ready 分支。
                protocol.send('DRAIN'); protocol.wait('test_cleanup_drained', i + 1, config['timeout_s'])
        protocol.send('CLOSE'); protocol.wait('session_closed', timeout=config['timeout_s'])
        proc.stdin.close()
        if proc.wait(timeout=config['timeout_s']):
            raise RuntimeError('native runtime exited unsuccessfully')
        protocol.reader.join(timeout=5)
        result = {'turns': turns, 'lifecycle_check': validate_lifecycle(protocol.events, turns, mechanism),
                  'timing_valid_for_performance': False, 'result_directory': str(out.relative_to(ROOT))}
        result['passed'] = all(t['response_check']['passed'] for t in turns) and result['lifecycle_check']['passed']
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as exc:
        runtime_log.flush()
        (out / 'failure.json').write_text(json.dumps({'error': str(exc), 'type': type(exc).__name__,
            'runtime_log_tail': (out / 'runtime.log').read_text(errors='replace')[-5000:]}, ensure_ascii=False))
        raise RuntimeError(f'{exc}; original logs: {out}') from exc
    finally:
        if proc is not None and proc.poll() is None:
            # 只终止本次临时目录所属进程，不能影响并行用户实验。
            stop = (f'p=$(cat {remote}/pid); if tr "\\0" " " < /proc/$p/cmdline | '
                    f'grep -F -q "{remote}/agent "; then kill -TERM "$p"; fi')
            try:
                call('shell', stop)
            finally:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=5)
        if protocol:
            protocol.reader.join(timeout=5)
            (out / 'native-events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in protocol.events))
        runtime_log.close()
        # 成功与失败都保留设备端原始文件；清理仅限本次目录。
        cleanup_errors = []
        for operation in [('pull', remote, str(out / 'device')), ('shell', 'rm', '-rf', remote)]:
            try:
                call(*operation)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            (out / 'cleanup-errors.json').write_text(json.dumps(cleanup_errors, ensure_ascii=False, indent=2))
