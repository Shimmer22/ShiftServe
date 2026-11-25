"""持久手机进程的 Tool 生命周期驱动；只执行显式白名单只读工具。"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ..environment import ROOT, PKG, build, native_source_hashes
from ..validation.contracts import check_response
from ..validation.tool_session import validate_tool_config, validate_lifecycle
from ..tools.adapters import execute_tool
from ..scheduling.router_table import export_table
from .protocol import Protocol
from .device_transport import DeviceTransport
from .phone_preflight import require_phone_preflight


def run_tool_session(config_path, scenario_path, device, *, control=False, mechanism=False):
    config = json.loads(Path(config_path).read_text())
    scenario = json.loads(Path(scenario_path).read_text())
    validate_tool_config(config, scenario, mechanism or control)
    private_runtime=None
    if config.get('htp_execution_prepare',False):
        private_runtime=(ROOT/config['execution_prepare_dir']).resolve()
        for name in ('libllama.so','libggml-base.so','libggml-hexagon.so'):
            if not (private_runtime/name).is_file():raise ValueError('private execution runtime missing: '+name)
    binary = build()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / f'{stamp}-tool-session-{"control" if control else "mechanism" if mechanism else "auto"}'
    out.mkdir(parents=True)
    remote = '/data/local/tmp/a2o_tool_' + stamp
    adb = ['adb', '-s', device]
    def call(*args):
        return subprocess.run(adb + list(args), check=True, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sources = native_source_hashes()
    table = out / 'router.tsv'
    metadata = export_table(ROOT / config['calibration_run'], table, config['router_decode_evaluations'])
    for k in ('context_size', 'batch_size', 'htp_queue', 'htp_opbatch'):
        if config[k] != metadata['config'][k]:
            raise ValueError('runtime/calibration configuration mismatch: ' + k)
    manifest = {'config': config, 'scenario': scenario, 'device': device, 'control': control, 'mechanism_test': mechanism,
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'source_files_sha256': sources,
                'runtime_package': str(PKG), 'calibration': metadata, 'agent': 'read-only tools; no generated actions executed',
                'timing_valid_for_performance': False}
    if private_runtime:
        manifest['execution_preparation']={'enabled':True,'development_unvalidated':True,
            'predicted_microbatch':config.get('predicted_input_tokens',64),'prefault_bytes':config.get('prefault_bytes',0),
            'libraries':{n:hashlib.sha256((private_runtime/n).read_bytes()).hexdigest() for n in ('libllama.so','libggml-base.so','libggml-hexagon.so')}}
    online_defaults={'alpha':0.2,'min_samples':3,'every':8,'capacity':256,'ttl_ms':300000}
    online_options={**online_defaults,**config.get('online_cost',{})}
    manifest['self_calibration']={'enabled':config.get('self_calibrating',False),
        'scope':'ordinary exact H/P/D; offline candidates; no execution-plan or mechanism-fixture updates',
        'options':online_options,'device_validated':False}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out / 'system.txt').write_text(scenario['system'])
    (out / 'empty.txt').write_text('')
    proc = None; protocol = None; turns = []
    runtime_log = (out / 'runtime.log').open('w')
    def host_event(name, turn, **fields):
        event = {'event': name, 'turn': turn, 'host_monotonic_ns': time.monotonic_ns(), **fields}
        with (out / 'host-events.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    try:
        call('get-state'); call('shell', 'mkdir', remote)
        for p, name in [(binary, 'agent'), (table, 'router.tsv'), (out / 'system.txt', 'system.txt'), (out / 'empty.txt', 'empty.txt')]:
            call('push', str(p), remote + '/' + name)
        call('shell', 'chmod', '755', remote + '/agent')
        if private_runtime:
            for name in ('libllama.so','libggml-base.so','libggml-hexagon.so'):
                destination=remote+'/'+name;call('push',str(private_runtime/name),destination)
                actual=call('shell','sha256sum',destination).stdout.decode().split()[0]
                if actual!=manifest['execution_preparation']['libraries'][name]:raise RuntimeError('private library hash mismatch')
        args = [remote + '/agent', '--tool-session', '1', '--mode', 'hybrid', '--policy', 'none', '--pre-move', '0',
                '--model', '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf', '--system', remote + '/system.txt',
                '--history', remote + '/empty.txt', '--previous', remote + '/empty.txt', '--prompt', remote + '/empty.txt',
                '--vision-backend', 'cpu', '--cpu-repack', '0', '--ctx-size', str(config['context_size']),
                '--batch-size', str(config['batch_size']), '--htp-queue', str(config['htp_queue']),
                '--htp-opbatch', str(config['htp_opbatch']), '--router-table', remote + '/router.tsv',
                '--router-d', str(config['router_decode_evaluations']), '--max-tokens', str(config['max_tokens']),
                '--mechanism-test', str(int(mechanism)), '--out', remote + '/result.json']
        args += ['--self-calibrating',str(int(config.get('self_calibrating',False)))]
        for name,value in online_options.items():
            if name not in online_defaults:raise ValueError('unknown native online cost option: '+name)
            args += ['--online-'+name.replace('_','-'),str(value)]
        if private_runtime:
            args += ['--htp-execution-prepare','1','--htp-predict-input',str(config.get('predicted_input_tokens',64)),
                     '--htp-prefault-bytes',str(config.get('prefault_bytes',0))]
        library_path=(remote+':' if private_runtime else '')+'/data/local/tmp/llama.cpp/lib'
        env = ('export LD_LIBRARY_PATH='+shlex.quote(library_path)+' ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
               f'GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE={config["htp_queue"]} GGML_HEXAGON_OPBATCH={config["htp_opbatch"]}; ')
        # 此入口不加载视觉编码器；CPU/HTP 参数显式覆盖。
        shell = f'cd /data/local/tmp/llama.cpp; {env}echo $$ > {remote}/pid; exec ' + shlex.join(args)
        (out / 'command.txt').write_text(shell)
        require_phone_preflight(DeviceTransport(ROOT, 'adb', device=device), out)
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
            stop = (f'p=$(cat {remote}/pid); if /system/bin/grep -a -F -q "{remote}/agent" /proc/"$p"/cmdline 2>/dev/null; '
                    f'then kill -TERM "$p"; fi')
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
