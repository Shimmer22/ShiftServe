"""Build and run the same Android agent executable for all ablations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT.parent / 'shared/runtime/llama.cpp-hexagon-pkg'


def default_device():
    return os.environ.get('DEVICE') or json.loads((ROOT / 'configs/device.json').read_text())['serial']


def build():
    ndk = sorted((Path.home() / 'Library/Android/sdk/ndk').iterdir(),
                 key=lambda p: tuple(map(int, p.name.split('.'))))[-1]
    cxx = next((ndk / 'toolchains/llvm/prebuilt').glob('*/bin/aarch64-linux-android28-clang++'))
    binary = ROOT / 'build/agent_runtime'
    binary.parent.mkdir(exist_ok=True)
    subprocess.run([str(cxx), '-std=c++17', '-O2', '-fPIC', '-pie',
                    '-I' + str(PKG / 'include'), str(ROOT / 'src/a2o/agent_runtime.cpp'),
                    '-L' + str(PKG / 'lib'), '-lllama', '-lmtmd', '-lggml', '-lggml-base',
                    '-llog', '-ldl', '-lm', '-Wl,-rpath,/data/local/tmp/llama.cpp/lib',
                    '-o', str(binary)], check=True)
    return binary


def validate(config):
    if config['mode'] not in {'cpu', 'npu', 'hybrid'}:
        raise ValueError('mode must be cpu, npu, hybrid')
    if config['kv_policy'] not in {'none', 'recent'}:
        raise ValueError('kv_policy must be none, recent')
    if config['pre_move'] and config['mode'] != 'hybrid':
        raise ValueError('pre_move requires hybrid mode')
    if config.get('format', 'chat') not in {'raw', 'chat'}:
        raise ValueError('format must be raw, chat')
    if config.get('vision_backend', 'auto') not in {'auto', 'cpu', 'opencl', 'htp'}:
        raise ValueError('vision_backend must be auto, cpu, opencl, htp')
    if config.get('vision_prepare', 'arrival') not in {'arrival', 'a2o'}:
        raise ValueError('vision_prepare must be arrival, a2o')
    if not 1 <= config.get('batch_size', 512) <= 512:
        raise ValueError('batch_size must be within [1,512]')
    if config.get('vision_flash', 'auto') not in {'auto', 'off', 'on'}:
        raise ValueError('vision_flash must be auto, off, on')
    if config.get('htp_queue', 16) < 1:
        raise ValueError('htp_queue must be positive')


def check_response(result, scenario):
    if 'turns' in scenario:
        turns = result.get('turns', [])
        checks = [check_response(r, s) for r, s in zip(turns, scenario['turns'])]
        responses_ok = len(turns) == len(scenario['turns']) and all(c['passed'] for c in checks)
        state_ok = all(r['prefix_match'] and r['restore_checked'] and r['eog_committed'] for r in turns)
        image_ok = (result['image_encode_calls'] == sum(bool(t.get('image')) for t in scenario['turns'])
                    and all(r['image_encode_calls'] == int(bool(s.get('image'))) for r, s in zip(turns, scenario['turns'])))
        numeric_ok = all(not r['compared'] or (r['logits_cosine'] >= .999 and r['logits_nrmse'] <= .05
                                               and r['logits_top1_equal']) for r in turns)
        compared = sum(r['compared'] for r in turns)
        return {'passed': responses_ok and state_ok and image_ok and numeric_ok,
                'outputs_passed': responses_ok, 'state_passed': state_ok,
                'single_encode_passed': image_ok, 'numerical_passed': numeric_ok if compared else None,
                'numerical_compared_turns': compared, 'turns': checks}
    text = result['output'].strip()
    check = scenario['check']
    if check['type'] == 'exact':
        return {'passed': text == check['value'], 'expected': check['value']}
    if check['type'] == 'nonempty':
        return {'passed': bool(text)}
    if check['type'] == 'one_of':
        return {'passed': text in check['values'], 'expected_values': check['values'],
                'strict_format_passed': text == check['strict_value']}
    if check['type'] == 'json_action':
        fenced = text.startswith('```json\n') and text.endswith('\n```')
        payload = text[len('```json\n'):-len('\n```')] if fenced else text
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            return {'passed': False, 'reason': str(exc)}
        ok = (isinstance(obj, dict) and obj.get('action') == check['action']
              and all(key in obj for key in check.get('keys', []))
              and all(obj.get(key) == value for key, value in check.get('equals', {}).items()))
        result = {'passed': ok, 'json_action_passed': ok, 'parsed': obj,
                  'response_format': 'json_code_fence' if fenced else 'json',
                  'strict_format_passed': not fenced}
        if 'screen_contains' in check:
            screen = obj.get('screen') if isinstance(obj, dict) else None
            screen_ok = (isinstance(screen, str)
                         and any(name.casefold() in screen.casefold() for name in check['screen_contains']))
            result['screen_passed'] = screen_ok
            result['passed'] = ok and screen_ok
        return result
    raise ValueError('unknown response check')


def run(config_path, scenario_path, device, tag):
    config = json.loads(config_path.read_text())
    scenario = json.loads(scenario_path.read_text())
    validate(config)
    binary = build()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / f'{stamp}-{tag}'
    out.mkdir(parents=True)
    adb = ['adb', '-s', device]
    remote = '/data/local/tmp/a2o_agent_' + stamp

    def call(args, log=None):
        start = time.perf_counter()
        try:
            if log:
                with (out / log).open('w') as stream:
                    proc = subprocess.run(adb + args, text=True, stdout=stream,
                                          stderr=subprocess.STDOUT, timeout=config.get('timeout_s', 180))
                output = (out / log).read_text()
            else:
                proc = subprocess.run(adb + args, text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=30)
                output = proc.stdout
        except subprocess.TimeoutExpired:
            if log == 'runtime.log':
                stop = (f'p=$(cat {remote}/pid); '
                        f'if tr "\\0" " " < /proc/$p/cmdline | grep -F -q "{remote}/agent "; '
                        'then kill -TERM "$p"; fi')
                subprocess.run(adb + ['shell', stop], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=15)
            (out / 'failure.json').write_text(json.dumps({'reason': 'timeout', 'stage': log or args[0]}))
            raise RuntimeError('Timed out; owned phone runtime terminated; logs: ' + str(out))
        if proc.returncode:
            (out / 'failure.json').write_text(json.dumps({'reason': 'nonzero exit', 'stage': log or args[0],
                                                        'returncode': proc.returncode}))
            raise RuntimeError(output[-5000:])
        return (time.perf_counter() - start) * 1000

    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [ROOT / 'src/a2o/agent_runtime.cpp', *sorted((ROOT / 'src/a2o').glob('*.h'))]}
    manifest = {'config': config, 'scenario': scenario, 'device': device,
                'config_file': str(config_path.resolve().relative_to(ROOT)),
                'scenario_file': str(scenario_path.resolve().relative_to(ROOT)),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                'source_files_sha256': sources,
                'runtime_package': str(PKG), 'execution': 'phone', 'agent': 'observation replay; no device actions'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    call(['get-state'])
    call(['shell', 'mkdir', remote])
    try:
        call(['push', str(binary), remote + '/agent'])
        call(['shell', 'chmod', '755', remote + '/agent'])
        args = [remote + '/agent', '--model', '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf',
                '--mode', config['mode'], '--policy', config['kv_policy'],
                '--pre-move', str(int(config['pre_move'])), '--format', config.get('format', 'chat'),
                '--max-tokens', str(scenario.get('max_tokens', 64)),
                '--keep-history', str(config.get('keep_history', 96)),
                '--a2o-ms', str(config.get('a2o_ms', 0)), '--out', remote + '/result.json']
        args += ['--roundtrip', str(int(config.get('roundtrip', False)))]
        args += ['--vision-backend', config.get('vision_backend', 'auto')]
        args += ['--vision-prepare', config.get('vision_prepare', 'arrival'),
                 '--trace-vision', str(int(config.get('trace_vision', False)))]
        args += ['--vision-flash', config.get('vision_flash', 'auto')]
        args += ['--batch-size', str(config.get('batch_size', 512)),
                 '--cpu-repack', str(int(config.get('cpu_repack', True)))]
        args += ['--ctx-size', str(config.get('context_size', 2048))]
        for key in ['system', 'history', 'previous', 'prompt']:
            path = out / (key + '.txt')
            path.write_text(scenario.get(key, ''))
            call(['push', str(path), remote + '/' + path.name])
            args += ['--' + key, remote + '/' + path.name]
        if scenario.get('turns'):
            rows = []
            manifest['image_sha256'] = {}
            for i, turn in enumerate(scenario['turns']):
                prompt_path = out / f'turn-{i}.txt'
                prompt_path.write_text(turn['prompt'])
                call(['push', str(prompt_path), remote + '/' + prompt_path.name])
                image_remote = '-'
                if turn.get('image'):
                    image_path = ROOT / turn['image']
                    manifest['image_sha256'][str(i)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                    image_remote = remote + f'/turn-{i}.png'
                    call(['push', str(image_path), image_remote])
                rows.append('\t'.join([config['prefill_routes'][i], image_remote,
                                       remote + '/' + prompt_path.name, str(turn.get('max_tokens', 80)),
                                       config.get('decode_routes', ['cpu'] * len(scenario['turns']))[i]]))
            (out / 'turns.tsv').write_text('\n'.join(rows) + '\n')
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            call(['push', str(out / 'turns.tsv'), remote + '/turns.tsv'])
            args += ['--session', remote + '/turns.tsv',
                     '--compare-prefill', str(int(config.get('compare_prefill', True))),
                     '--mmproj', '/data/local/tmp/models/mmproj-model-f16.gguf']
        if scenario.get('image'):
            image = ROOT / scenario['image']
            call(['push', str(image), remote + '/image.png'])
            args += ['--image', remote + '/image.png', '--mmproj',
                     '/data/local/tmp/models/mmproj-model-f16.gguf']
            if config.get('vision_embeddings'):
                embeddings = (ROOT / config['vision_embeddings']).resolve()
                manifest['vision_embeddings_sha256'] = hashlib.sha256(embeddings.read_bytes()).hexdigest()
                (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
                call(['push', str(embeddings), remote + '/vision.f32'])
                args += ['--vision-embeddings', remote + '/vision.f32']
        shell = ('cd /data/local/tmp/llama.cpp; '
                 'unset MTMD_VISIONTRIM MTMD_VISIONTRIM_VIT_LAYER MTMD_VISIONTRIM_RETAIN_RATIO '
                 'MTMD_VISIONTRIM_DVTS_RATIO MTMD_VISIONTRIM_TGVC_ITER MTMD_VISIONTRIM_VIT_RETAIN_RATIO '
                 'MTMD_VISIONTRIM_VIT_BLOCK_MODE MTMD_VIT_TIMING MTMD_VIT_GRAPH_REUSE MTMD_BACKEND_DEVICE; '
                 'export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
                 'ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib GGML_HEXAGON_EXPERIMENTAL=1; '
                 + (f'export GGML_HEXAGON_OPQUEUE={int(config["htp_queue"])}; ' if 'htp_queue' in config else '')
                 + (f'export GGML_HEXAGON_OPBATCH={int(config["htp_opbatch"])}; ' if 'htp_opbatch' in config else '')
                 + f'echo $$ > {remote}/pid; exec ' + shlex.join(args))
        (out / 'command.txt').write_text(shell)
        wall = call(['shell', shell], 'runtime.log')
        call(['pull', remote + '/result.json', str(out / 'runtime.json')])
        if scenario.get('turns'):
            for i in range(len(scenario['turns'])):
                for kind in ['prompt', 'prefix']:
                    call(['pull', remote + f'/result.json.turn-{i}.{kind}.txt', str(out / f'turn-{i}.{kind}.txt')])
                if config.get('compare_prefill', True):
                    for backend in ['cpu', 'htp']:
                        call(['pull', remote + f'/result.json.turn-{i}.{backend}.logits.f32',
                              str(out / f'turn-{i}.{backend}.logits.f32')])
        else:
            call(['pull', remote + '/result.json.prompt.txt', str(out / 'formatted_prompt.txt')])
        result = json.loads((out / 'runtime.json').read_text())
        if scenario.get('image') or any(t.get('image') for t in scenario.get('turns', [])):
            expected = 'CLIP using ' + result['vision_backend'] + ' backend'
            if expected not in (out / 'runtime.log').read_text():
                raise RuntimeError('Vision backend did not match requested backend: ' + expected)
        result['response_check'] = check_response(result, scenario)
        result['process_wall_ms'] = wall
        if not result.get('session'):
            result['budget_adjusted_response_ms'] = (
                result['timing_ms']['observation_to_response_ms']
                + result['timing_ms']['prep_overrun_ms'])
        (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps({'run': str(out), **result}, ensure_ascii=False, indent=2), flush=True)
        return result
    except Exception:
        subprocess.run(adb + ['pull', remote, str(out / 'failure-device')],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    finally:
        call(['shell', 'rm', '-rf', remote])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--tag', default='agent')
    a = parser.parse_args()
    result = run(a.config, a.scenario, a.device, a.tag)
    if not result['response_check']['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
