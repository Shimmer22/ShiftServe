"""独立、显式授权的单次 GUI 闭环；回放入口仍不执行动作。"""
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from ..environment import ROOT, build, default_device, native_source_hashes
from .protocol import Protocol


def validate_tap(action, bounds, width, height, memo):
    if set(action) != {'action', 'x', 'y', 'memo'} or action['action'] != 'tap' or action['memo'] != memo:
        raise ValueError('unexpected model action or task code')
    if any(type(action[k]) is not int or not 0 <= action[k] < limit for k, limit in (('x', width), ('y', height))):
        raise ValueError('tap requires integer pixel coordinates within the model image')
    x, y = action['x'], action['y']
    left, top, right, bottom = bounds
    if not left < x < right or not top < y < bottom:
        raise ValueError('model tap is outside the test button')
    return x, y


def run_gui_loop(config_path, device=None):
    config = json.loads(Path(config_path).read_text())
    device = device or default_device()
    if (config['mode'], config['prefill_backend'], config['decode_backend'], config['vision_backend']) != ('hybrid', 'cpu', 'cpu', 'opencl'):
        raise ValueError('single loop baseline explicitly requires CPU LLM and OpenCL vision')
    if config['kv_policy'] != 'none' or config['pre_move'] or config['cpu_repack'] or config['compare_prefill']:
        raise ValueError('single loop baseline disables preparation, repack and paired execution')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / (stamp + '-single-gui-loop'); out.mkdir(parents=True)
    remote = '/data/local/tmp/a2o_loop_' + stamp
    adb = ['adb', '-s', device]
    report = {'passed': False, 'device': device, 'config': config, 'turns': [],
              'timing_valid_for_performance': False, 'result_directory': str(out.relative_to(ROOT)),
              'scope': 'one actual browser button tap; in-memory test page only; no settings/user-data changes'}
    proc = sampler = protocol = server = None
    reverse_installed = False
    task_code = 'PINE-' + str(secrets.randbelow(90) + 10)
    token = secrets.token_hex(12); prefix = '/' + token + '/'
    state = {'loaded': threading.Event(), 'clicked': threading.Event(), 'rendered': threading.Event()}
    event_lock = threading.Lock()

    def save():
        (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    def event(name, **fields):
        with event_lock, (out / 'host-events.jsonl').open('a') as f:
            f.write(json.dumps({'event': name, 'host_monotonic_ns': time.monotonic_ns(), **fields}) + '\n')

    def call(*args, binary=False, check=True):
        result = subprocess.run(adb + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        with (out / 'adb.log').open('a') as log:
            log.write(json.dumps({'args': args, 'returncode': result.returncode,
                'stderr': result.stderr.decode(errors='replace')}) + '\n')
        if check: result.check_returncode()
        return result.stdout if binary else result.stdout.decode()

    page = (ROOT / 'scenarios/single-loop-page.html').read_text().replace('{{CODE}}', task_code).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body=b''):
            self.send_response(status); self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self.reply(200, page) if self.path == prefix else self.reply(404)

        def do_POST(self):
            kind = self.path[len(prefix):] if self.path.startswith(prefix) else ''
            if kind == 'loaded':
                state['loaded'].set()
            elif kind == 'complete' and state['loaded'].is_set() and not state['clicked'].is_set():
                state['clicked'].set()
            elif kind == 'rendered' and state['clicked'].is_set():
                state['rendered'].set()
            else:
                self.reply(409); return
            event('page_' + kind); self.reply(200)

    runtime_log = (out / 'runtime.log').open('w')
    sampler_log = (out / 'device-resources.log').open('w')
    save()
    try:
        call('get-state')
        report['identity'] = call('shell', 'getprop ro.product.model; getprop ro.soc.model; getprop ro.serialno')
        power = call('shell', 'dumpsys power'); (out / 'power-before.txt').write_text(power)
        if 'mWakefulness=Awake' not in power:
            raise RuntimeError('device must already be Awake; no automatic wake/settings changes')
        processes = call('shell', 'ps -A -o PID,NAME,ARGS'); (out / 'processes-before.txt').write_text(processes)
        if any('llama-server' in line or '/agent ' in line or 'agent_runtime' in line for line in processes.splitlines()):
            raise RuntimeError('another model experiment is running')
        binary = build()
        report.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      source_files_sha256=native_source_hashes(), task_code=task_code,
                      page_sha256=hashlib.sha256(page).hexdigest())
        port = config['port']
        if any(line.split()[-2] == 'tcp:' + str(port) for line in call('reverse', '--list').splitlines() if len(line.split()) >= 3):
            raise RuntimeError('test reverse port is already owned')
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        call('reverse', 'tcp:' + str(port), 'tcp:' + str(port)); reverse_installed = True
        call('shell', 'mkdir', remote)
        call('push', str(binary), remote + '/agent'); call('shell', 'chmod', '755', remote + '/agent')
        system = ('You are a GUI agent. Follow the task using the screenshot. Return exactly one JSON object, no markdown. '
                  'Tap coordinates x and y are integer PIXEL coordinates in the provided image. '
                  'Remember the task code shown before clicking. Never click browser chrome.')
        (out / 'system.txt').write_text(system); call('push', str(out / 'system.txt'), remote + '/system.txt')
        grammar = ROOT / config['response_grammar']
        report['response_grammar_sha256'] = hashlib.sha256(grammar.read_bytes()).hexdigest()
        call('push', str(grammar), remote + '/response.gbnf')
        args = [remote + '/agent', '--session', 'live', '--live-session', '1', '--mode', 'hybrid',
                '--model', '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf',
                '--mmproj', '/data/local/tmp/models/mmproj-model-f16.gguf', '--system', remote + '/system.txt',
                '--history', remote + '/system.txt', '--previous', remote + '/system.txt', '--prompt', remote + '/system.txt',
                '--policy', 'none', '--pre-move', '0', '--cpu-repack', '0', '--compare-prefill', '0',
                '--vision-backend', 'opencl', '--vision-flash', 'off',
                '--ctx-size', str(config['context_size']), '--batch-size', str(config['batch_size']),
                '--response-grammar', remote + '/response.gbnf',
                '--timing-valid-for-performance', '0', '--out', remote + '/result.json']
        shell = ('cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
                 'ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib GGML_HEXAGON_EXPERIMENTAL=1 '
                 f'GGML_HEXAGON_OPQUEUE={config["htp_queue"]} GGML_HEXAGON_OPBATCH={config["htp_opbatch"]} '
                 'MTMD_BACKEND_DEVICE=GPUOpenCL; '
                 f'echo $$ > {remote}/pid; exec ' + shlex.join(args))
        (out / 'command.txt').write_text(shell)
        event('model_process_start')
        proc = subprocess.Popen(adb + ['shell', '-T', shell], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=runtime_log, text=True, bufsize=1)
        protocol = Protocol(proc, out)
        protocol.wait('session_ready', timeout=config['timeout_s'])
        pid = call('shell', 'cat', remote + '/pid').strip()
        sampler = subprocess.Popen(adb + ['shell', f'while kill -0 {int(pid)} 2>/dev/null; do '
            'echo SAMPLE; cat /proc/uptime /proc/stat; '
            f'cat /proc/{int(pid)}/status; sleep 0.5; done'], stdout=sampler_log, stderr=subprocess.STDOUT)
        url = f'http://127.0.0.1:{port}{prefix}'
        event('page_open_issued')
        call('shell', 'am', 'start', '-a', 'android.intent.action.VIEW', '-d', url, '-p', config['browser_package'])
        if not state['loaded'].wait(30):
            raise RuntimeError('test page did not load; no action executed')
        from PIL import Image
        for turn in range(2):
            foreground = call('shell', 'dumpsys window')
            (out / f'foreground-{turn}.txt').write_text(foreground)
            focused = [line for line in foreground.splitlines() if 'mCurrentFocus=' in line]
            if not any(config['browser_package'] + '/' in line for line in focused):
                raise RuntimeError('test browser is not focused')
            raw_image = out / f'screenshot-{turn}.png'
            raw_image.write_bytes(call('exec-out', 'screencap', '-p', binary=True))
            image = out / f'observation-{turn}.png'
            with Image.open(raw_image) as im:
                width, height = im.size
                im.thumbnail((config['image_max_edge'], config['image_max_edge']), Image.Resampling.LANCZOS)
                model_width, model_height = im.size
                im.save(image)
            event('observation_captured', turn=turn,
                  screenshot_sha256=hashlib.sha256(raw_image.read_bytes()).hexdigest(),
                  model_input_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                  screenshot_size=[width, height])
            if turn == 0:
                prompt = ('Complete this task by tapping the green COMPLETE TASK button. '
                          'Read and remember the task code shown on the page. '
                          f'The image is {model_width} pixels wide and {model_height} pixels high. '
                          'Return action tap, x and y in image pixels, and memo containing the actual PINE code read from the page.')
            else:
                prompt = ('The previous action has been executed. Inspect the NEW screenshot. '
                          'If the page now shows DONE, return action done and memo containing the actual PINE code from the first screenshot. '
                          'Use the code from our previous conversation; it is no longer visible. Otherwise return {"action":"failed"}.')
            text = out / f'prompt-{turn}.txt'; text.write_text(prompt)
            call('push', str(image), remote + f'/image-{turn}.png')
            call('push', str(text), remote + f'/prompt-{turn}.txt')
            event('inference_issued', turn=turn)
            protocol.send('cpu', remote + f'/image-{turn}.png', remote + f'/prompt-{turn}.txt', config['max_tokens'], 'cpu')
            response = protocol.wait('response_published', turn, config['timeout_s'])
            result = response['result']; report['turns'].append(result); save()
            event('response_received', turn=turn)
            if not result['eog_committed'] or not result['prefix_match'] or not result['restore_checked'] or result['image_encode_calls'] != 1:
                raise RuntimeError('response/state/image lifecycle check failed')
            action = json.loads(result['output'])
            (out / f'action-{turn}.json').write_text(json.dumps(action, indent=2) + '\n')
            if turn == 0:
                call('shell', 'uiautomator', 'dump', remote + '/window.xml')
                xml = call('shell', 'cat', remote + '/window.xml'); (out / 'window-before.xml').write_text(xml)
                nodes = [node for node in ET.fromstring(xml).iter('node')
                         if node.get('text') == 'COMPLETE TASK' or node.get('content-desc') == 'COMPLETE TASK']
                if len(nodes) != 1:
                    raise RuntimeError('cannot identify unique test button bounds; no tap executed')
                bounds = tuple(map(int, re.findall(r'\d+', nodes[0].get('bounds', ''))))
                if len(bounds) != 4: raise RuntimeError('invalid button bounds')
                model_bounds = (bounds[0]*model_width/width, bounds[1]*model_height/height,
                                bounds[2]*model_width/width, bounds[3]*model_height/height)
                mx, my = validate_tap(action, model_bounds, model_width, model_height, task_code)
                x, y = round(mx*width/model_width), round(my*height/model_height)
                report['executed_action'] = {'x': x, 'y': y, 'button_bounds': bounds}
                event('action_issued', x=x, y=y)
                call('shell', 'input', 'tap', str(x), str(y))
                event('action_command_completed')
                if not state['rendered'].wait(15): raise RuntimeError('real click/page update not observed')
                event('environment_change_confirmed')
            else:
                if action != {'action': 'done', 'memo': task_code}:
                    raise RuntimeError('new screenshot completion/memory check failed')
                if result['cached_tokens'] != report['turns'][0]['evaluated_tokens'] or result['cached_state_bytes'] != report['turns'][0]['state_bytes']:
                    raise RuntimeError('history was not carried into the second turn')
        protocol.send('CLOSE'); protocol.wait('session_closed', timeout=30)
        proc.stdin.close()
        if proc.wait(timeout=30): raise RuntimeError('model process exited unsuccessfully')
        report.update(passed=True, actual_click_received=state['clicked'].is_set(),
                      actual_page_rendered=state['rendered'].is_set(), history_memory_passed=True,
                      single_process=True, image_encode_calls=sum(r['image_encode_calls'] for r in report['turns']))
    except Exception as exc:
        report['failure'] = str(exc); event('failure', reason=str(exc)); raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                call('shell', f'p=$(cat {remote}/pid); if tr "\\0" " " < /proc/$p/cmdline | grep -F -q "{remote}/agent "; then kill -TERM "$p"; fi')
                proc.wait(timeout=15)
            except Exception as exc:
                report['termination_error'] = str(exc)
        if sampler is not None:
            try: sampler.wait(timeout=3)
            except subprocess.TimeoutExpired: sampler.terminate(); sampler.wait(timeout=5)
        if protocol:
            protocol.reader.join(timeout=5)
            (out / 'native-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in protocol.events))
        runtime_log.close(); sampler_log.close()
        for args in [('pull', remote, str(out / 'device')), ('shell', 'rm', '-rf', remote)]:
            try: call(*args)
            except Exception as exc: report.setdefault('cleanup_errors', []).append(str(exc))
        if reverse_installed:
            try: call('reverse', '--remove', 'tcp:' + str(config['port']))
            except Exception as exc: report.setdefault('cleanup_errors', []).append(str(exc))
        if server: server.shutdown(); server.server_close()
        if report.get('cleanup_errors') or report.get('termination_error'): report['passed'] = False
        save(); print(str(out), flush=True)
    return report
