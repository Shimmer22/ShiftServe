#!/usr/bin/env python3
"""串行手机实验：真实 HTTP 等待与状态准备并发，完整输出成本校准。"""
import argparse
from datetime import datetime
import hashlib
import http.server
import json
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import build, native_source_hashes, PKG
from a2o.runtime.protocol import Protocol
from a2o.validation.contracts import check_response
from a2o.validation.oracle import active_model_processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/oracle-grid.json')
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--calibration-only', action='store_true', help='仅作独立诊断，不与正式时延矩阵混合')
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    if a.calibration_only:
        config['calibration_only'] = True
    if config.get('htp_profile', 0) not in (0, 1, 2) or config.get('htp_use_hmx') not in (None, 0, 1):
        raise ValueError('HTP profile/HMX 必须显式取支持的值')
    if a.repetitions is not None:
        config['repetitions'] = a.repetitions
    if config['repetitions'] < 3:
        raise ValueError('成本校准及测量至少三次，不选择成功子集')
    if (config['mode'], config['kv_policy'], config['pre_move'], config['cpu_repack']) != ('hybrid', 'none', False, False):
        raise ValueError('要求显式 hybrid、无压缩/通用预搬移/CPU repack')
    if config['history_tokens'] < 128 or config['history_tokens'] + max(config['new_tokens']) + 64 > config['context_size']:
        raise ValueError('历史长度或 context 容量不满足完整响应条件')
    if not config['new_tokens'] or len(set(config['new_tokens'])) != len(config['new_tokens']) or any(not 16 <= p <= 128 for p in config['new_tokens']):
        raise ValueError('P 必须是 16..128 的不重复整数')
    if 0 not in config['gap_ms'] or len(set(config['gap_ms'])) != len(config['gap_ms']) or any(g < 0 for g in config['gap_ms']):
        raise ValueError('G 必须包含 0（用于校准），不能重复或为负')
    if config['strategies'] != ['after_return', 'always', 'cost_aware'] or config['routes'] != ['CC', 'HC']:
        raise ValueError('要求完整三策略、CC/HC 及 Router 重放')
    limits = config.setdefault('numeric_acceptance', {'min_cosine': 0.999, 'max_nrmse': 0.05})
    if not 0 <= limits['min_cosine'] <= 1 or (limits['max_nrmse'] is not None and limits['max_nrmse'] < 0):
        raise ValueError('数值阈值不合法')
    probe_skel = None
    if config.get('htp_probe_skel'):
        if not config.get('calibration_only') or config.get('htp_profile') != 2:
            raise ValueError('私有route探针仅允许独立profile=2诊断，PMU字段不再代表硬件计数')
        probe_skel = Path(config['htp_probe_skel']).resolve()
        if not probe_skel.is_file(): raise ValueError('探针库不存在')
    prepack_dir = Path(config['htp_kv_prepack_dir']).resolve() if config.get('htp_kv_prepack_dir') else None
    if prepack_dir:
        if probe_skel or config.get('htp_profile',0)==2: raise ValueError('KV 原型与路由/PMU 探针分开')
        if config.get('htp_profile',0)==1 and not config.get('calibration_only'):
            raise ValueError('KV basic profile 仅允许独立校准诊断，不混入正式计时')
        for n in ('libggml-hexagon.so','libggml-htp-v79.so'):
            if not (prepack_dir/n).is_file(): raise ValueError('私有 KV 原型库缺失')
    binary = build()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / (stamp + '-http-oracle-grid'); out.mkdir()
    print(str(out), flush=True)
    adb = ['adb', '-s', a.device]
    remote = '/data/local/tmp/a2o_oracle_' + stamp
    def call(*args):
        return subprocess.run(adb + list(args), check=True, capture_output=True, timeout=60)
    bodies = {}
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        def do_GET(self):
            params = parse_qs(urlparse(self.path).query)
            p, gap, trial = (int(params[k][0]) for k in ('p', 'gap', 'id'))
            if p not in bodies or gap not in config['gap_ms']:
                self.send_error(400); return
            received = time.monotonic_ns()
            time.sleep(gap / 1000)
            body = bodies[p]
            self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers()
            self.wfile.write(body); self.wfile.flush()
            with (out / 'http-server.jsonl').open('a') as f:
                f.write(json.dumps({'id': trial, 'P': p, 'gap_requested_ms': gap, 'received_ns': received,
                    'sent_ns': time.monotonic_ns(), 'body_sha256': hashlib.sha256(body).hexdigest()}) + '\n')
        def log_message(self, *args):
            pass
    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    port = server.server_port
    manifest = {'config': config, 'device': a.device, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source_files_sha256': native_source_hashes(), 'runtime_package': str(PKG),
        'python_files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / 'scripts/analyze/oracle.py')},
        'scope': '真实 HTTP Tool + 原生并发准备；文本固定 H/P、完整自由生成到 EOG；无图像、无能耗结论',
        'clock': '设备 steady_clock 计量 HTTP/准备/完整响应；服务器 monotonic_ns 独立，不跨时钟相减',
        'preparation': '从共同已存在 CPU sequence snapshot 导入 HTP，包含 clear/import/字节核验；不含共同快照首次导出',
        'decode': 'CPU 完整响应，含 EOG 前向评估；所有 P/策略/路径生成 ID 必须一致',
        'router': '独立完整 D 校准；发起时仅按所有 P 是否有潜在收益决定准备，返回后按实际 P 及 ready 决定 CC/HC；不等 pending',
        'oracle': '强制路径重放是实测反事实，不是同时可观测的单轮真值；严格选路只允许返回时 ready 的 HC；另报允许等待/返回后准备的扩展上界',
        'backend_diagnostic': {'profile': config.get('htp_profile', 0), 'use_hmx': config.get('htp_use_hmx', 'default'), 'calibration_only': config.get('calibration_only', False)},
        'vision_backend': 'none', 'decode_backend': 'cpu', 'kv_type': 'F16', 'flash_attention': False,
        'http_transport': 'Android loopback via adb reverse to host controlled HTTP server',
        'model': '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf'}
    if probe_skel:
        manifest['route_probe'] = {'library': str(probe_skel), 'sha256': hashlib.sha256(probe_skel.read_bytes()).hexdigest(),
            'transport': 'a2o-route-v1', 'pmu_values_are_hardware_counters': False,
            'shared_install_replaced': False, 'remote_library': remote + '/probe-lib/libggml-htp-v79.so'}
    if prepack_dir:
        manifest['kv_prepack'] = {'enabled':True,'libraries':{n: {'path':str(prepack_dir/n),'sha256':hashlib.sha256((prepack_dir/n).read_bytes()).hexdigest()} for n in ('libggml-hexagon.so','libggml-htp-v79.so')},'scope':'history K/V tiles; Qwen F16 d128 2KV-heads; profiler reserved RPC mode; profile=0','calibration':'first HC uncached; subsequent HC must have identical prefill logits; preparation validates every tile element'}
        manifest['preparation'] += '; plus historical K/V native tiles when explicitly enabled'
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log = (out / 'runtime.log').open('w'); proc = None; protocol = None; rows = []
    try:
        call('get-state')
        processes = call('shell', 'ps -A -o PID,ARGS').stdout.decode()
        (out / 'device-processes-before.txt').write_text(processes)
        if active_model_processes(processes):
            raise RuntimeError('设备已有模型实验进程，拒绝争抢资源')
        call('shell', 'mkdir', remote); call('push', str(binary), remote + '/agent'); call('shell', 'chmod', '755', remote + '/agent')
        if prepack_dir:
            call('shell','mkdir',remote+'/probe-lib')
            for n in ('libggml-hexagon.so','libggml-htp-v79.so'):
                dest=remote+('/'+n if n=='libggml-hexagon.so' else '/probe-lib/'+n)
                call('push',str(prepack_dir/n),dest)
                if call('shell','sha256sum',dest).stdout.decode().split()[0] != manifest['kv_prepack']['libraries'][n]['sha256']: raise RuntimeError('private KV library hash mismatch')
        if probe_skel:
            call('shell', 'mkdir' , remote + '/probe-lib')
            call('push', str(probe_skel), remote + '/probe-lib/libggml-htp-v79.so')
            actual = call('shell', 'sha256sum', remote + '/probe-lib/libggml-htp-v79.so').stdout.decode().split()[0]
            if actual != manifest['route_probe']['sha256']: raise RuntimeError('设备探针库hash不符')
        (out / 'empty.txt').write_text(''); call('push', str(out / 'empty.txt'), remote + '/empty.txt')
        call('reverse', f'tcp:{port}', f'tcp:{port}')
        args = [remote + '/agent', '--oracle-profile', '1', '--model', manifest['model'], '--mode', 'hybrid', '--policy', 'none', '--pre-move', '0',
            '--vision-backend', 'cpu', '--cpu-repack', '0', '--ctx-size', str(config['context_size']), '--batch-size', str(config['batch_size']),
            '--oracle-fixture', config.get('input_fixture', 'legacy_truncated_text'),
            '--profile-history', str(config['history_tokens']), '--profile-inputs', ','.join(map(str, config['new_tokens'])),
            '--oracle-min-cosine', str(limits['min_cosine']), '--oracle-max-nrmse', 'none' if limits['max_nrmse'] is None else str(limits['max_nrmse']),
            '--http-port', str(port), '--out', remote + '/trials.jsonl']
        if prepack_dir: args += ['--htp-kv-prepack','1']
        if config.get('htp_kv_release_before_decode'):
            if not prepack_dir: raise ValueError('提前释放只适用于私有 KV 原型')
            args += ['--htp-kv-release-before-decode','1']
        if config.get('release_calibration_state'):
            args += ['--oracle-release-calibration-state','1']
        for name in ('system', 'history', 'previous', 'prompt'):
            args += ['--' + name, remote + '/empty.txt']
        extra_env = f'GGML_HEXAGON_PROFILE={config.get("htp_profile", 0)} '
        if config.get('htp_use_hmx') is not None:
            extra_env += f'GGML_HEXAGON_USE_HMX={config["htp_use_hmx"]} '
        adsp_path = remote + '/probe-lib' if (probe_skel or prepack_dir) else '/data/local/tmp/llama.cpp/lib'
        shell = ('cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=' + shlex.quote(remote+':/data/local/tmp/llama.cpp/lib') + ' ' + 'ADSP_LIBRARY_PATH=' + shlex.quote(adsp_path) + ' '
            f'{extra_env}GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE={config["htp_queue"]} GGML_HEXAGON_OPBATCH={config["htp_opbatch"]}; '
            f'echo $$ > {remote}/pid; exec ' + shlex.join(args))
        (out / 'command.txt').write_text(shell)
        proc = subprocess.Popen(adb + ['shell', '-T', shell], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        protocol = Protocol(proc, out); protocol.wait('session_ready', timeout=config['timeout_s'])
        for p in config['new_tokens']:
            call('pull', remote + f'/trials.jsonl.P{p}.txt', str(out / f'P{p}.txt')); bodies[p] = (out / f'P{p}.txt').read_bytes()
        costs = {}; qualified = {}; counter = 0
        def trial(p, gap, strategy, route, phase, rep, pack=-1):
            nonlocal counter
            cc, hc = costs.get(p, (0, 0))
            prepare = any(qualified.get(key, False) and h < c for key, (c, h) in costs.items())
            protocol.send('TRIAL', counter, p, gap, strategy, route, phase, cc, hc, int(prepare), int(qualified.get(p, False)), pack)
            row = protocol.wait('trial_completed', counter, config['timeout_s'])['result']
            row['repetition'] = rep
            row['response_check'] = check_response(row, {'check': {'type': 'json_action', 'action': 'wait'}})
            rows.append(row)
            with (out / 'checked-rows.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            diagnostic = row['route'] == 'HC' and (phase == 'calibration' or not qualified.get(p, False))
            if (not row['passed'] or not row['response_check']['passed']) and not diagnostic:
                raise RuntimeError(f'响应或状态检查失败 trial={counter}')
            print(f'{counter} {phase} P={p} G={gap} {strategy}/{route}->{row["route"]} ready={row["ready_at_return"]} gap={row["actual_gap_ms"]:.1f} response={row["response_after_return_ms"]:.1f}', flush=True)
            counter += 1
        # 每个 P 先 CPU 参考，再独立三组完整路径校准，不把校准混入 Oracle 测量。
        for p in config['new_tokens']:
            for rep in range(3):
                for route in ('CC', 'HC'):
                    trial(p, 0, 'after_return', route, 'calibration', rep)
            costs[p] = tuple(statistics.median(r['prefill_ms'] + r['handoff_checked_ms'] + r['decode_ms']
                for r in rows if r['phase'] == 'calibration' and r['P'] == p and r['route'] == route) for route in ('CC', 'HC'))
            qualified[p] = all(r['numerical_passed'] and r['passed'] and r['response_check']['passed'] for r in rows if r['phase'] == 'calibration' and r['P'] == p)
            print(f'CALIBRATION P={p} HC_qualified={qualified[p]} costs={costs[p]}', flush=True)
        (out / 'candidate-qualification.json').write_text(json.dumps(qualified, indent=2))
        (out / 'full-response-costs.json').write_text(json.dumps(costs, indent=2))
        rng = random.Random(config['seed'])
        if config.get('prepack_comparison'):
            for rep in range(0 if config.get('calibration_only') else config['repetitions']):
                conditions=[(p,g,s,pack) for p in config['new_tokens'] for g in config['gap_ms'] for s in ('after_return','always') for pack in (0,1)]
                rng.shuffle(conditions)
                for p,g,s,pack in conditions: trial(p,g,s,'HC','measurement',rep,pack)

        for rep in range(0 if (config.get('calibration_only') or config.get('prepack_comparison')) else config['repetitions']):
            conditions = [(p, g, s, r) for p in config['new_tokens'] for g in config['gap_ms']
                for s in config['strategies'] for r in (*config['routes'], 'router')]
            rng.shuffle(conditions)
            for p, gap, strategy, route in conditions:
                trial(p, gap, strategy, route, 'measurement', rep)
        protocol.send('CLOSE'); protocol.wait('session_closed', timeout=60); proc.stdin.close()
        if proc.wait(timeout=60):
            raise RuntimeError('native 退出失败')
        (out / 'completion.json').write_text(json.dumps({'passed': True, 'trials': len(rows), 'measurement_trials': len(rows)-6*len(config['new_tokens']), 'all_HC_candidates_numerically_qualified': all(qualified.values())}, indent=2))
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps({'error': repr(exc)}, ensure_ascii=False, indent=2)); raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                call('shell', f'p=$(cat {remote}/pid); if tr "\\0" " " < /proc/$p/cmdline | grep -F -q "{remote}/agent "; then kill -TERM "$p"; fi')
                proc.wait(timeout=30)
            except Exception as exc:
                (out / 'stop-error.txt').write_text(str(exc))
        if protocol:
            protocol.reader.join(timeout=5)
            (out / 'native-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in protocol.events))
        log.close(); server.shutdown(); server.server_close(); thread.join(timeout=5)
        for args in [('pull', remote, str(out / 'device')), ('reverse', '--remove', f'tcp:{port}')]:
            try:
                call(*args)
            except Exception as exc:
                with (out / 'cleanup-errors.txt').open('a') as f: f.write(str(exc) + '\n')
        # 原始文件已拉取才清理本次目录；失败时留设备副本供检查。
        if (out / 'completion.json').exists():
            call('shell', 'rm', '-rf', remote)
    if config.get('calibration_only'):
        result = {'diagnostic_only': True, 'timing_valid_for_formal_comparison': False, 'HC_qualified': qualified,
            'trials': len(rows), 'config': config, 'costs': costs,
            'response_and_state_passed': all(r['passed'] and r['response_check']['passed'] for r in rows)}
        (out / 'diagnostic-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    elif not config.get('prepack_comparison'):
        subprocess.run([sys.executable, str(ROOT / 'scripts/analyze/oracle.py'), str(out)], check=True)


if __name__ == '__main__':
    main()
