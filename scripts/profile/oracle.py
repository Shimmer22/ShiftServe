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
from a2o.validation.observation import measurement_grid
from a2o.runtime.device_transport import DeviceTransport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device')
    parser.add_argument('--transport', choices=('ssh', 'adb'), default='ssh')
    parser.add_argument('--ssh-config', type=Path, default=ROOT / 'configs/termux-ssh.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/oracle-grid.json')
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--calibration-only', action='store_true', help='仅作独立诊断，不与正式时延矩阵混合')
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    wait_cases = config.get('pending_wait_cases', [])
    if wait_cases:
        if (config.get('paired_post_tool') or config.get('observation_scan') or config.get('prepack_comparison')
                or config.get('cpu_thread_scan') or config.get('htp_profile', 0)
                or config['repetitions'] < 20 or config['repetitions'] % 2
                or not 0 < config['pending_wait_cap_ms'] <= 1000
                or config['pending_wait_margin_ms'] < 0 or config['pending_wait_prepare_factor'] < 1
                or len({tuple(c) for c in wait_cases}) != len(wait_cases)
                or any(len(c) != 3 or c[0] not in config['new_tokens'] or c[1] not in config['gap_ms']
                       or c[2] not in ('always', 'cost_aware') for c in wait_cases)):
            raise ValueError('有限等待要求独立无profile配对、至少20个偶数块和显式预算参数')
    paired_post_tool = config.get('paired_post_tool', False)
    if paired_post_tool and (config['new_tokens'] != [31] or config['gap_ms'] != [0, 200]
            or not config.get('post_tool_routing') or config.get('observation_scan')
            or config.get('calibration_only') or config.get('prepack_comparison')
            or config.get('cpu_thread_scan') or config.get('htp_profile', 0)
            or config['repetitions'] < 20 or config['repetitions'] % 2):
        raise ValueError('P31配对复测要求独立校准、G200、偶数且至少20块，无其他扫描或profile')
    thread_scan=config.get('cpu_thread_scan',[])
    if thread_scan and (4 not in thread_scan or len(set(thread_scan))!=len(thread_scan) or
                        any(type(t) is not int or not 1<=t<=8 for t in thread_scan) or
                        config.get('observation_scan') or config.get('prepack_comparison')):
        raise ValueError('CPU线程扫描必须含4线程参考，线程1..8且不重复，并与主图/tiles矩阵分开')
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
    if not config['new_tokens'] or len(set(config['new_tokens'])) != len(config['new_tokens']) or any(not 16 <= p <= 2048 for p in config['new_tokens']):
        raise ValueError('P 必须是 16..2048 的不重复整数')
    if 0 not in config['gap_ms'] or len(set(config['gap_ms'])) != len(config['gap_ms']) or any(g < 0 for g in config['gap_ms']):
        raise ValueError('G 必须包含 0（用于校准），不能重复或为负')
    if config['strategies'] != ['after_return', 'always', 'cost_aware'] or config['routes'] != ['CC', 'HC']:
        raise ValueError('要求完整三策略、CC/HC 及 Router 重放')
    grid = measurement_grid(config)
    if config.get('observation_scan') and (config.get('prepack_comparison') or config.get('htp_kv_prepack_dir')
            or config.get('calibration_only') or config.get('htp_profile', 0) or not config.get('post_tool_routing')):
        raise ValueError('Observation 主图要求无 tiles/profile 的正式测量，并启用真实 Post-tool 对照')
    pack_strategies=config.get('prepack_strategies',['after_return','always'])
    measured_gaps=config.get('measurement_gap_ms',config['gap_ms'])
    if pack_strategies not in (['after_return','always'],['always']):
        raise ValueError('prepack_strategies must be after_return/always or always only')
    if not measured_gaps or len(set(measured_gaps))!=len(measured_gaps) or any(g not in config['gap_ms'] for g in measured_gaps):
        raise ValueError('measurement_gap_ms must be a nonempty unique subset of gap_ms')
    if not config.get('prepack_comparison') and ('prepack_strategies' in config or 'measurement_gap_ms' in config):
        raise ValueError('focused measurements require prepack_comparison')
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
    skip_calibration=config.get('skip_calibration',False)
    if skip_calibration and (not config.get('prepack_comparison') or pack_strategies!=['always'] or not config.get('response_reference')):
        raise ValueError('skip_calibration requires forced HC early comparison and recorded response reference')
    binary = build()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / (stamp + '-http-oracle-grid'); out.mkdir()
    print(str(out), flush=True)
    transport = DeviceTransport(ROOT, a.transport, a.device, a.ssh_config)
    remote = transport.remote_root + '/a2o_oracle_' + stamp
    runtime_lib = transport.config['runtime_library_dir'] if a.transport == 'ssh' else '/data/local/tmp/llama.cpp/lib'
    def call(*args):
        return transport.call(*args)
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
    manifest = {'config': config, 'device': transport.identity, 'transport': a.transport, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source_files_sha256': native_source_hashes(), 'runtime_package': str(PKG),
        'python_files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / 'scripts/analyze/oracle.py')},
        'scope': '真实 HTTP Tool + 原生并发准备；文本固定 H/P、完整自由生成到 EOG；无图像、无能耗结论',
        'clock': '设备 CLOCK_BOOTTIME 统一计量 HTTP/准备/完整响应，包含系统挂起；steady_clock 仅作排除挂起的诊断；服务器 monotonic_ns 独立，不跨时钟相减',
        'clock_id': 'CLOCK_BOOTTIME',
        'preparation': '从共同已存在 CPU sequence snapshot 导入 HTP，包含 clear/import/字节核验；不含共同快照首次导出',
        'decode': 'CPU 完整响应，含 EOG 前向评估；所有 P/策略/路径生成 ID 必须一致',
        'router': '独立完整 D 校准；发起时仅按所有 P 是否有潜在收益决定准备，返回后按实际 P 及 ready 决定 CC/HC；不等 pending',
        'oracle': '强制路径重放是实测反事实，不是同时可观测的单轮真值；严格选路只允许返回时 ready 的 HC；另报允许等待/返回后准备的扩展上界',
        'backend_diagnostic': {'profile': config.get('htp_profile', 0), 'use_hmx': config.get('htp_use_hmx', 'default'), 'calibration_only': config.get('calibration_only', False)},
        'vision_backend': 'none', 'decode_backend': 'cpu', 'kv_type': 'F16', 'flash_attention': False,
        'http_transport': 'Android loopback via ' + ('SSH reverse forwarding' if a.transport == 'ssh' else 'adb reverse') + ' to host controlled HTTP server',
        'model': '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf'}
    manifest['numeric_calibration_performed']=not skip_calibration
    manifest['history_backend']=config.get('history_backend','cpu')
    if config.get('post_tool_routing'):
        manifest['post_tool_routing'] = 'after_return/router 在 Tool 完整返回后按独立校准 CC 与 HC+准备成本选路，允许按需准备；不是 ready-only Router'
    if wait_cases:
        manifest['pending_wait'] = '全额准备上界=独立三次HC校准prepare最大值×显式factor；HC+上界+margin<CC才准入，最多等min(cap,CC-HC-margin)，不读取本轮完成时间/实时剩余时间'
        manifest['expected_trials'] = 6 * len(config['new_tokens']) + (0 if config.get('calibration_only') else 2 * len(wait_cases) * config['repetitions'])
        manifest['pending_wait_pairs'] = '同一H/P/G/strategy相邻AB/BA，ready_only与bounded各半先运行；所有case在每块随机排列'
    if config.get('observation_scan'):
        manifest['measurement_grid'] = grid
        manifest['expected_trials'] = 6 * len(config['new_tokens']) + len(grid) * config['repetitions']
    if paired_post_tool:
        manifest['measurement_grid'] = [[31, 200, 'after_return', r] for r in ('CC', 'router')]
        manifest['expected_trials'] = 6 + 2 * config['repetitions']
        manifest['paired_design'] = '每块CC/router相邻；AB/BA各半后随机打乱；各H独立seed；CPU prefill/decode均4线程'
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
        if a.transport == 'ssh':
            if config.get('require_awake') or config.get('power_mode') != 'termux_wakelock':
                raise ValueError('SSH灭屏运行要求显式power_mode=termux_wakelock、require_awake=false')
            manifest['ssh_power_check'] = transport.prepare_ssh(out)
            manifest['process_visibility'] = 'Termux UID only; host campaign lock and remote Termux lock serialize model runs'
        identity = {key: call('shell', 'getprop', key).stdout.decode().strip()
                    for key in ('ro.product.model', 'ro.product.device', 'ro.soc.model', 'ro.build.version.release', 'ro.serialno')}
        manifest['device_identity'] = identity
        manifest['device_identity_access_limited'] = a.transport == 'ssh' and not identity['ro.serialno']
        power = call('shell', 'dumpsys', 'power').stdout.decode()
        (out / 'power-before.txt').write_text(power)
        manifest['power_precondition'] = ('termux_wakelock_clock_checked' if a.transport == 'ssh'
                                        else 'Awake' if 'mWakefulness=Awake' in power else 'not_Awake')
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        if config.get('require_awake') and manifest['power_precondition'] != 'Awake':
            raise RuntimeError('设备未处于 Awake，拒绝开始主图采样；不自动唤醒')
        hashes = call('shell', 'sha256sum ' + shlex.quote(manifest['model']) +
                      ' /data/local/tmp/llama.cpp/lib/libllama.so /data/local/tmp/llama.cpp/lib/libggml-hexagon.so /data/local/tmp/llama.cpp/lib/libggml-htp-*.so').stdout.decode()
        (out / 'device-files-sha256.txt').write_text(hashes)
        manifest['device_files_sha256'] = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                                          for line in hashes.splitlines()}
        if a.transport == 'ssh':
            loaded_hashes = call('shell', 'sha256sum ' + shlex.quote(runtime_lib) + '/*.so').stdout.decode()
            (out / 'loaded-runtime-sha256.txt').write_text(loaded_hashes)
            manifest['loaded_runtime_files_sha256'] = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                                                       for line in loaded_hashes.splitlines()}
            source_hashes = call('shell', 'sha256sum /data/local/tmp/llama.cpp/lib/*.so').stdout.decode()
            by_name = lambda s: {Path(line.split(maxsplit=1)[1]).name: line.split()[0] for line in s.splitlines()}
            if by_name(source_hashes) != by_name(loaded_hashes):
                raise RuntimeError('Termux私有运行库与原运行库hash不一致')
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        processes = call('shell', 'ps -A -o PID,ARGS').stdout.decode()
        (out / 'device-processes-before.txt').write_text(processes)
        if active_model_processes(processes):
            raise RuntimeError('设备已有模型实验进程，拒绝争抢资源')
        call('shell', 'mkdir', remote); call('push', str(binary), remote + '/agent'); call('shell', 'chmod', '755', remote + '/agent')
        if a.transport == 'ssh':
            # 复用本机NDK的匹配C++运行库，仅放本轮私有目录，不改共享库。
            ndk = sorted((Path.home() / 'Library/Android/sdk/ndk').iterdir(),
                         key=lambda p: tuple(map(int, p.name.split('.'))))[-1]
            cpp = next((ndk / 'toolchains/llvm/prebuilt').glob('*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'))
            call('push', str(cpp), remote + '/libc++_shared.so')
            manifest['private_cpp_runtime_sha256'] = hashlib.sha256(cpp.read_bytes()).hexdigest()
            (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
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
        if config.get('history_backend')=='htp':args += ['--oracle-history-backend','htp']
        if config.get('post_tool_routing'):
            args += ['--oracle-post-tool-routing', '1']
        if skip_calibration:
            reference=config['response_reference']
            args += ['--oracle-reference-ids',','.join(map(str,reference['generated_ids'])),'--oracle-reference-output',reference['output']]
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
        adsp_path = remote + '/probe-lib' if (probe_skel or prepack_dir) else runtime_lib
        library_path = remote + ':' + runtime_lib
        if a.transport == 'ssh':
            library_path += ':/vendor/lib64:/vendor/lib64/egl'
        manifest['library_path'] = library_path
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        shell = ('cd ' + shlex.quote(transport.remote_root if a.transport == 'ssh' else '/data/local/tmp/llama.cpp') + '; export LD_LIBRARY_PATH=' + shlex.quote(library_path) + ' ' + 'ADSP_LIBRARY_PATH=' + shlex.quote(adsp_path) + ' '
            f'{extra_env}GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE={config["htp_queue"]} GGML_HEXAGON_OPBATCH={config["htp_opbatch"]}; '
            f'echo $$ > {remote}/pid; exec ' + shlex.join(args))
        (out / 'command.txt').write_text(shell)
        proc = subprocess.Popen(transport.shell_command(shell), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        protocol = Protocol(proc, out); protocol.wait('session_ready', timeout=config['timeout_s'])
        for p in config['new_tokens']:
            call('pull', remote + f'/trials.jsonl.P{p}.txt', str(out / f'P{p}.txt')); bodies[p] = (out / f'P{p}.txt').read_bytes()
            if config.get('fixture_reference_run') and bodies[p] != (ROOT/config['fixture_reference_run']/f'P{p}.txt').read_bytes():
                raise ValueError('快速构造的 HTTP 输入与原夹具不同')
        if config.get('history_fixture_reference'):
            call('pull', remote+'/trials.jsonl.prefix.txt', str(out/'history-prefix.txt'))
            if (out/'history-prefix.txt').read_bytes() != (ROOT/config['history_fixture_reference']).read_bytes():
                raise ValueError('快速构造的历史输入与原夹具不同')
        costs = {}; qualified = {}; counter = 0
        def trial(p, gap, strategy, route, phase, rep, pack=-1, cpu_threads=4, comparison_order=None, pending_mode=0):
            nonlocal counter
            cc, hc = costs.get(p, (0, 0))
            if config.get('post_tool_routing') and strategy == 'after_return' and route == 'router':
                hc += statistics.median(r['prepare_ms'] for r in rows
                    if r['phase'] == 'calibration' and r['P'] == p and r['route'] == 'HC')
            prepare = any(qualified.get(key, False) and h < c for key, (c, h) in costs.items())
            upper = (max(r['prepare_ms'] for r in rows if r['phase'] == 'calibration' and r['P'] == p and r['route'] == 'HC')
                     * config['pending_wait_prepare_factor']) if pending_mode else 0
            protocol.send('TRIAL', counter, p, gap, strategy, route, phase, cc, hc, int(prepare), int(qualified.get(p, False)), pack, cpu_threads,
                          pending_mode, upper, config.get('pending_wait_cap_ms', 0), config.get('pending_wait_margin_ms', 0))
            row = protocol.wait('trial_completed', counter, config['timeout_s'])['result']
            row['repetition'] = rep
            if comparison_order is not None:
                row['comparison_block'] = rep
                row['comparison_order'] = comparison_order
            row['response_check'] = check_response(row, {'check': {'type': 'json_action', 'action': 'wait'}})
            rows.append(row)
            with (out / 'checked-rows.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            diagnostic = row['route'] == 'HC' and (phase == 'calibration' or (not skip_calibration and not qualified.get(p, False)))
            if (not row['passed'] or not row['response_check']['passed']) and not diagnostic:
                raise RuntimeError(f'响应或状态检查失败 trial={counter}')
            print(f'{counter} {phase} P={p} T={cpu_threads} G={gap} {strategy}/{route}->{row["route"]} policy={row["wait_policy"]} ready={row["ready_at_return"]} gap={row["actual_gap_ms"]:.1f} response={row["response_after_return_ms"]:.1f}', flush=True)
            counter += 1
        # 每个 P 先 CPU 参考，再独立三组完整路径校准，不把校准混入 Oracle 测量。
        for p in ([] if (skip_calibration or thread_scan) else config['new_tokens']):
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
        if wait_cases and not config.get('calibration_only'):
            if not all(qualified.values()):
                raise RuntimeError('有限等待候选未全部通过独立校准，拒绝进入测量')
            orders = {}
            for case in wait_cases:
                sequence = [(0, 1), (1, 0)] * (config['repetitions'] // 2)
                rng.shuffle(sequence); orders[tuple(case)] = sequence
            for block in range(config['repetitions']):
                cases = list(wait_cases); rng.shuffle(cases)
                for p, gap, strategy in cases:
                    for position, mode in enumerate(orders[(p, gap, strategy)][block]):
                        trial(p, gap, strategy, 'router', 'measurement', block,
                              comparison_order=position, pending_mode=mode)
        if paired_post_tool:
            orders = [('CC', 'router'), ('router', 'CC')] * (config['repetitions'] // 2)
            rng.shuffle(orders)
            for block, order in enumerate(orders):
                for position, route in enumerate(order):
                    trial(31, 200, 'after_return', route, 'measurement', block,
                          comparison_order=position)
        if thread_scan:
            for p in config['new_tokens']:
                order=[t for t in thread_scan if t!=4];rng.shuffle(order)
                for t in [4,*order]:trial(p,0,'after_return','CC','calibration',0,cpu_threads=t)
                qualified[p]=all(r['numerical_passed'] and r['passed'] and r['response_check']['passed'] for r in rows if r['P']==p)
                if not qualified[p]:raise RuntimeError(f'CPU线程数值或响应验收失败 P={p}')
            (out/'candidate-qualification.json').write_text(json.dumps(qualified,indent=2))
            for rep in range(config['repetitions']):
                conditions=[(p,t) for p in config['new_tokens'] for t in thread_scan];rng.shuffle(conditions)
                for p,t in conditions:trial(p,200,'after_return','CC','measurement',rep,cpu_threads=t)
        if config.get('prepack_comparison'):
            for rep in range(0 if config.get('calibration_only') else config['repetitions']):
                conditions=[(p,g,s,pack) for p in config['new_tokens'] for g in measured_gaps for s in pack_strategies for pack in (0,1)]
                rng.shuffle(conditions)
                for p,g,s,pack in conditions: trial(p,g,s,'HC','measurement',rep,pack)

        for rep in range(0 if (config.get('calibration_only') or config.get('prepack_comparison') or thread_scan or paired_post_tool or wait_cases) else config['repetitions']):
            conditions = list(grid)
            rng.shuffle(conditions)
            for p, gap, strategy, route in conditions:
                trial(p, gap, strategy, route, 'measurement', rep)
        protocol.send('CLOSE'); protocol.wait('session_closed', timeout=60); proc.stdin.close()
        if proc.wait(timeout=60):
            raise RuntimeError('native 退出失败')
        (out / 'completion.json').write_text(json.dumps({'passed': True, 'trials': len(rows), 'measurement_trials': sum(r['phase']=='measurement' for r in rows), 'numeric_calibration_performed':not skip_calibration,'all_HC_candidates_numerically_qualified': not thread_scan and not skip_calibration and all(qualified.values()),'all_CPU_thread_candidates_numerically_qualified': bool(thread_scan) and all(qualified.values())}, indent=2))
    except BaseException as exc:
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
        if proc is None or proc.poll() is not None:
            transport.release_lock()
    if config.get('calibration_only'):
        result = {'diagnostic_only': True, 'timing_valid_for_formal_comparison': False, 'HC_qualified': qualified,
            'trials': len(rows), 'config': config, 'costs': costs,
            'response_and_state_passed': all(r['passed'] and r['response_check']['passed'] for r in rows)}
        (out / 'diagnostic-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    elif not config.get('prepack_comparison') and not config.get('observation_scan') and not thread_scan and not paired_post_tool and not wait_cases:
        subprocess.run([sys.executable, str(ROOT / 'scripts/analyze/oracle.py'), str(out)], check=True)


if __name__ == '__main__':
    main()
