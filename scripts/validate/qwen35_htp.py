#!/usr/bin/env python3
"""持锁验证现有 HTP runtime 能否加载 Qwen3.5；失败保留原始日志。"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.phone_preflight import require_phone_preflight
from a2o.validation.oracle import active_model_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--context-size', type=int, default=16384)
    parser.add_argument('--max-buffer-mib', type=int, default=1024)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--kv-offload', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    host_lock = (ROOT / 'results/.termux-model.lock').open('a')
    fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    remote = t.remote_root + '/' + out.name
    q = shlex.quote
    try:
        ps = t.call('shell', 'ps -A -o PID,ARGS').stdout.decode()
        (out / 'processes-before.txt').write_text(ps)
        if active_model_processes(ps):
            raise RuntimeError('设备存在模型进程，拒绝并发')
        t.prepare_ssh(out)
        t.call('shell', 'mkdir', remote)
        files = {'system': '', 'prompt': '<|im_start|>system\nRespond with JSON only.<|im_end|>\n'
                 '<|im_start|>user\nReply exactly {"status":"ready"}.<|im_end|>\n'
                 '<|im_start|>assistant\n<think>\n\n</think>\n\n', 'empty': ''}
        for name, value in files.items():
            path = out / (name + '.txt')
            path.write_text(value)
            t.call('push', str(path), remote + '/' + path.name)
        binary = t.remote_root + '/decode_20261106-161056-356843/agent'
        if args.binary:
            binary = remote + '/agent'
            t.call('push', str(args.binary), binary)
            t.call('shell', 'chmod', '755', binary)
        model = '/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf'
        command = [binary, '--model', model, '--mode', 'npu', '--policy', 'none',
                   '--format', 'raw',
                   '--pre-move', '0', '--cpu-repack', '0', '--vision-backend', 'cpu',
                   '--ctx-size', str(args.context_size), '--batch-size', '128', '--max-tokens', '128',
                   '--system', remote + '/system.txt', '--prompt', remote + '/prompt.txt',
                   '--history', remote + '/empty.txt', '--previous', remote + '/empty.txt',
                   '--out', remote + '/result.json']
        if not args.kv_offload:
            if not args.binary:
                raise ValueError('显式 CPU KV 配置需要本轮新构建，旧 agent 不识别该开关')
            command += ['--htp-kv-offload', '0']
        cpp_runtime = t.remote_root + '/a2o_oracle_20261106-143411-669461'
        env = ('export LD_LIBRARY_PATH=' + q(str(Path(binary).parent) + ':' + cpp_runtime + ':' + t.config['runtime_library_dir'] + ':/vendor/lib64:/vendor/lib64/egl')
               + ' ADSP_LIBRARY_PATH=' + q(t.config['runtime_library_dir'])
               + ' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1'
               + ' GGML_HEXAGON_MBUF=' + str(args.max_buffer_mib*1024*1024) + '; ')
        # Android toybox 启动时清理 linker 环境，必须在 timeout 的子 shell 内设置。
        shell = 'timeout ' + str(args.timeout) + ' /system/bin/sh -c ' + q(env + 'exec ' + shlex.join(command))
        (out / 'command.txt').write_text(shell + '\n')
        hashes = t.call('shell', 'sha256sum ' + q(binary) + ' ' + q(model)).stdout.decode()
        (out / 'asset-hashes.txt').write_text(hashes)
        with (out / 'runtime.log').open('w') as log:
            require_phone_preflight(t, out)
            result = subprocess.run(t.shell_command(shell), stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout + 30)
        (out / 'exit-code.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        t.call('pull', remote, str(out / 'device'))
        passed = False
        if result.returncode == 0:
            data = json.loads((out / 'device/result.json').read_text())
            passed = data['eog'] and json.loads(data['output']) == {'status': 'ready'}
        summary = dict(passed=passed, returncode=result.returncode, backend='HTP0 explicitly requested',
                       model=model, context_size=args.context_size, batch_size=128,
                       max_buffer_mib=args.max_buffer_mib,
                       kv_offload=bool(args.kv_offload),
                       scope='load and normal response only; not state migration or e2e',
                       gpu_candidate=False, kv_policy='none', pre_move=False)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
        if not passed:
            raise RuntimeError('Qwen3.5 HTP 前置验收失败，禁止将该条件作为有效 E2E 性能数据')
    except BaseException as exc:
        (out / 'failure.json').write_text(json.dumps({'error': repr(exc)}, ensure_ascii=False) + '\n')
        raise
    finally:
        if t.lock_owner:
            t.release_lock()
            subprocess.run(t.prefix + ['termux-wake-unlock'], capture_output=True, timeout=30)


if __name__ == '__main__':
    main()
