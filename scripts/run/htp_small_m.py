#!/usr/bin/env python3
"""Termux持锁串行比较原版、内部补齐、负对照；所有库仅加载私有目录。"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('refusing to overwrite existing phone measurements')
    t = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    q = shlex.quote
    remote = t.remote_root + '/' + out.name
    binary = out / 'build-bench/probe'
    builds = {key: json.loads((out / ('build-' + key) / 'completion.json').read_text())
              for key in ['guarded', 'force', 'guarded-profile']}
    order = [('baseline-a', None, 0), ('guarded-a', 'guarded', 0), ('force', 'force', 0),
             ('guarded-b', 'guarded', 0), ('baseline-b', None, 0), ('guarded-profile', 'guarded-profile', 2)]

    def status(phase, **kw):
        (out / 'status.json').write_text(json.dumps(dict(phase=phase, updated_unix=time.time(), **kw), ensure_ascii=False, indent=2) + '\n')

    active = False
    try:
        status('acquiring_lock')
        t.prepare_ssh(out)
        t.call('shell', 'mkdir', '-p', remote)
        t.call('push', str(binary), remote + '/probe')
        registry = ROOT / 'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'
        t.call('push', str(registry), remote + '/libggml.so')
        t.call('shell', 'chmod', '755', remote + '/probe')
        for key, build in builds.items():
            path = Path(build['binary'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == build['sha256']
            t.call('shell', 'mkdir', '-p', remote + '/' + key)
            t.call('push', str(path), remote + '/' + key + '/libggml-htp-v75.so')
        lib = remote + ':' + t.remote_root + '/a2o_oracle_20261106-143411-669461:' + t.config['runtime_library_dir'] + ':/vendor/lib64'
        model = '/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf'
        manifest = dict(order=order, builds=builds, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                        model=model, rows=31, repetitions_per_process=10, cpu_reference_threads=6,
                        scope='热权重单矩阵；外部M31保持不变；原版/guarded进程顺序ABBA；不代表完整prefill或模型状态验收')
        manifest['original_dsp_sha256'] = t.call('shell', 'sha256sum', t.config['runtime_library_dir'] + '/libggml-htp-v75.so').stdout.decode().split()[0]
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        for label, key, profile in order:
            status('running', condition=label)
            adsp = remote + '/' + key if key else t.config['runtime_library_dir']
            env = 'export LD_LIBRARY_PATH=' + q(lib) + ' ADSP_LIBRARY_PATH=' + q(adsp) + ' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1 GGML_HEXAGON_USE_HMX=1 GGML_HEXAGON_PROFILE=' + str(profile) + '; '
            command = env + shlex.join([remote + '/probe', model, remote + '/' + label + '.json']) + '; rc=$?; echo "$rc" > ' + q(remote + '/' + label + '.exit')
            launch = 'nohup /system/bin/sh -c ' + q(command) + ' > ' + q(remote + '/' + label + '.log') + ' 2>&1 < /dev/null & echo $! > ' + q(remote + '/pid')
            active = True
            t.call('shell', launch)
            for _ in range(180):
                result = t.call('shell', 'if [ -f ' + q(remote + '/' + label + '.exit') + ' ]; then cat ' + q(remote + '/' + label + '.exit') + '; else echo running; fi').stdout.decode().strip()
                if result != 'running':
                    break
                time.sleep(2)
            else:
                raise RuntimeError('phone process timeout; lock retained')
            active = False
            t.call('pull', remote + '/' + label + '.log', str(out / (label + '.log')))
            if result != '0':
                raise RuntimeError(label + ' exited ' + result)
            t.call('pull', remote + '/' + label + '.json', str(out / (label + '.json')))
            data = json.loads((out / (label + '.json')).read_text())
            assert len(data['cases']) == 3
            for case in data['cases']:
                assert len(case['trials']) == 10 and case['output_guard_passed']
                for condition in ['plain31', 'pad32']:
                    name = label + '.json.' + case['tensor'] + '.' + condition + '.f32'
                    t.call('pull', remote + '/' + name, str(out / name))
        status('complete')
    except Exception as exc:
        status('failed', error=str(exc), process_may_be_active=active)
        raise
    finally:
        if t.lock_owner and not active:
            t.release_lock()
            subprocess.run(t.prefix + ['termux-wake-unlock'], check=True, capture_output=True)


if __name__ == '__main__':
    main()
