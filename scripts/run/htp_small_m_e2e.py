#!/usr/bin/env python3
"""Termux SSH持锁串行运行完整模型原版/内部补齐ABBA；相同H4096快照。"""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('refusing to overwrite previous measurements')
    t = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    remote = t.remote_root + '/' + out.name
    q = shlex.quote
    source = ROOT / 'results/20261111-prefill-p24-128-geometric-cpu6-t10/evidence/token-source.response.json'
    choice = json.loads(source.read_text())['choices'][0]
    fixture = dict(prefix_ids=choice['prompt_token_ids'][:4096], input_ids=choice['prompt_token_ids'][4096:4127],
                   generated_ids=choice['token_ids'][:16])
    assert tuple(map(len, fixture.values())) == (4096, 31, 16)
    (out / 'fixture.json').write_text(json.dumps(fixture) + '\n')
    guarded = json.loads((ROOT / 'results/20261112-htp-device-small-m/build-guarded/completion.json').read_text())
    binary = out / 'build-bench/probe'
    seed = t.remote_root + '/20261111-pd-prefill-dense-h4096-cpu6-t10-service/replay-cache/7a6d633cad63502d.json'
    model = '/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf'
    order = [('baseline-a', False), ('guarded-a', True), ('guarded-b', True), ('baseline-b', False)]
    active = False

    def status(phase, **kw):
        (out / 'status.json').write_text(json.dumps(dict(phase=phase, updated_unix=time.time(), **kw), ensure_ascii=False, indent=2) + '\n')

    try:
        status('acquiring_lock')
        t.prepare_ssh(out)
        t.call('shell', 'mkdir', '-p', remote + '/guarded')
        t.call('push', str(binary), remote + '/probe')
        t.call('shell', 'chmod', '755', remote + '/probe')
        t.call('push', str(out / 'fixture.json'), remote + '/fixture.json')
        t.call('push', str(ROOT / 'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so'), remote + '/libggml.so')
        assert hashlib.sha256(Path(guarded['binary']).read_bytes()).hexdigest() == guarded['sha256']
        t.call('push', guarded['binary'], remote + '/guarded/libggml-htp-v75.so')
        t.call('pull', seed, str(out / 'seed-metadata.json'))
        identities = t.call('shell', 'sha256sum', *[t.config['runtime_library_dir'] + '/' + x for x in ['libllama.so', 'libggml-cpu.so', 'libggml-htp-v75.so']]).stdout.decode()
        manifest = dict(order=order, guarded_build=guarded, seed=seed, model=model, runtime_library_identities=identities,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), fixture_sha256=hashlib.sha256((out / 'fixture.json').read_bytes()).hexdigest(),
            H=4096, P=31, D=16, repetitions_per_process=10, htp_host_threads=4, snapshot_cpu_threads=6,
            backend='HTP0, original and guarded; same host binary/libraries; private DSP only',
            timing='restore_work, prefill, forced_decode; init and all validation excluded; no vision/tool/network latency')
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        library_path = remote + ':' + t.remote_root + '/a2o_oracle_20261106-143411-669461:' + t.config['runtime_library_dir'] + ':/vendor/lib64'
        for label, use_guarded in order:
            status('running', condition=label)
            adsp = remote + '/guarded' if use_guarded else t.config['runtime_library_dir']
            env = 'export LD_LIBRARY_PATH=' + q(library_path) + ' ADSP_LIBRARY_PATH=' + q(adsp) + ' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1 GGML_HEXAGON_USE_HMX=1 GGML_HEXAGON_PROFILE=0; '
            command = env + shlex.join([remote + '/probe', model, remote + '/fixture.json', seed, remote + '/' + label + '.json'])
            command += '; rc=$?; echo "$rc" > ' + q(remote + '/' + label + '.exit')
            launch = 'nohup /system/bin/sh -c ' + q(command) + ' > ' + q(remote + '/' + label + '.log') + ' 2>&1 < /dev/null & echo $! > ' + q(remote + '/pid')
            active = True
            t.call('shell', launch)
            for tick in range(300):
                result = t.call('shell', 'if [ -f ' + q(remote + '/' + label + '.exit') + ' ]; then cat ' + q(remote + '/' + label + '.exit') + '; else echo running; fi').stdout.decode().strip()
                if result != 'running':
                    break
                if tick % 3 == 0:
                    # 只读拉取进度JSON；写入中读到的残缺JSON忽略，不启动额外设备任务。
                    state = t.call('shell', 'if [ -f ' + q(remote + '/' + label + '.json') + ' ]; then cat ' + q(remote + '/' + label + '.json') + '; else echo "{}"; fi').stdout
                    try:
                        current = json.loads(state)
                        status('running', condition=label, completed_trials=len(current.get('trials', [])), validation_ready='validation' in current)
                    except json.JSONDecodeError:
                        pass
                time.sleep(3)
            else:
                raise RuntimeError('phone timeout; lock retained')
            active = False
            t.call('pull', remote + '/' + label + '.log', str(out / (label + '.log')))
            if result != '0':
                raise RuntimeError(label + ' failed exit=' + result)
            for suffix in ['.json', '.json.prefill.f32', '.json.forced.f32']:
                t.call('pull', remote + '/' + label + suffix, str(out / (label + suffix)))
            data = json.loads((out / (label + '.json')).read_text())
            assert data['complete'] and len(data['trials']) == 10
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
