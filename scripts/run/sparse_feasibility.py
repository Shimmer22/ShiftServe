#!/usr/bin/env python3
"""Termux SSH持锁串行运行CPU/HTP，回收部分结果与所有失败日志。"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport

DEFAULT_OUT = ROOT / 'results/20261112-ffn-sparse-feasibility'


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/sparse-feasibility.json')
    parser.add_argument('--binary', type=Path, default=DEFAULT_OUT / 'build/probe')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'completion.json').exists():
        raise RuntimeError('结果已完成，不覆盖')
    binary = args.binary.resolve()
    config = args.config.resolve()
    cfg = json.loads(config.read_text())
    source = ROOT / cfg['source']
    choice = json.loads(source.read_text())['choices'][0]
    normal = json.loads((ROOT / cfg['normal_source']).read_text())['__verbose']['prompt']
    fixture = dict(cfg, prefix_ids=choice['prompt_token_ids'][:cfg['H']],
                   input_ids=choice['prompt_token_ids'][cfg['H']:cfg['H'] + cfg['P']],
                   generated_ids=choice['token_ids'][:max(max(cfg['verify_k'], default=0) + 1, cfg['sparse_D'])],
                   normal_prompt=normal)
    assert len(fixture['prefix_ids']) == cfg['H'] and len(fixture['input_ids']) == cfg['P']
    assert len(fixture['generated_ids']) > max(cfg['verify_k'], default=0)
    save(OUT / 'fixture.json', fixture)
    save(OUT / 'manifest.json', dict(config=cfg, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
         fixture_sha256=hashlib.sha256((OUT / 'fixture.json').read_bytes()).hexdigest(),
         scope='CPU/HTP独立进程串行；完整prefill与decode在同一上下文；固定前部通道；无质量及先验有效性结论',
         zero_ratio='移除FFN三个矩阵，保留SiLU(norm(x))作为形状兼容替代；仅为性能诊断，不是合法模型',
         overhead='所有条件关闭图复用；每条件一次预热；打包和完整状态恢复独立计时；不计入稳态decode',
         cpu_threads=6, htp_host_threads=4, kv_type='F16', kv_offload=False, kv_compaction=False, pre_move=False))
    t = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    q = shlex.quote
    remote = t.remote_root + '/' + OUT.name
    def status(phase, **kw):
        save(OUT / 'status.json', dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **kw))
    held = False
    running = False
    try:
        ps = t.call('shell', 'ps -A -o PID,ARGS').stdout.decode()
        (OUT / 'processes-before.txt').write_text(ps)
        busy = [line for line in ps.splitlines() if any(name in line for name in
                ['/service ', '/probe ', 'llama-server ', '/agent '])]
        if busy:
            raise RuntimeError('已有模型进程：' + '\n'.join(busy))
        status('acquiring_lock')
        t.prepare_ssh(OUT)
        held = True
        t.call('shell', 'mkdir', '-p', remote)
        for local, name in [(binary, 'probe'), (OUT / 'fixture.json', 'fixture.json'),
                            (ROOT / 'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry/libggml.so', 'libggml.so')]:
            t.call('push', str(local), remote + '/' + name)
        t.call('shell', 'chmod', '755', remote + '/probe')
        identities = t.call('shell', 'sha256sum ' + shlex.join([remote + '/probe', remote + '/libggml.so',
            t.config['runtime_library_dir'] + '/libllama.so', t.config['runtime_library_dir'] + '/libggml-hexagon.so',
            t.config['runtime_library_dir'] + '/libggml-htp-v75.so'])).stdout
        (OUT / 'device-identities.txt').write_bytes(identities)
        assert identities.decode().split()[0] == hashlib.sha256(binary.read_bytes()).hexdigest()
        if cfg.get('expected_libllama_sha256'):
            assert any(line.split()[0] == cfg['expected_libllama_sha256'] and line.endswith('/libllama.so') for line in identities.decode().splitlines())
        lib = remote + ':' + t.remote_root + '/a2o_oracle_20261106-143411-669461:' + t.config['runtime_library_dir'] + ':/vendor/lib64'
        for backend in cfg['backends']:
            status('running', backend=backend)
            model = '/data/data/com.termux/files/home/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf'
            command = 'export LD_LIBRARY_PATH=' + q(lib) + ' ADSP_LIBRARY_PATH=' + q(t.config['runtime_library_dir'])
            command += ' GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_OPBATCH=1 GGML_HEXAGON_USE_HMX=1 GGML_HEXAGON_PROFILE=0; '
            command += shlex.join([remote + '/probe', model, remote + '/fixture.json', backend, remote + '/' + backend + '.json'])
            command += '; result=$?; echo "$result" > ' + q(remote + '/' + backend + '.exit')
            save(OUT / (backend + '-command.json'), dict(command=command))
            t.call('shell', 'test ! -e ' + q(remote + '/' + backend + '.exit'))
            launch = 'nohup /system/bin/sh -c ' + q(command) + ' > ' + q(remote + '/' + backend + '.log') + ' 2>&1 < /dev/null & echo $! > ' + q(remote + '/pid')
            t.call('shell', launch)
            running = True
            begin = time.monotonic()
            while True:
                result = t.call('shell', 'if [ -f ' + q(remote + '/' + backend + '.exit') + ' ]; then cat ' + q(remote + '/' + backend + '.exit') + '; else echo running; fi').stdout.decode().strip()
                exists = t.call('shell', 'test -f ' + q(remote + '/' + backend + '.json') + ' && echo yes || echo no').stdout.decode().strip()
                if exists == 'yes':
                    t.call('pull', remote + '/' + backend + '.json', str(OUT / (backend + '.json')))
                    partial = json.loads((OUT / (backend + '.json')).read_text())
                    status('running', backend=backend, rows=len(partial['rows']), conditions=len(partial.get('conditions', [])),
                           history_ready='state_bytes' in partial, elapsed_s=time.monotonic() - begin)
                if result != 'running':
                    running = False
                    t.call('pull', remote + '/' + backend + '.log', str(OUT / (backend + '.log')))
                    save(OUT / (backend + '-exit.json'), dict(exit_code=int(result)))
                    if result != '0':
                        raise RuntimeError(backend + '探针失败，退出码' + result)
                    assert partial['completed'] and partial['normal_response']['passed']
                    assert len(partial['verification']) == len(cfg['verify_k'])
                    break
                if time.monotonic() - begin > 3600:
                    raise RuntimeError('探针超时，保留进程和锁，禁止并行启动')
                time.sleep(10)
        (OUT / 'processes-after.txt').write_bytes(t.call('shell', 'ps -A -o PID,ARGS').stdout)
        save(OUT / 'completion.json', dict(completed=True, backends=cfg['backends']))
        status('complete')
    except Exception as exc:
        status('failed', error=str(exc), process_may_be_running=running)
        raise
    finally:
        if held and not running:
            t.release_lock()
            subprocess.run(t.prefix + ['termux-wake-unlock'], check=True, capture_output=True, timeout=30)


if __name__ == '__main__':
    main()
