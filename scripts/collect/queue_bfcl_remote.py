#!/usr/bin/env python3
"""远端持久队列：准备不使用 GPU；SRNO 与其他 GPU 任务退出后启动采集。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(command, logfile, **kwargs):
    with logfile.open('a') as log:
        log.write('\nCOMMAND ' + repr(command) + '\n')
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, **kwargs)


def busy(config):
    blockers = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            raw = (directory/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(directory/'cwd')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # A queue can be alive between two GPU jobs, so GPU utilization alone is insufficient.
        if ('/data/exuan/' in raw and any(s in raw.lower() for s in ('srno', 'queue_amp', 'ms_eval'))
            or (cwd.startswith(config['wait']['srno_root']) and 'queue' in raw)
            or ('queue_amp.sh' in raw and raw.split()[0].endswith(('bash', 'sh')))
            or (int(directory.name) == config['wait']['srno_queue_pid'] and 'queue' in raw)):
            # Ignore shell commands that merely inspect another task.
            if 'bash -c ' not in raw and 'ssh ' not in raw:
                blockers.append({'pid':int(directory.name), 'command':raw[:500]})
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    blockers.extend({'gpu_compute':line} for line in result.stdout.splitlines() if line.strip())
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if int(line) > config['wait']['gpu_utilization_max']:
            blockers.append({'gpu_utilization':int(line)})
    return blockers


def prepare(config, out):
    build = Path(config['build'])
    build.mkdir(parents=True, exist_ok=True)
    with (build/'a2o-build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cmake = config['cmake']
        run([cmake, '-S', config['source'], '-B', str(build), '-DGGML_CUDA=ON',
             '-DGGML_HEXAGON=OFF', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_CUDA_ARCHITECTURES=89', '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12',
             '-DCMAKE_C_COMPILER=/usr/bin/gcc-12', '-DCMAKE_CXX_COMPILER=/usr/bin/g++-12',
             '-DCMAKE_CXX_FLAGS=-include cstdarg',
             '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF'], out/'build.log')
        run(['nice', '-n', '19', cmake, '--build', str(build), '--target', 'llama-server', '-j', '2'],
            out/'build.log')
    write(out/'prepared.json', {'binary':str(build/'bin/llama-server'),
                               'sha256':digest(build/'bin/llama-server'), 'time':time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), '--config',
                   str(args.config.resolve()), '--output', str(out)]
        with (out/'queue.log').open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'queue_pid':process.pid, 'output':str(out)}))
        return
    if args.prepare_only:
        prepare(config, out)
        return
    with (out/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out/'queue.pid').write_text(str(os.getpid())+'\n')
        status = {'status':'waiting', 'pid':os.getpid(), 'config':config}
        write(out/'manifest.json', status)
        idle = 0
        while idle < config['wait']['idle_samples']:
            blockers = busy(config)
            idle = 0 if blockers else idle+1
            status.update(status='waiting', checked_at=time.time(), idle_samples=idle, blockers=blockers)
            write(out/'status.json', status)
            if idle < config['wait']['idle_samples']:
                time.sleep(config['wait']['poll_seconds'])
        # Verify artifact before allocating the GPU. A changed/incomplete upload is a hard failure.
        if digest(config['model']) != config['model_sha256']:
            raise ValueError('model SHA-256 mismatch')
        prepare(config, out)
        # Preparation may have taken time: recheck ownership before model initialization.
        if busy(config):
            raise RuntimeError('GPU or SRNO became busy during preparation; refusing to compete')
        server = None
        container_name = 'a2o-bfcl-collector-' + str(os.getpid())
        def stop(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        server_command = [str(Path(config['build'])/'bin/llama-server'), '-m', config['model'],
                          '--device', config['backend'], '-ngl', '99', '-np', str(config['slots']),
                          '-c', str(config['context_per_slot']*config['slots']), '-b', '512', '-ub', '128',
                          '-t', '4', '--host', '127.0.0.1', '--port', str(config['port']),
                          '--no-context-shift', '--slots', '--metrics']
        write(out/'server-command.json', server_command)
        try:
            status.update(status='starting', started_at=time.time(), blockers=[])
            write(out/'status.json', status)
            with (out/'server.log').open('a') as log:
                server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            deadline = time.monotonic()+1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'llama-server exited {server.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{config["port"]}/health', timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('server readiness')
                    time.sleep(2)
            serverlog = (out/'server.log').read_text(errors='replace')
            if 'CUDA0' not in serverlog or 'offloaded' not in serverlog:
                raise RuntimeError('CUDA allocation not confirmed in server log')
            # Read-only shared models/reference/source; only named results directory is writable.
            command = ['docker', 'run', '--rm', '--name', container_name, '--network', 'host',
                       '--cpus', '6', '--memory', '12g', '--entrypoint', 'python',
                       '-e', 'PYTHONDONTWRITEBYTECODE=1',
                       '-v', config['workspace']+':'+config['workspace']+':ro',
                       '-v', str(out)+':'+str(out), config['collector_image'],
                       config['project']+'/scripts/collect/bfcl.py', '--config',
                       str(args.config.resolve()), '--output', str(out)]
            status.update(status='running', server_pid=server.pid, collector_container=container_name)
            write(out/'status.json', status)
            write(out/'collector-command.json', command)
            run(command, out/'collector.log')
            run(['python3', config['project']+'/scripts/analyze/bfcl.py', str(out)], out/'analysis.log')
            completion = json.loads((out/'completion.json').read_text())
            status.update(status='completed', finished_at=time.time(), summary=completion['summary'])
        except BaseException:
            status.update(status='failed', finished_at=time.time(), error=traceback.format_exc())
            raise
        finally:
            subprocess.run(['docker', 'stop', '-t', '10', container_name], capture_output=True)
            if server and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            write(out/'status.json', status)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # Keep failures even when preflight failed before the server try/finally.
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output')+1])
            output.mkdir(parents=True, exist_ok=True)
            write(output/'failure.json', {'error':traceback.format_exc(), 'time':time.time()})
            if not '--prepare-only' in sys.argv:
                old = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
                old.update(status='failed', error=traceback.format_exc())
                write(output/'status.json', old)
        raise
