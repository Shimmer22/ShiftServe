#!/usr/bin/env python3
"""复用 wins 缓存 SDK/NDK 构建私有 Qwen3.5 后端验收入口。"""
import hashlib
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TARGET = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261107-qwen35-route-preflight'
BASE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
IMAGE = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', action='store_true')
    parser.add_argument('--host', default='wins')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--remote-directory')
    args = parser.parse_args()
    out = args.output or ROOT / ('results/20261107-toolsandbox-route-e2e/build-service-' + datetime.now().strftime('%H%M%S') if args.service else
                  'results/20261107-toolsandbox-route-e2e/build-preflight')
    target = args.remote_directory or TARGET
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run(['ssh', args.host, 'mkdir -p ' + shlex.quote(target + '/native')], check=True)
    subprocess.run(['rsync', '-a', str(ROOT / 'src/a2o/native') + '/', args.host + ':' + target + '/native/'], check=True)
    compile_command = ['/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++',
        '-std=c++17', '-O2', '-fPIC', '-pie', '-I/base/src/include', '-I/base/src/ggml/include',
        '-I/base/src/tools/mtmd', '-I/probe/native', '/probe/native/agent_runtime.cpp',
        '-L/base/build-snapdragon/bin', '-lllama', '-lmtmd', '-lggml', '-lggml-base',
        '-llog', '-ldl', '-lm', '-o', '/probe/agent']
    if args.service:
        compile_command[compile_command.index('/probe/native/agent_runtime.cpp')] = '/probe/native/session/phone_service.cpp'
        compile_command += ['-I/base/src/common', '-I/base/src/vendor', '-I/base/src/vendor/cpp-httplib',
                            '/base/src/vendor/cpp-httplib/httplib.cpp', '-lllama-common']
    command = ['docker', 'run', '--rm', '--pull=never', '-v', BASE + ':/base:ro', '-v', target + ':/probe',
               IMAGE, *compile_command]
    manifest = dict(image=IMAGE, remote_directory=target, command=command,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / 'src/a2o/native').rglob('*')) if p.is_file()},
        shared_modified=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (out / 'build.log').open('w') as log:
        subprocess.run(['ssh', args.host, shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = out / 'agent'
    subprocess.run(['scp', args.host + ':' + target + '/agent', str(binary)], check=True)
    (out / 'binary-sha256.txt').write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + '\n')
    print(binary)


if __name__ == '__main__':
    main()
