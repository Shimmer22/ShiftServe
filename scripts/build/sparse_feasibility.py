#!/usr/bin/env python3
"""缓存工具链内构建私有稀疏可行性探针，不修改shared与原库。"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/20261112-ffn-sparse-feasibility/build'
REMOTE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261112-ffn-sparse-feasibility'
BASE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
IMAGE = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.3'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ssh', 'wins', 'mkdir -p ' + shlex.quote(REMOTE + '/native')], check=True)
    subprocess.run(['rsync', '-a', str(ROOT / 'src/a2o/native') + '/', 'wins:' + REMOTE + '/native/'], check=True)
    cc = ['/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++',
          '-std=c++17', '-O2', '-fPIC', '-pie', '-I/base/src/include', '-I/base/src/ggml/include',
          '-I/base/src/src', '-I/base/src/ggml/src', '-I/base/src/tools/mtmd', '-I/base/src/vendor', '-I/probe/native',
          '/probe/native/profiling/sparse_feasibility.cpp', '-L/base/build-snapdragon/bin',
          '-lllama', '-lmtmd', '-lggml', '-lggml-base', '-llog', '-ldl', '-lm', '-o', '/probe/probe']
    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '-v', BASE + ':/base:ro',
               '-v', REMOTE + ':/probe', IMAGE, *cc]
    manifest = dict(command=command, shared_modified=False, source_sha256={
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / 'src/a2o/native').rglob('*')) if p.is_file()})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if (OUT / 'build.log').exists():
        (OUT / 'build.log').rename(OUT / ('build-' + str(time.time_ns()) + '.log'))
    with (OUT / 'build.log').open('w') as log:
        subprocess.run(['ssh', 'wins', shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['scp', 'wins:' + REMOTE + '/probe', str(OUT / 'probe')], check=True)
    manifest['binary_sha256'] = hashlib.sha256((OUT / 'probe').read_bytes()).hexdigest()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(OUT / 'probe')


if __name__ == '__main__':
    main()
