#!/usr/bin/env python3
"""只读复用缓存NDK和库，构建设备内补齐微基准。"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--e2e', action='store_true', help='构建完整模型P31/decode验收入口')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / ('src/a2o/native/profiling/htp_small_m_e2e.cpp' if args.e2e else 'src/a2o/native/profiling/htp_small_m.cpp')
    remote = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/' + out.parent.name + '-bench'
    base = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
    image = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.3'
    subprocess.run(['ssh', 'wins', 'mkdir -p ' + shlex.quote(remote)], check=True)
    subprocess.run(['scp', '-q', str(source), 'wins:' + remote + '/bench.cpp'], check=True)
    if args.e2e:
        subprocess.run(['rsync', '-a', str(ROOT / 'src/a2o/native') + '/', 'wins:' + remote + '/native/'], check=True)
    cc = ['/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++',
          '-std=c++17', '-O2', '-fPIC', '-pie', '-I/base/src/ggml/include', '-I/base/src/vendor', '/probe/bench.cpp',
          '-L/base/build-snapdragon/bin', '-lggml', '-lggml-base', '-lggml-cpu', '-ldl', '-lm', '-llog', '-o', '/probe/bench']
    if args.e2e:
        cc[cc.index('/probe/bench.cpp')] = '/probe/native/profiling/htp_small_m_e2e.cpp'
        cc += ['-I/base/src/include', '-I/base/src/tools/mtmd', '-I/probe/native', '-lllama', '-lmtmd']
    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '-v', base + ':/base:ro', '-v', remote + ':/probe', image, *cc]
    with (out / 'build.log').open('w') as log:
        subprocess.run(['ssh', 'wins', shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['scp', '-q', 'wins:' + remote + '/bench', str(out / 'probe')], check=True)
    manifest = dict(command=command, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256((out / 'probe').read_bytes()).hexdigest(), shared_modified=False)
    if args.e2e:
        manifest['native_source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'src/a2o/native').rglob('*')) if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
