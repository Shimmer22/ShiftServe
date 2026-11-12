#!/usr/bin/env python3
"""复用 wins 缓存工具链，只构建显式指定架构的私有 DSP 路由探针，不改共享安装库。"""
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.3'
BASE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
FILES = ['ggml/src/ggml-hexagon/htp/' + n for n in ['htp-ctx.h', 'main.c', 'matmul-ops.c']]


def remote(command, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', 'wins', shlex.join(command)], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--layout-probe', action='store_true')
    parser.add_argument('--dsp-version', choices=['v75', 'v79'], default='v79')
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = args.output or ROOT / 'results' / (stamp + '-htp-route-probe-build'); out.mkdir(parents=True)
    scratch = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/' + stamp
    patch = args.patch or ROOT / 'src/a2o/native/profiling/htp-route-probe.patch'
    local_source = ROOT.parent / 'shared/runtime/llama.cpp'
    files = FILES + (['ggml/src/ggml-hexagon/htp/hmx-matmul-ops.c'] if args.layout_probe else [])
    expected = {p: hashlib.sha256((local_source / p).read_bytes()).hexdigest() for p in files}
    setup = f'''import pathlib,hashlib,json,subprocess
base=pathlib.Path({BASE!r})/'src'
scratch=pathlib.Path({scratch!r})
expected={expected!r}
for p,sha in expected.items():
    assert hashlib.sha256((base/p).read_bytes()).hexdigest()==sha, 'remote/local source mismatch: '+p
scratch.mkdir(parents=True)
subprocess.run(['cp','-as',str(base),str(scratch/'src')],check=True)
for p in expected:
    subprocess.run(['cp','--remove-destination',str(base/p),str(scratch/'src'/p)],check=True)
(scratch/'source-hashes.json').write_text(json.dumps(expected,indent=2))
'''
    remote(['python3', '-'], input=setup, text=True)
    subprocess.run(['scp', '-q', str(patch), 'wins:' + scratch + '/probe.patch'], check=True)
    remote(['sh', '-c', 'cd ' + shlex.quote(scratch + '/src') + ' && patch --batch --forward -p1 -i ../probe.patch'], stdout=subprocess.PIPE, text=True)
    sdk = '/opt/hexagon/6.4.0.2'; tools = sdk + '/tools/HEXAGON_Tools/19.0.04'
    build_dir = '/probe/build-' + args.dsp_version
    library = 'libggml-htp-' + args.dsp_version + '.so'
    configure = ['cmake', '-S', '/probe/src/ggml/src/ggml-hexagon/htp', '-B', build_dir, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_TOOLCHAIN_FILE=/probe/src/ggml/src/ggml-hexagon/htp/cmake-toolchain.cmake',
        '-DHEXAGON_SDK_ROOT=' + sdk, '-DHEXAGON_TOOLS_ROOT=' + tools, '-DDSP_VERSION=' + args.dsp_version,
        '-DPREBUILT_LIB_DIR=toolv19_' + args.dsp_version, '-DHEXAGON_HTP_DEBUG=OFF', '-DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128',
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    shell = 'set -eu\n' + shlex.join(configure) + '\n' + shlex.join(['cmake', '--build', build_dir, '-j', '4'])
    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only', '--cpus=4', '--user', '1000:1000',
        '--tmpfs', '/tmp:rw,exec,size=1g', '-v', BASE + ':' + BASE + ':ro', '-v', scratch + ':/probe:rw',
        '--entrypoint', '/bin/sh', IMAGE, '-c', shell]
    manifest = {'remote_host': 'wins', 'remote_directory': scratch, 'image': IMAGE, 'source_files_sha256': expected,
        'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(), 'command': command,
        'probe_transport': 'a2o-route-v1: eight profiling PMU slots carry route metadata, NOT hardware PMU values',
        'wire_abi_changed': False, 'shared_source_or_install_modified': False, 'dsp_variant': args.dsp_version}
    if args.layout_probe:
        manifest['probe_transport'] = 'a2o-layout-v2: PMU slots carry route,M,N,K,type,mc,nc,layout; NOT hardware PMU'
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out / 'probe.patch').write_bytes(patch.read_bytes())
    print(str(out), flush=True)
    try:
        with (out / 'build.log').open('w') as log:
            remote(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps({'error': str(exc)}, indent=2)); raise
    target = ROOT / 'build/htp-route-probe' / stamp; target.mkdir(parents=True)
    binary = target / library
    subprocess.run(['scp', '-q', 'wins:' + scratch + '/build-' + args.dsp_version + '/' + library, str(binary)], check=True)
    subprocess.run(['scp', '-q', 'wins:' + scratch + '/build-' + args.dsp_version + '/CMakeCache.txt', str(out / 'CMakeCache.txt')], check=True)
    subprocess.run(['scp', '-q', 'wins:' + scratch + '/build-' + args.dsp_version + '/compile_commands.json', str(out / 'compile_commands.json')], check=True)
    manifest.update(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out / 'completion.json').write_text(json.dumps({'build_passed': True, 'binary': str(binary), 'sha256': manifest['binary_sha256']}, indent=2))
    print(json.dumps({'build': str(out), 'binary': str(binary)}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
