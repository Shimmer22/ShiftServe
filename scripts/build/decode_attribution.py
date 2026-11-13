#!/usr/bin/env python3
"""构建短decode入口与只计时的私有ggml-base；不改shared算法或库。"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/20261113-decode-attribution/build'
BASE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
REMOTE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261113-decode-attribution'
IMAGE = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.3'

HOST_BUILD = r'''
import json,pathlib,shlex,subprocess
base=pathlib.Path('/base/build-snapdragon')
entries=json.loads((base/'compile_commands.json').read_text())
entry=next(e for e in entries if e['file'].endswith('/ggml-backend.cpp'))
a=shlex.split(entry['command']);old=a[a.index('-o')+1];args=[];i=0
while i<len(a):
    if a[i] in ('-MF','-MT','-MQ'):i+=2;continue
    if a[i] in ('-MD','-MMD','-MP'):i+=1;continue
    s=a[i].replace('/workspace/src','/base/src').replace('/workspace/build-snapdragon','/base/build-snapdragon')
    if s.endswith('/ggml-backend.cpp'):s='/probe/ggml-backend.cpp'
    args.append(s);i+=1
args[args.index('-o')+1]='/probe/backend.o';args.append('-I/probe')
subprocess.run(args,cwd=base,check=True)
commands=subprocess.check_output(['ninja','-C',str(base),'-t','commands','bin/libggml-base.so'],text=True).splitlines()
link=next(shlex.split(s) for s in commands if '-shared' in shlex.split(s) and '-o' in shlex.split(s) and pathlib.Path(shlex.split(s)[shlex.split(s).index('-o')+1]).name=='libggml-base.so')
start=next(i for i,s in enumerate(link) if 'clang++' in s);link=link[start:]
if '&&' in link:link=link[:link.index('&&')]
link=[('/probe/backend.o' if s==old else s.replace('/workspace/build-snapdragon','/base/build-snapdragon')) for s in link]
link[link.index('-o')+1]='/probe/libggml-base.so';subprocess.run(link,cwd=base,check=True)
cc=['/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++','-std=c++17','-O2','-fPIC','-pie','-I/base/src/include','-I/base/src/ggml/include','-I/base/src/src','-I/base/src/ggml/src','-I/base/src/tools/mtmd','-I/base/src/vendor','-I/probe/native','/probe/native/profiling/decode_attribution.cpp','-L/probe','-L/base/build-snapdragon/bin','-lllama','-lmtmd','-lggml','-lggml-base','-llog','-ldl','-lm','-o','/probe/probe']
subprocess.run(cc,check=True)
if pathlib.Path('/probe/llama-context.cpp').exists():
    entry=next(e for e in entries if e['file'].endswith('/src/llama-context.cpp'))
    a=shlex.split(entry['command']);old=a[a.index('-o')+1];args=[];i=0
    while i<len(a):
        if a[i] in ('-MF','-MT','-MQ'):i+=2;continue
        if a[i] in ('-MD','-MMD','-MP'):i+=1;continue
        s=a[i].replace('/workspace/src','/base/src').replace('/workspace/build-snapdragon','/base/build-snapdragon')
        if s.endswith('/llama-context.cpp'):s='/probe/llama-context.cpp'
        args.append(s);i+=1
    args[args.index('-o')+1]='/probe/context.o';subprocess.run(args,cwd=base,check=True)
    commands=subprocess.check_output(['ninja','-C',str(base),'-t','commands','bin/libllama.so'],text=True).splitlines()
    link=next(shlex.split(s) for s in commands if '-shared' in shlex.split(s) and '-o' in shlex.split(s) and pathlib.Path(shlex.split(s)[shlex.split(s).index('-o')+1]).name=='libllama.so')
    start=next(i for i,s in enumerate(link) if 'clang++' in s);link=link[start:]
    if '&&' in link:link=link[:link.index('&&')]
    link=[('/probe/context.o' if s==old else s.replace('/workspace/build-snapdragon','/base/build-snapdragon')) for s in link]
    link[link.index('-o')+1]='/probe/libllama.so';subprocess.run(link,cwd=base,check=True)
'''


def main():
    global OUT, REMOTE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--followup',action='store_true');parser.add_argument('--tag');args=parser.parse_args()
    if args.tag:
        assert all(c.isalnum() or c=='-' for c in args.tag)
        OUT=OUT.parent/args.tag;REMOTE+='-'+args.tag
    elif args.followup:OUT=OUT.parent/'build-followup';REMOTE+='-followup'
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT.parent / 'shared/runtime/llama.cpp/ggml/src/ggml-backend.cpp'
    original = source.read_text()
    changed = original.replace('#include <vector>', '#include <vector>\n#include "decode_timing.h"', 1)
    replacements = {
        'enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {':
            '\n    a2o_attr_compute a2o_timing(backend,cgraph);',
        'void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {':
            '\n    a2o_attr_copy a2o_timing(ggml_nbytes(src));',
        'void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {':
            '\n    a2o_attr_copy a2o_timing(ggml_nbytes(src));',
        'void ggml_backend_synchronize(ggml_backend_t backend) {':
            '\n    a2o_attr_scope a2o_timing(std::string("SYNC/")+ggml_backend_name(backend),a2o_attr_copy_depth==0&&a2o_attr_compute_depth==0);',
    }
    for signature, addition in replacements.items():
        assert changed.count(signature) == 1
        changed = changed.replace(signature, signature + addition)
    (OUT / 'ggml-backend.cpp').write_text(changed)
    (OUT / 'timing.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile='a/ggml-backend.cpp', tofile='b/ggml-backend.cpp')))
    (OUT / 'host-build.py').write_text(HOST_BUILD)
    subprocess.run(['ssh', 'wins', 'mkdir -p ' + shlex.quote(REMOTE + '/native')], check=True)
    subprocess.run(['rsync', '-a', str(ROOT / 'src/a2o/native') + '/', 'wins:' + REMOTE + '/native/'], check=True)
    for p in [OUT / 'ggml-backend.cpp', OUT / 'host-build.py', ROOT / 'src/a2o/native/profiling/decode_timing.h']:
        subprocess.run(['scp', str(p), 'wins:' + REMOTE + '/' + p.name], check=True)
    if args.followup:
        context=(ROOT.parent/'shared/runtime/llama.cpp/src/llama-context.cpp').read_text()
        policy=ROOT/'src/a2o/native/backend/cpu-fused-gdn.patch'
        (OUT/'llama-context.cpp').write_text(context)
        with (OUT/'context-patch.log').open('w') as log:
            subprocess.run(['patch','-p1','-i',str(policy.resolve())],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,check=True)
        (OUT/'cpu-fused-gdn.patch').write_text(policy.read_text())
        subprocess.run(['scp',str(OUT/'llama-context.cpp'),'wins:'+REMOTE+'/llama-context.cpp'],check=True)
        actual_context=subprocess.check_output(['ssh','wins','sha256sum '+BASE+'/src/src/llama-context.cpp'],text=True).split()[0]
        assert actual_context==hashlib.sha256(context.encode()).hexdigest()
    actual = subprocess.check_output(['ssh', 'wins', 'sha256sum ' + BASE + '/src/ggml/src/ggml-backend.cpp'], text=True).split()[0]
    assert actual == hashlib.sha256(original.encode()).hexdigest(), 'remote/shared source mismatch'
    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '-v', BASE + ':/base:ro', '-v', REMOTE + ':/probe', IMAGE, 'python3', '/probe/host-build.py']
    manifest = dict(command=command, shared_modified=False, original_backend_sha256=actual,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'src/a2o/native').rglob('*') if p.is_file()})
    if args.followup:manifest['original_context_sha256']=actual_context
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (OUT / 'build.log').open('w') as log:
        subprocess.run(['ssh', 'wins', shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ['probe', 'libggml-base.so']+(['libllama.so'] if args.followup else []):
        subprocess.run(['scp', 'wins:' + REMOTE + '/' + name, str(OUT / name)], check=True)
        manifest[name + '_sha256'] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
