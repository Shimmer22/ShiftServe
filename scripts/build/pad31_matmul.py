#!/usr/bin/env python3
"""只读复用缓存NDK与原库，构建独立31/32矩阵补零探针。"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'results/20261111-pad31-matmul/build';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'src/a2o/native/profiling/pad31_matmul.cpp'
remote='/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261111-pad31-matmul'
base='/home/wins/ai-workspace/tools/llama.cpp-hexagon-build';image='ghcr.io/snapdragon-toolchain/arm64-android:v0.3'
subprocess.run(['ssh','wins','mkdir -p '+shlex.quote(remote)],check=True)
subprocess.run(['scp',str(source),'wins:'+remote+'/pad31_matmul.cpp'],check=True)
cc=['/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++','-std=c++17','-O2','-fPIC','-pie','-I/base/src/ggml/include','-I/base/src/vendor','/probe/pad31_matmul.cpp','-L/base/build-snapdragon/bin','-lggml','-lggml-base','-lggml-cpu','-ldl','-lm','-llog','-o','/probe/pad31_matmul']
command=['docker','run','--rm','--pull=never','--network=none','-v',base+':/base:ro','-v',remote+':/probe',image,*cc]
manifest=dict(command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),shared_modified=False)
with (out/'build.log').open('w') as log:subprocess.run(['ssh','wins',shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['scp','wins:'+remote+'/pad31_matmul',str(out/'probe')],check=True)
manifest['binary_sha256']=hashlib.sha256((out/'probe').read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out/'probe')
