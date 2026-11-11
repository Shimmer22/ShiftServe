#!/usr/bin/env python3
"""私有 registry 只注册 CPU/HTP，复用原始对象；不初始化 OpenCL。"""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/20261107-toolsandbox-route-e2e/build-cpu-htp-registry';out.mkdir(exist_ok=False)
target='/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261107-qwen35-route-preflight';base='/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
script=r'''
import json,shlex,subprocess,pathlib
base=pathlib.Path('/base/build-snapdragon');probe=pathlib.Path('/probe');cc=json.loads((base/'compile_commands.json').read_text())
e=next(e for e in cc if e['file'].endswith('/ggml-backend-reg.cpp'))
a=shlex.split(e['command']);a.remove('-DGGML_USE_OPENCL');a=[x.replace('/workspace/src','/base/src').replace('/workspace/build-snapdragon','/base/build-snapdragon') for x in a];a[a.index('-o')+1]='/probe/registry.o';subprocess.run(a,cwd=base,check=True)
commands=subprocess.check_output(['ninja','-C',str(base),'-t','commands','bin/libggml.so'],text=True).splitlines()
a=next(shlex.split(s) for s in commands if '-shared' in s and '-o bin/libggml.so' in s);a=a[next(i for i,s in enumerate(a) if 'clang++' in s):];a=a[:a.index('&&')] if '&&' in a else a
for i,x in enumerate(a):
 if x.endswith('/ggml-backend-reg.cpp.o'):a[i]='/probe/registry.o'
 else:a[i]=x.replace('/workspace/src','/base/src').replace('/workspace/build-snapdragon','/base/build-snapdragon')
a[a.index('-o')+1]='/probe/libggml.so';a=[x for x in a if 'libggml-opencl.so' not in x];subprocess.run(a,cwd=base,check=True)
'''
(out/'build.py').write_text(script);subprocess.run(['scp',str(out/'build.py'),'wins:'+target+'/registry-build.py'],check=True)
cmd=['docker','run','--rm','-v',base+':/base:ro','-v',target+':/probe','ghcr.io/snapdragon-toolchain/arm64-android:v0.3','python3','/probe/registry-build.py']
with (out/'build.log').open('w') as f:subprocess.run(['ssh','wins',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['scp','wins:'+target+'/libggml.so',str(out/'libggml.so')],check=True)
print(out)
