#!/usr/bin/env python3
"""构建私有CPU逐节点计时库；只利用已有barrier，不增加同步。"""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/20261113-decode-attribution/build-fine'
REMOTE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261113-decode-attribution-build-fine'
BASE = '/home/wins/ai-workspace/tools/llama.cpp-hexagon-build'
IMAGE = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.3'

CPU_BUILD = r'''
import json,pathlib,shlex,subprocess
base=pathlib.Path('/base/build-snapdragon')
entries=json.loads((base/'compile_commands.json').read_text())
entry=next(e for e in entries if e['file'].endswith('/ggml-cpu.c'))
a=shlex.split(entry['command']);old=a[a.index('-o')+1];args=[];i=0
while i<len(a):
    if a[i] in ('-MF','-MT','-MQ'):i+=2;continue
    if a[i] in ('-MD','-MMD','-MP'):i+=1;continue
    s=a[i].replace('/workspace/src','/base/src').replace('/workspace/build-snapdragon','/base/build-snapdragon')
    if s.endswith('/ggml-cpu.c'):s='/probe/ggml-cpu.c'
    args.append(s);i+=1
args[args.index('-o')+1]='/probe/cpu.o';subprocess.run(args,cwd=base,check=True)
commands=subprocess.check_output(['ninja','-C',str(base),'-t','commands','bin/libggml-cpu.so'],text=True).splitlines()
link=next(shlex.split(s) for s in commands if '-shared' in shlex.split(s) and '-o' in shlex.split(s) and pathlib.Path(shlex.split(s)[shlex.split(s).index('-o')+1]).name=='libggml-cpu.so')
start=next(i for i,s in enumerate(link) if 'clang++' in s);link=link[start:]
if '&&' in link:link=link[:link.index('&&')]
link=[('/probe/cpu.o' if s==old else '/probe/libggml-base.so' if s.endswith('libggml-base.so') else s.replace('/workspace/build-snapdragon','/base/build-snapdragon')) for s in link]
link[link.index('-o')+1]='/probe/libggml-cpu.so';subprocess.run(link,cwd=base,check=True)
'''


def main():
    subprocess.run([sys.executable, str(ROOT/'scripts/build/decode_attribution.py'), '--followup', '--tag', 'build-fine'], check=True)
    source = ROOT.parent/'shared/runtime/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c'
    original = source.read_text()
    marker = 'static thread_ret_t ggml_graph_compute_thread(void * data) {'
    declarations = '''extern int a2o_cpu_node_enabled(void);
extern void a2o_cpu_node_record(const struct ggml_tensor *, int64_t, int);
'''
    changed = original.replace(marker, declarations+marker, 1)
    marker = '    for (int node_n = 0; node_n < cgraph->n_nodes &&'
    assert changed.count(marker) == 1
    changed = changed.replace(marker, '''    const bool a2o_timing = state->ith == 0 && a2o_cpu_node_enabled();
    struct ggml_tensor *a2o_last = NULL;
    int64_t a2o_last_start = 0;
    int a2o_last_fused = 0;
''' + marker, 1)
    marker = '        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);'
    assert changed.count(marker) == 1
    changed = changed.replace(marker, '        const int64_t a2o_start = a2o_timing ? ggml_time_us() : 0;\n'+marker, 1)
    marker = '''        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
'''
    assert changed.count(marker) == 1
    changed = changed.replace(marker, marker+'''        if (a2o_timing) {
            if (node_n + 1 < cgraph->n_nodes) a2o_cpu_node_record(node, ggml_time_us()-a2o_start, n_fused);
            else { a2o_last=node; a2o_last_start=a2o_start; a2o_last_fused=n_fused; }
        }
''', 1)
    marker = '''    ggml_barrier(state->threadpool);

    return 0;
}

#ifndef GGML_USE_OPENMP'''
    assert changed.count(marker) == 1
    changed = changed.replace(marker, '''    ggml_barrier(state->threadpool);
    if (a2o_timing && a2o_last) a2o_cpu_node_record(a2o_last, ggml_time_us()-a2o_last_start, a2o_last_fused);

    return 0;
}

#ifndef GGML_USE_OPENMP''', 1)
    (OUT/'ggml-cpu.c').write_text(changed)
    (OUT/'cpu-timing.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile='a/ggml-cpu.c', tofile='b/ggml-cpu.c')))
    (OUT/'cpu-build.py').write_text(CPU_BUILD)
    actual = subprocess.check_output(['ssh','wins','sha256sum '+BASE+'/src/ggml/src/ggml-cpu/ggml-cpu.c'], text=True).split()[0]
    assert actual == hashlib.sha256(original.encode()).hexdigest()
    for name in ['ggml-cpu.c','cpu-build.py']:
        subprocess.run(['scp', str(OUT/name), 'wins:'+REMOTE+'/'+name], check=True)
    command = ['docker','run','--rm','--pull=never','--network=none','-v',BASE+':/base:ro','-v',REMOTE+':/probe',IMAGE,'python3','/probe/cpu-build.py']
    with (OUT/'cpu-build.log').open('w') as log:
        subprocess.run(['ssh','wins',shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['scp','wins:'+REMOTE+'/libggml-cpu.so',str(OUT/'libggml-cpu.so')], check=True)
    manifest = json.loads((OUT/'manifest.json').read_text())
    manifest.update(original_cpu_sha256=actual, cpu_timing_command=command,
                    cpu_timing_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                    cpu_library_sha256=hashlib.sha256((OUT/'libggml-cpu.so').read_bytes()).hexdigest(),
                    cpu_timing_scope='thread0 through existing barriers; fused nodes timed as one group; no added barrier')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
