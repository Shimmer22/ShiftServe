#!/usr/bin/env python3
"""私有 host bridge + v79 DSP 历史 KV tiles 原型；shared 源码/库只读。"""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/build'))
from kernel_probe import remote, BASE, IMAGE

def main():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out=ROOT/'results'/(stamp+'-kv-prepack-build');out.mkdir()
    target=ROOT/'build/kv-prepack'/stamp;target.mkdir(parents=True)
    source=ROOT.parent/'shared/runtime/llama.cpp'
    prefix='ggml/src/ggml-hexagon/'
    names=['ggml-hexagon.cpp','htp/htp-ctx.h','htp/main.c','htp/hmx-matmul-ops.c']
    originals={prefix+n:(source/prefix/n).read_text() for n in names}
    changed=dict(originals)
    def replace(name,old,new):
        key=prefix+name
        assert changed[key].count(old)==1,(name,old)
        changed[key]=changed[key].replace(old,new)
    replace('htp/htp-ctx.h','struct htp_context {','struct htp_context {\n    struct a2o_kv_cache *a2o_cache;')
    replace('htp/main.c','AEEResult htp_iface_profiler(', '#include "a2o-kv-prepack-main.inc"\n\nAEEResult htp_iface_profiler(')
    replace('htp/main.c','    if (mode == HTP_PROF_PMU) {','    if (mode == A2O_PREPACK_MODE) return a2o_prepack_rpc(ctx,pmu_conf->events);\n\n    if (mode == HTP_PROF_PMU) {')
    replace('htp/main.c','    if (ctx->worker_pool) {','    a2o_cache_clear(ctx);\n\n    if (ctx->worker_pool) {')
    replace('ggml-hexagon.cpp','    std::atomic<int>      op_pending;','    std::vector<ggml_tensor*> a2o_kv_tensors;\n    std::atomic<int>      op_pending;')
    replace('ggml-hexagon.cpp','    if (tensor->view_src != NULL && tensor->view_offs == 0) {','    if (!tensor->view_src && (strncmp(tensor->name,"cache_k_l",9)==0 || strncmp(tensor->name,"cache_v_l",9)==0)) sess->a2o_kv_tensors.push_back(tensor);\n\n    if (tensor->view_src != NULL && tensor->view_offs == 0) {')
    replace('ggml-hexagon.cpp','static void * ggml_backend_hexagon_get_proc_address(', '#include "a2o-kv-prepack-host.inc"\n\nstatic void * ggml_backend_hexagon_get_proc_address(')
    replace('ggml-hexagon.cpp','    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0 && opt_hostbuf) {','    if (strcmp(name,"a2o_kv_control_v1")==0) return (void*)a2o_kv_control;\n    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0 && opt_hostbuf) {')
    replace('htp/hmx-matmul-ops.c','#include "vtcm-utils.h"','#include "vtcm-utils.h"\n#include "a2o-kv-prepack-load.inc"')
    # 仅改变 GQA batched F16 路径，其他算子保持原样。
    replace('htp/hmx-matmul-ops.c','''                {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);''','''                bool a2o_cached = ctx->a2o_cache != NULL;
                if (!a2o_cached) {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);''')
    replace('htp/hmx-matmul-ops.c','''                    TIMER_START(weight_load);
                    {
                        dma_queue_pop(ctx->dma[0]);''','''                    TIMER_START(weight_load);
                    if (a2o_cached) {
                        if (!a2o_load_kv_tiles(ctx,weight_group,params->k,params->weight_stride,nc,n_cols,vtcm_weight,buf_curr)) {
                            HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
                            return -1;
                        }
                    } else {
                        dma_queue_pop(ctx->dma[0]);''')
    patch=''.join(''.join(difflib.unified_diff(originals[p].splitlines(True),changed[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p)) for p in originals)
    (out/'prototype.patch').write_text(patch)
    assets={prefix+'htp/'+n:(ROOT/'src/a2o/native/state'/n).read_bytes() for n in ['htp-kv-prepack.h','htp-kv-prepack-main.inc','htp-kv-prepack-load.inc']}
    assets={p.replace('htp-kv-prepack','a2o-kv-prepack'):v for p,v in assets.items()}
    assets[prefix+'a2o-kv-prepack-host.inc']=(ROOT/'src/a2o/native/state/htp-kv-prepack-host.inc').read_bytes()
    expected={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in originals}
    scratch='/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/'+stamp+'-kv-prepack'
    setup=f'''import pathlib,hashlib,subprocess
base=pathlib.Path({BASE!r})/'src'; scratch=pathlib.Path({scratch!r})
expected={expected!r}
for p,s in expected.items(): assert hashlib.sha256((base/p).read_bytes()).hexdigest()==s,p
scratch.mkdir(parents=True)
subprocess.run(['cp','-as',str(base),str(scratch/'src')],check=True)
for p in expected: subprocess.run(['cp','--remove-destination',str(base/p),str(scratch/'src'/p)],check=True)
'''
    remote(['python3','-'],input=setup,text=True)
    for p,data in assets.items():
        local=out/Path(p).name;local.write_bytes(data)
        subprocess.run(['scp','-q',str(local),'wins:'+scratch+'/src/'+p],check=True)
    subprocess.run(['scp','-q',str(out/'prototype.patch'),'wins:'+scratch+'/prototype.patch'],check=True)
    remote(['sh','-c','cd '+shlex.quote(scratch+'/src')+' && patch --batch -p1 -i ../prototype.patch'],capture_output=True)
    sdk='/opt/hexagon/6.4.0.2';htools=sdk+'/tools/HEXAGON_Tools/19.0.04'
    host_script='''import json,shlex,subprocess,pathlib
cc=json.loads(pathlib.Path('/base/build-snapdragon/compile_commands.json').read_text())
e=next(e for e in cc if e['file'].endswith('/ggml-hexagon.cpp'))
a=shlex.split(e['command'])
a=[s.replace('/workspace/src','/probe/src').replace('/workspace/build-snapdragon','/base/build-snapdragon') for s in a]
a[a.index('-o')+1]='/probe/host.o'
subprocess.run(a,cwd='/base/build-snapdragon',check=True)
objs=list(pathlib.Path('/base/build-snapdragon/ggml/src/ggml-hexagon').rglob('htp-drv.cpp.o'))+list(pathlib.Path('/base/build-snapdragon/ggml/src/ggml-hexagon').rglob('htp_iface_stub.c.o'))
assert len(objs)==2,objs
subprocess.run([a[0],*[v for v in a if v.startswith(('--target=','--sysroot='))],'-static-libstdc++','-shared','-Wl,-soname,libggml-hexagon.so','-o','/probe/libggml-hexagon.so','/probe/host.o',*[str(p) for p in objs],'/base/build-snapdragon/bin/libggml-base.so','-llog','-ldl'],check=True)
'''
    (out/'host-build.py').write_text(host_script)
    subprocess.run(['scp','-q',str(out/'host-build.py'),'wins:'+scratch+'/host-build.py'],check=True)
    configure=['cmake','-S','/probe/src/ggml/src/ggml-hexagon/htp','-B','/probe/build-v79','-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_TOOLCHAIN_FILE=/probe/src/ggml/src/ggml-hexagon/htp/cmake-toolchain.cmake','-DHEXAGON_SDK_ROOT='+sdk,'-DHEXAGON_TOOLS_ROOT='+htools,'-DDSP_VERSION=v79','-DPREBUILT_LIB_DIR=toolv19_v79','-DHEXAGON_HTP_DEBUG=OFF','-DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128']
    shell='set -eu\n'+shlex.join(configure)+'\ncmake --build /probe/build-v79 -j 4\npython3 /probe/host-build.py'
    command=['docker','run','--rm','--pull=never','--network=none','--read-only','--cpus=4','--user','1000:1000','--tmpfs','/tmp:rw,exec,size=1g','-v',BASE+':/base:ro','-v',BASE+':'+BASE+':ro','-v',scratch+':/probe:rw','--entrypoint','/bin/sh',IMAGE,'-c',shell]
    manifest={'source_sha256':expected,'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'assets_sha256':{p:hashlib.sha256(v).hexdigest() for p,v in assets.items()},'remote_directory':scratch,'command':command,'scope':'private v79 + host bridge; profiler reserved mode; no wire ABI change; Qwen 36 layers, 2 KV heads, d128 only','shared_modified':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
    try:
        with (out/'build.log').open('w') as log: remote(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':str(e)}));raise
    manifest['binaries']={}
    for name,rel in [('libggml-hexagon.so','libggml-hexagon.so'),('libggml-htp-v79.so','build-v79/libggml-htp-v79.so')]:
        p=target/name;subprocess.run(['scp','-q','wins:'+scratch+'/'+rel,str(p)],check=True)
        manifest['binaries'][name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(target,flush=True)

if __name__=='__main__':main()
