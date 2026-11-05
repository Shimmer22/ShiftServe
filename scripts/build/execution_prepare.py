#!/usr/bin/env python3
"""生成并构建私有 host 准备路径；shared 和已安装库只读，不接触手机。开发稿未验证。"""
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/build'))
from kernel_probe import BASE, IMAGE, remote


def make_sources(source):
    paths=['src/llama-context.h','src/llama-context.cpp','src/llama-kv-cache.h','src/llama-kv-cache.cpp',
           'ggml/src/ggml-backend.cpp','ggml/src/ggml-hexagon/ggml-hexagon.cpp']
    original={p:(source/p).read_text() for p in paths};changed=dict(original)
    def replace(path,old,new):
        if changed[path].count(old)!=1:raise RuntimeError('source anchor mismatch: '+path+' '+old[:80])
        changed[path]=changed[path].replace(old,new)
    replace('src/llama-context.h','struct llama_context {',
        'struct a2o_execution_request;\nstruct a2o_execution_stats;\nstruct llama_context {\n'
        '    int a2o_execution_control(uint32_t,const a2o_execution_request *,a2o_execution_stats *);')
    replace('src/llama-context.cpp','//\n// llama_context\n//',
        '#include "a2o-llama-execution-context.inc"\n\n//\n// llama_context\n//')
    replace('src/llama-context.cpp','llama_context::~llama_context() {',
        'llama_context::~llama_context() {\n    a2o_context_forget(this);')
    replace('src/llama-context.cpp','    auto * res = gf_res_prev.get();\n    auto * gf  = res->get_gf();',
        '    const bool a2o_force_rebuild=a2o_execution_before_ubatch(this,ubatch);\n'
        '    auto * res = gf_res_prev.get();\n    auto * gf  = res->get_gf();')
    replace('src/llama-context.cpp','    if (!graph_reuse_disable && res->can_reuse(gparams)) {',
        '    const bool a2o_reuse=!a2o_force_rebuild && !graph_reuse_disable && res->can_reuse(gparams);\n'
        '    a2o_execution_graph_decision(this,a2o_reuse,gf->uid);\n    if (a2o_reuse) {')
    replace('src/llama-context.cpp',
        'size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {',
        'size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {\n'
        '    a2o_execution_stats a2o_stats{};\n'
        '    if(a2o_execution_control(A2O_EXEC_DISCARD,nullptr,&a2o_stats)) throw std::runtime_error("execution invalidation before state import failed");')
    replace('src/llama-kv-cache.h','    uint32_t get_n_kv(const slot_info & sinfo) const;',
        '    uint32_t get_n_kv(const slot_info & sinfo) const;\n    uint32_t a2o_preview_n_kv(const slot_info &) const;')
    replace('src/llama-kv-cache.h','    uint32_t get_n_kv() const;',
        '    uint32_t get_n_kv() const;\n    void a2o_preview_graph();')
    changed['src/llama-kv-cache.cpp']+='\n#include <stdexcept>\n#include "a2o-kv-execution-preview.inc"\n'
    changed['ggml/src/ggml-backend.cpp']+='\n#include "a2o-ggml-execution-prepare.inc"\n'
    host='ggml/src/ggml-hexagon/ggml-hexagon.cpp'
    replace(host,'static void ggml_backend_hexagon_free(ggml_backend_t backend) {',
        'static void a2o_forget_session(ggml_hexagon_session *);\nstatic void ggml_backend_hexagon_free(ggml_backend_t backend) {\n'
        '    a2o_forget_session(static_cast<ggml_hexagon_session*>(backend->context));')
    replace(host,'static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {',
        '#include "a2o-htp-execution-prepare.inc"\n\n'
        'static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {\n'
        '    const int a2o_prepared=a2o_compute_prepared(backend,graph);\n'
        '    if(a2o_prepared) return a2o_prepared>0?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;')
    replace(host,'bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, ggml_hexagon_opbatch* op_batch) {',
        'bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, ggml_hexagon_opbatch* op_batch, bool reset=true) {')
    replace(host,'        op_batch->reset();','        if(reset) op_batch->reset();')
    replace(host,'    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0 && opt_hostbuf) {',
        '    if (strcmp(name,"a2o_backend_execution_prepare_v1")==0) return (void*)a2o_backend_execution_prepare;\n'
        '    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0 && opt_hostbuf) {')
    assets={
        'src/a2o-llama-execution-context.inc':'backend/llama-execution-context.inc',
        'src/a2o-kv-execution-preview.inc':'state/kv-execution-preview.inc',
        'ggml/src/a2o-ggml-execution-prepare.inc':'scheduling/ggml-execution-prepare.inc',
        'ggml/src/ggml-hexagon/a2o-htp-execution-prepare.inc':'backend/htp-execution-prepare.inc'}
    data={p:(ROOT/'src/a2o/native'/name).read_bytes() for p,name in assets.items()}
    header=(ROOT/'src/a2o/native/backend/execution-preparation.h').read_bytes()
    for directory in ('src','ggml/src','ggml/src/ggml-hexagon'):data[directory+'/a2o-execution-preparation.h']=header
    return original,changed,data


# 复用原始编译参数、未修改对象和缓存 NDK。每个修改对象独立输出，不写 /base。
BUILD_HOST=r'''
import json,pathlib,shlex,subprocess
base=pathlib.Path('/base/build-snapdragon');probe=pathlib.Path('/probe')
cc=json.loads((base/'compile_commands.json').read_text());objects={}
(probe/'objects').mkdir(exist_ok=True);(probe/'lib').mkdir(exist_ok=True)
for suffix in ('src/llama-context.cpp','src/llama-kv-cache.cpp','ggml/src/ggml-backend.cpp','ggml/src/ggml-hexagon/ggml-hexagon.cpp'):
    candidates=[e for e in cc if e['file'].endswith('/'+suffix)]
    assert len(candidates)==1,suffix
    entry=candidates[0];args=shlex.split(entry['command']);old=args[args.index('-o')+1]
    clean=[];i=0
    while i<len(args):
        if args[i] in ('-MF','-MT','-MQ'):i+=2;continue
        if args[i] in ('-MD','-MMD','-MP'):i+=1;continue
        clean.append(args[i].replace('/workspace/src','/probe/src').replace('/workspace/build-snapdragon','/base/build-snapdragon'));i+=1
    output=str(probe/'objects'/(pathlib.Path(suffix).stem+'.o'));clean[clean.index('-o')+1]=output
    subprocess.run(clean,cwd=base,check=True);objects[old]=output
libraries={}
for name in ('ggml-base','ggml-hexagon','llama'):
    target='bin/lib'+name+'.so'
    commands=subprocess.check_output(['ninja','-C',str(base),'-t','commands',target],text=True).splitlines()
    links=[]
    for command in commands:
        a=shlex.split(command)
        if '-shared' in a and '-o' in a and pathlib.Path(a[a.index('-o')+1]).name.startswith('lib'+name+'.so'):links.append(a)
    assert len(links)==1,(target,len(links))
    args=links[0];original_output=args[args.index('-o')+1];start=next(i for i,s in enumerate(args) if 'clang++' in s or s.endswith('clang'))
    args=args[start:];end=args.index('&&') if '&&' in args else len(args);args=args[:end]
    for i,arg in enumerate(args):
        if arg in objects:args[i]=objects[arg]
        elif arg in libraries:args[i]=libraries[arg]
        elif not arg.startswith('-') and any(pathlib.Path(arg).name.startswith('lib'+dep+'.so') for dep in ('ggml-base','ggml-hexagon') if 'bin/lib'+dep+'.so' in libraries):
            dep=next(dep for dep in ('ggml-base','ggml-hexagon') if 'bin/lib'+dep+'.so' in libraries and pathlib.Path(arg).name.startswith('lib'+dep+'.so'))
            args[i]=libraries['bin/lib'+dep+'.so']
        else:args[i]=arg.replace('/workspace/build-snapdragon','/base/build-snapdragon')
    output=str(probe/'lib'/('lib'+name+'.so'));args[args.index('-o')+1]=output
    subprocess.run(args,cwd=base,check=True);libraries[target]=output;libraries[original_output]=output
'''


def main():
    source=ROOT.parent/'shared/runtime/llama.cpp';original,changed,assets=make_sources(source)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out=ROOT/'results'/(stamp+'-execution-prepare-build');out.mkdir(parents=True)
    target=ROOT/'build/execution-prepare'/stamp;target.mkdir(parents=True)
    scratch='/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/'+stamp+'-execution-prepare'
    patch=''.join(''.join(difflib.unified_diff(original[p].splitlines(True),changed[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p)) for p in original)
    (out/'prototype.patch').write_text(patch);(out/'build-host.py').write_text(BUILD_HOST)
    hashes={p:hashlib.sha256(s.encode()).hexdigest() for p,s in original.items()}
    setup=f'''import pathlib,hashlib,subprocess
base=pathlib.Path({BASE!r})/'src';scratch=pathlib.Path({scratch!r});expected={hashes!r}
for p,h in expected.items():assert hashlib.sha256((base/p).read_bytes()).hexdigest()==h,p
scratch.mkdir(parents=True)
subprocess.run(['cp','-as',str(base),str(scratch/'src')],check=True)
for p in expected:subprocess.run(['cp','--remove-destination',str(base/p),str(scratch/'src'/p)],check=True)
'''
    manifest={'source_sha256':hashes,'assets_sha256':{p:hashlib.sha256(b).hexdigest() for p,b in assets.items()},
        'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'remote_directory':scratch,'image':IMAGE,
        'scope':'private host-only libllama/ggml-base/hexagon; no DSP change; no graph compute during prepare',
        'shared_modified':False,'development_unvalidated':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
    try:
        remote(['python3','-'],input=setup,text=True)
        for i,(path,data) in enumerate(assets.items()):
            local=out/(str(i)+'-'+Path(path).name);local.write_bytes(data)
            subprocess.run(['scp','-q',str(local),'wins:'+scratch+'/src/'+path],check=True)
        for name in ('prototype.patch','build-host.py'):
            subprocess.run(['scp','-q',str(out/name),'wins:'+scratch+'/'+name],check=True)
        remote(['sh','-c','cd '+shlex.quote(scratch+'/src')+' && patch --batch -p1 -i ../prototype.patch'],capture_output=True)
        command=['docker','run','--rm','--pull=never','--network=none','--read-only','--cpus=4','--user','1000:1000','--tmpfs','/tmp:rw,exec,size=1g',
            '-v',BASE+':/base:ro','-v',BASE+':'+BASE+':ro','-v',scratch+':/probe:rw','--entrypoint','python3',IMAGE,'/probe/build-host.py']
        manifest['command']=command
        with (out/'build.log').open('w') as log:remote(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        manifest['binaries']={}
        for name in ('libllama.so','libggml-base.so','libggml-hexagon.so'):
            path=target/name;subprocess.run(['scp','-q','wins:'+scratch+'/lib/'+name,str(path)],check=True)
            manifest['binaries'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(target,flush=True)
    except Exception as error:
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (out/'failure.json').write_text(json.dumps({'error':repr(error)},indent=2));raise

if __name__=='__main__':main()
