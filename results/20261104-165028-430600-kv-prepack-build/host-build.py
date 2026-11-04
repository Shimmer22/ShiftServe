import json,shlex,subprocess,pathlib
cc=json.loads(pathlib.Path('/base/build-snapdragon/compile_commands.json').read_text())
e=next(e for e in cc if e['file'].endswith('/ggml-hexagon.cpp'))
a=shlex.split(e['command'])
a=[s.replace('/workspace/src','/probe/src').replace('/workspace/build-snapdragon','/base/build-snapdragon') for s in a]
a[a.index('-o')+1]='/probe/host.o'
subprocess.run(a,cwd='/base/build-snapdragon',check=True)
objs=list(pathlib.Path('/base/build-snapdragon/ggml/src/ggml-hexagon').rglob('htp-drv.cpp.o'))+list(pathlib.Path('/base/build-snapdragon/ggml/src/ggml-hexagon').rglob('htp_iface_stub.c.o'))
assert len(objs)==2,objs
subprocess.run([a[0],*[v for v in a if v.startswith(('--target=','--sysroot='))],'-static-libstdc++','-shared','-Wl,-soname,libggml-hexagon.so','-o','/probe/libggml-hexagon.so','/probe/host.o',*[str(p) for p in objs],'/base/build-snapdragon/bin/libggml-base.so','-llog','-ldl'],check=True)
