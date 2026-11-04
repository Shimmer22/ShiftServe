"""工程路径、共享工具链与原生构建。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT.parent / 'shared/runtime/llama.cpp-hexagon-pkg'
NATIVE = ROOT / 'src/a2o/native'


def default_device():
    return os.environ.get('DEVICE') or json.loads((ROOT / 'configs/device.json').read_text())['serial']


def build():
    ndk = sorted((Path.home() / 'Library/Android/sdk/ndk').iterdir(),
                 key=lambda p: tuple(map(int, p.name.split('.'))))[-1]
    cxx = next((ndk / 'toolchains/llvm/prebuilt').glob('*/bin/aarch64-linux-android28-clang++'))
    binary = ROOT / 'build/agent_runtime'
    binary.parent.mkdir(exist_ok=True)
    subprocess.run([str(cxx), '-std=c++17', '-O2', '-fPIC', '-pie',
                    '-I' + str(PKG / 'include'), '-I' + str(NATIVE), str(NATIVE / 'agent_runtime.cpp'),
                    '-L' + str(PKG / 'lib'), '-lllama', '-lmtmd', '-lggml', '-lggml-base',
                    '-llog', '-ldl', '-lm', '-Wl,-rpath,/data/local/tmp/llama.cpp/lib',
                    '-o', str(binary)], check=True)
    return binary


def native_source_hashes():
    """递归记录全部原生机制，避免拆分目录后漏记源码。"""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(NATIVE.rglob('*')) if p.suffix in {'.h', '.cpp', '.inc'}}
