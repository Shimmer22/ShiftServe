#!/usr/bin/env python3
"""在私有v75 DSP中接入小M策略；复用既有打包补零/有效行写回，不改共享库。"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'shared/runtime/llama.cpp'
HTP = 'ggml/src/ggml-hexagon/htp/'
FILES = ['htp-ctx.h', 'main.c', 'matmul-ops.c', 'hmx-matmul-ops.c']


def make_patch(policy, profile):
    """以相同缓存源码生成独立干预，诊断标签按需编入。"""
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for name in FILES:
            target = work / HTP / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE / HTP / name, target)
        if profile:
            subprocess.run(['patch', '--batch', '-p1', '-i', str(ROOT / 'src/a2o/native/profiling/htp-layout-probe.patch')],
                           cwd=work, check=True, capture_output=True)
        target = work / HTP / 'matmul-ops.c'
        text = target.read_text()
        text = text.replace('#include "hmx-ops.h"', '#include "hmx-ops.h"\n#include "a2o-small-m-policy.h"', 1)
        old = '    if (m_hmx == 0) {'
        new = '''    // External M stays unchanged. HMX packs the partial tile with zeros
    // in VTCM and stores only valid rows; no extra host/RPC tensor is created.
    const int a2o_small_m = a2o_htp_small_m_allowed(m_total, (int)src0->ne[0],
            (int)src0->ne[1], wtype == HTP_TYPE_Q4_0, is_batched);
    if (m_hmx == 0 && !a2o_small_m) {'''
        assert text.count(old) == 1
        text = text.replace(old, new)
        # 新路径失败必须显式失败，不能递归尝试同一HMX分支。
        old = '    if (ret != 0) {\n'
        new = '''    if (ret != 0 && a2o_small_m) {
        FARF(ERROR, "A2O small-M HMX failed (m=%d k=%d n=%d ret=%d)", m_total, k, n, ret);
        return ret;
    }
    if (ret != 0) {
'''
        assert text.count(old) == 1
        target.write_text(text.replace(old, new))
        patch = ''
        for name in FILES:
            relative = HTP + name
            patch += ''.join(difflib.unified_diff((BASE / relative).read_text().splitlines(True),
                (work / relative).read_text().splitlines(True), fromfile='a/' + relative, tofile='b/' + relative))
        header = (ROOT / 'src/a2o/native/backend/htp_small_m_policy.h').read_text()
        header = '#define A2O_HTP_SMALL_M_POLICY ' + str(policy) + '\n' + header
        patch += ''.join(difflib.unified_diff([], header.splitlines(True), fromfile='/dev/null', tofile='b/' + HTP + 'a2o-small-m-policy.h'))
        return patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['original', 'force', 'guarded'], required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    patch = args.output.resolve() / 'small-m.patch'
    patch.write_text(make_patch({'original': 0, 'force': 1, 'guarded': 2}[args.policy], args.profile))
    subprocess.run([sys.executable, str(ROOT / 'scripts/build/kernel_probe.py'), '--patch', str(patch),
        '--dsp-version', 'v75', '--layout-probe', '--output', str(args.output / 'dsp')], check=True)
    # --layout-probe同时让旧构建器实体化hmx源文件；是否编译标签由本生成器决定。
    manifest_path = args.output / 'dsp/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['profiling_tags_compiled'] = args.profile
    manifest['small_m_policy'] = args.policy
    if not args.profile:
        manifest['probe_transport'] = 'none; compute variant without route instrumentation'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    completion = json.loads((args.output / 'dsp/completion.json').read_text())
    completion.update(policy=args.policy, diagnostic=args.profile, external_m_unchanged=True,
        header_sha256=hashlib.sha256((ROOT / 'src/a2o/native/backend/htp_small_m_policy.h').read_bytes()).hexdigest())
    (args.output / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')


if __name__ == '__main__':
    main()
