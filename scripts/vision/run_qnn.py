#!/usr/bin/env python3
"""在手机 HTP 上运行 QAIRT 视觉 context，并取回 embedding。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import default_device
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.phone_preflight import require_phone_preflight


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=ROOT / 'results/qnn-vision-reference')
    p.add_argument('--runtime', type=Path, default=ROOT.parent / 'shared/runtime/qairt-2.51.1/android')
    p.add_argument('--device', default=default_device())
    p.add_argument('--tag', default='qnn-vision')
    a = p.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / (stamp + '-' + a.tag)
    out.mkdir()
    adb = ['adb', '-s', a.device]
    base = '/data/local/tmp/a2o_qnn'
    remote = base + '/run_' + stamp
    digest = hashlib.file_digest(a.context.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        h = hashlib.sha256()
        with a.context.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        digest = h.hexdigest()
    model = base + '/models/' + digest[:16] + '.bin'
    manifest = {'device': a.device, 'backend': 'QNN HTP', 'sdk': '2.51.1.261002',
                'context_sha256': digest, 'context': str(a.context.resolve()),
                'inputs': str(a.inputs.resolve()), 'scope': 'vision only; fixture preprocessing on CUDA host',
                'reference': json.loads((a.inputs / 'validation.json').read_text())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')

    def call(args):
        subprocess.run(adb + args, check=True, stdout=subprocess.DEVNULL)
    call(['shell', 'mkdir', '-p', base + '/lib', base + '/models', remote])
    for asset in a.runtime.iterdir():
        call(['push', str(asset), base + '/lib/' + asset.name])
    call(['shell', 'chmod', '755', base + '/lib/qnn-net-run'])
    if subprocess.run(adb + ['shell', 'test', '-f', model]).returncode:
        call(['push', str(a.context), model])
    names = ['pixel_values', 'position_ids_cos', 'position_ids_sin',
             'window_attention_mask', 'full_attention_mask']
    for name in names:
        call(['push', str(a.inputs / (name + '.raw')), remote + '/' + name + '.raw'])
    (out / 'input_list.txt').write_text(' '.join(name + ':=' + remote + '/' + name + '.raw' for name in names) + '\n')
    call(['push', str(out / 'input_list.txt'), remote + '/input_list.txt'])
    args = [base + '/lib/qnn-net-run', '--backend', base + '/lib/libQnnHtp.so',
            '--retrieve_context', model, '--input_list', remote + '/input_list.txt',
            '--output_dir', remote + '/output', '--profiling_level', 'basic', '--log_level', 'info']
    shell = ('export LD_LIBRARY_PATH=' + base + '/lib:/data/local/tmp/llama.cpp/lib; '
             'export ADSP_LIBRARY_PATH=' + base + '/lib; exec ' + shlex.join(args))
    (out / 'command.txt').write_text(shell + '\n')
    start = time.perf_counter()
    with (out / 'runtime.log').open('w') as log:
        require_phone_preflight(DeviceTransport(ROOT, 'adb', device=a.device), out)
        proc = subprocess.run(adb + ['shell', shell], stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start
    if proc.returncode:
        (out / 'failure.json').write_text(json.dumps({'returncode': proc.returncode}) + '\n')
        raise SystemExit('QNN failed; logs: ' + str(out))
    call(['pull', remote + '/output', str(out / 'output')])
    call(['shell', 'rm', '-rf', remote])
    import numpy as np
    encoded = np.fromfile(out / 'output/Result_0/image_features.raw', dtype=np.float32)
    reference = np.fromfile(a.inputs / 'adapted.f32', dtype=np.float32)
    delta = encoded - reference
    result = {'process_wall_s': wall, 'finite': bool(np.isfinite(encoded).all()),
              'output_floats': int(encoded.size),
              'cosine': float(np.dot(encoded, reference) / (np.linalg.norm(encoded) * np.linalg.norm(reference))),
              'nrmse': float(np.linalg.norm(delta) / np.linalg.norm(reference)),
              'max_abs': float(np.abs(delta).max())}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'run': str(out), **result}, indent=2), flush=True)


if __name__ == '__main__':
    main()
