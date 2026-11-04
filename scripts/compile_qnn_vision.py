#!/usr/bin/env python3
"""在 Linux QAIRT 环境将视觉 ONNX 转换并编译为 SM8750 HTP context。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='export_qnn_vision.py output directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dlc', type=Path, help='reuse an already converted DLC')
    p.add_argument('--opt-level', type=int, choices=[0, 1, 2, 3], default=1)
    a = p.parse_args()
    a.sdk, a.input, a.output = a.sdk.resolve(), a.input.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    tools = a.sdk / 'bin/x86_64-linux-clang'
    libs = a.sdk / 'lib/x86_64-linux-clang'
    env = dict(os.environ, PYTHONPATH=str(a.sdk / 'lib/python'),
               LD_LIBRARY_PATH=str(libs), QNN_SDK_ROOT=str(a.sdk), QAIRT_SDK_ROOT=str(a.sdk))
    env['PATH'] = str(Path(sys.executable).parent) + ':' + str(tools) + ':' + env['PATH']
    record = {'sdk': str(a.sdk), 'input': str(a.input), 'target': 'SM8750 / v79',
              'float_bitwidth': 16, 'optimization_level': a.opt_level, 'commands': []}

    def run(name, cmd):
        record['commands'].append([str(x) for x in cmd])
        (a.output / 'commands.json').write_text(json.dumps(record, indent=2) + '\n')
        print(name, flush=True)
        start = time.perf_counter()
        with (a.output / (name + '.log')).open('w') as log:
            subprocess.run([str(x) for x in cmd], env=env, check=True, stdout=log,
                           stderr=subprocess.STDOUT, cwd=a.output)
        record[name + '_wall_s'] = time.perf_counter() - start

    dlc = a.dlc.resolve() if a.dlc else a.output / 'vision_encoder.dlc'
    if not a.dlc:
        run('convert', [sys.executable, tools / 'qairt-converter',
                        '--input_network', a.input / 'vision_encoder.onnx',
                        '--output_path', dlc, '--float_bitwidth', '16'])
    backend = {'devices': [{'dsp_arch': 'v79', 'soc_model': 69}]}
    (a.output / 'htp.json').write_text(json.dumps(backend, indent=2) + '\n')
    ext = {'backend_extensions': {'shared_library_path': str(libs / 'libQnnHtpNetRunExtensions.so'),
                                  'config_file_path': str(a.output / 'htp.json')}}
    (a.output / 'extensions.json').write_text(json.dumps(ext, indent=2) + '\n')
    run('context', [tools / 'qnn-context-binary-generator', '--backend', libs / 'libQnnHtp.so',
                    '--model', libs / 'libQnnModelDlc.so', '--dlc_path', dlc,
                    '--binary_file', 'vision_encoder', '--output_dir', a.output / 'context',
                    '--config_file', a.output / 'extensions.json',
                    '--optimization_level_override', str(a.opt_level)])
    (a.output / 'commands.json').write_text(json.dumps(record, indent=2) + '\n')
    print(a.output / 'context/vision_encoder.bin', flush=True)


if __name__ == '__main__':
    main()
