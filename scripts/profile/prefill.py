#!/usr/bin/env python3
"""B 阶段实测矩阵；手机按 H 分进程串行运行，Gap 仅离线分析。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import build, default_device, native_source_hashes
from a2o.validation.contracts import check_response


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'configs/prefill-grid.json')
    p.add_argument('--device', default=default_device())
    p.add_argument('--histories', help='optional comma-separated subset')
    p.add_argument('--inputs', help='optional comma-separated subset')
    p.add_argument('--repetitions', type=int)
    p.add_argument('--timeout-s', type=int, help='per-history runtime timeout')
    p.add_argument('--direct-check', action='store_true', help='compare the original CPU context before any restore')
    p.add_argument('--tag', default='prefill-grid')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    for key, value in [('history_tokens', a.histories), ('new_tokens', a.inputs)]:
        if value:
            config[key] = list(map(int, value.split(',')))
    if a.repetitions is not None:
        config['repetitions'] = a.repetitions
    if a.timeout_s is not None:
        config['timeout_s_per_history'] = a.timeout_s
    binary = build()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / (stamp + '-' + a.tag)
    out.mkdir()
    source_hashes = native_source_hashes()
    paths = bool(config.get('paths'))
    manifest = {'config': config, 'device': a.device, 'source_files_sha256': source_hashes,
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'scope': 'four paths short-action calibration' if paths else 'phase B paired text microbenchmark; common GPU ViT excluded; ideal overlap simulated offline',
                'decode_backend': 'per_path' if paths else 'cpu', 'cpu_repack': False, 'kv_type': 'F16', 'llm_flash_attention': False,
                'direct_cpu_check': a.direct_check}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    adb = ['adb', '-s', a.device]
    remote = '/data/local/tmp/a2o_profile_' + stamp
    def call(args):
        subprocess.run(adb + args, check=True, stdout=subprocess.DEVNULL)
    call(['get-state'])
    call(['shell', 'mkdir', remote])
    call(['push', str(binary), remote + '/agent'])
    call(['shell', 'chmod', '755', remote + '/agent'])
    (out / 'empty.txt').write_text('')
    call(['push', str(out / 'empty.txt'), remote + '/empty.txt'])
    try:
        for h in config['history_tokens']:
            args = [remote + '/agent', '--model', '/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf',
                    '--mode', 'hybrid', '--policy', 'none', '--pre-move', '0', '--vision-backend', 'cpu',
                    '--batch-size', str(config['batch_size']), '--ctx-size', str(config['context_size']),
                    '--cpu-repack', '0', '--profile-history', str(h),
                    '--profile-inputs', ','.join(map(str, config['new_tokens'])),
                    '--profile-repeats', str(config['repetitions']), '--out', remote + f'/H{h}.jsonl']
            if a.direct_check:
                args += ['--profile-direct-check', '1']
            if paths:
                args += ['--path-profile', '1']
            for name in ['system', 'history', 'previous', 'prompt']:
                args += ['--' + name, remote + '/empty.txt']
            shell = ('cd /data/local/tmp/llama.cpp; '
                     'export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
                     f'GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_OPQUEUE={config["htp_queue"]} '
                     f'GGML_HEXAGON_OPBATCH={config["htp_opbatch"]}; '
                     f'echo $$ > {remote}/pid; exec ' + shlex.join(args))
            (out / f'H{h}.command.txt').write_text(shell + '\n')
            print(f'START H={h} output={out}', flush=True)
            try:
                with (out / f'H{h}.log').open('w') as log:
                    result = subprocess.run(adb + ['shell', shell], stdout=log, stderr=subprocess.STDOUT,
                                            timeout=config['timeout_s_per_history'])
                if result.returncode:
                    raise RuntimeError(f'H={h} failed with exit {result.returncode}')
                call(['pull', remote + f'/H{h}.jsonl', str(out / f'H{h}.jsonl')])
                call(['pull', remote + f'/H{h}.jsonl.prefix.txt', str(out / f'H{h}.prefix.txt')])
                for length in config['new_tokens']:
                    call(['pull', remote + f'/H{h}.jsonl.P{length}.suffix.txt', str(out / f'H{h}.P{length}.suffix.txt')])
            except subprocess.TimeoutExpired:
                stop = (f'p=$(cat {remote}/pid); '
                        f'if tr "\\0" " " < /proc/$p/cmdline | grep -F -q "{remote}/agent "; then kill -TERM "$p"; fi')
                subprocess.run(adb + ['shell', stop], check=False)
                raise
            print(f'COMPLETE H={h}', flush=True)
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps({'reason': repr(exc)}, indent=2) + '\n')
        subprocess.run(adb + ['pull', remote, str(out / 'failure-device')], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    finally:
        call(['shell', 'rm', '-rf', remote])
    rows = [json.loads(line) for path in sorted(out.glob('H*.jsonl')) for line in path.read_text().splitlines()]
    for row in rows:
        row['response_checks'] = ({row['path']: check_response(row, {'check': {'type': 'json_action', 'action': 'wait'}})} if paths else
                                 {backend: check_response(row[backend], {'check': {'type': 'json_action', 'action': 'wait'}})
                                  for backend in ['cpu', 'htp']})
        row['responses_passed'] = all(x['passed'] for x in row['response_checks'].values())
        if paths:
            row['responses_passed'] = row['responses_passed'] and row['eog']
    (out / 'checked-rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    passed = all(r['responses_passed'] and r['numerical_passed'] for r in rows)
    (out / 'completion.json').write_text(json.dumps({'trials' if paths else 'pairs': len(rows), 'passed': passed}, indent=2) + '\n')
    print(json.dumps({'run': str(out), 'trials' if paths else 'pairs': len(rows), 'passed': passed}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
