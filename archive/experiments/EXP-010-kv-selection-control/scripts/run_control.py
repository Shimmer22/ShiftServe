#!/usr/bin/env python3
"""Compare keep-all vs existing policy with identical EXP-009 runtime/prompt.

Deliberately preserves the raw, untemplated prompt to diagnose the original EOG.
Does not measure CPU/HTP switching or claim GUI quality or latency gains.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'EXP-009-real-kv-e2e'
OUT = ROOT / 'results'
DEVICE = os.environ.get('DEVICE', '10.29.11.213:42629')
REMOTE = '/data/local/tmp/a2o_exp010_control'
PROMPT = 'Next screenshot observation: Wi-Fi switch is on the screen. Reply with the next GUI action briefly.'
ADB = ['adb', '-s', DEVICE]


def call(args, log=None):
    start = time.perf_counter()
    p = subprocess.run(ADB + args, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=180)
    if log:
        (OUT / log).write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(p.stdout[-4000:])
    return (time.perf_counter() - start) * 1000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OLD / 'results/phone_pull/segments.json').read_text())
    (OUT / 'keep_all.txt').write_text('1\n' * meta['n_tokens_total'])
    results = {'device': DEVICE, 'backend': 'HTP0', 'prompt': PROMPT,
               'chat_template': False, 'repeats': 1, 'conditions': {}}
    call(['shell', f'mkdir {REMOTE}'])  # refuse reuse of any pre-existing directory
    try:
        call(['push', str(OLD / 'tools/build/kv_gui_e2e'), REMOTE + '/tool'])
        call(['shell', f'chmod 755 {REMOTE}/tool'])
        for name, mask in [('keep_all', OUT / 'keep_all.txt'),
                           ('policy', OLD / 'results/keep_mask.txt')]:
            remote_dir = REMOTE + '/' + name
            call(['shell', f'mkdir {remote_dir}'])
            call(['push', str(OLD / 'results/phone_pull/kv_state.bin'), remote_dir + '/kv_state.bin'])
            call(['push', str(mask), remote_dir + '/keep_mask.txt'])
            base = ('export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
                    'ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib '
                    'GGML_HEXAGON_EXPERIMENTAL=1; '
                    f'{REMOTE}/tool --dump-dir {remote_dir} '
                    '-m /data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf '
                    '--no-mmap --device HTP0 -ngl 99 -c 2048 -n 24 -t 4 ')
            apply_wall = call(['shell', base + f'--phase apply --keep-mask {remote_dir}/keep_mask.txt'], name + '_apply.log')
            prefill_wall = call(['shell', base + "--phase prefill --next-prompt '" + PROMPT + "'"], name + '_prefill.log')
            for filename in ['apply_stats.json', 'prefill_stats.json', 'generation.txt']:
                call(['pull', remote_dir + '/' + filename, str(OUT / (name + '_' + filename))])
            results['conditions'][name] = {
                'apply': json.loads((OUT / (name + '_apply_stats.json')).read_text()),
                'prefill': json.loads((OUT / (name + '_prefill_stats.json')).read_text()),
                'apply_process_wall_ms': apply_wall,
                'prefill_process_wall_ms': prefill_wall,
            }
        (OUT / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    finally:
        call(['shell', f'rm -rf {REMOTE}'])


if __name__ == '__main__':
    main()
