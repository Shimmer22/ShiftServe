#!/usr/bin/env python3
"""主机 wins ToolSandbox 与小米14 Pro 手机服务端联合测试入口：对比 Qwen3-8B CPU 基线与 Prefill-prior KV 裁剪。"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.workloads.toolsandbox import save


def run(cmd, log_path: Path):
    with log_path.open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def sync_workspace_to_wins(out: Path):
    remote_proj = 'wins:/home/wins/ai-workspace/projects/a2o-gap-profiling/'
    for rel in ['configs', 'scripts', 'src']:
        run(['rsync', '-rz', '--exclude=__pycache__', '--exclude=*.pyc', str(ROOT / rel) + '/', remote_proj + rel + '/'], out / f'sync-{rel}.log')


def phone_service(action: str, cfg: dict, cond_name: str, binary: Path, out: Path, registry: Path | None):
    folder = out / f'service-{cond_name}'
    cmd = [
        sys.executable, 'scripts/run/route_phone.py', action,
        '--condition', 'cpu',
        '--model', f"/data/data/com.termux/files/home/.local/a2o/models/{cfg['model']}",
        '--model-id', cfg['model_id'],
        '--model-sha256', cfg['model_sha256'],
        '--cpu-threads', str(cfg.get('cpu_threads', 4)),
        '--cpu-repack', '1' if cfg.get('cpu_repack', True) else '0',
        '--context-size', str(cfg['context_size']),
        '--output', str(folder),
        '--retry-network',
    ]
    if action == 'start':
        cmd += ['--binary', str(binary)]
        if registry:
            cmd += ['--registry', str(registry)]
        if cfg.get('kv_policy', 'none') != 'none':
            cmd += [
                '--kv-reduce', '1',
                '--kv-policy', str(cfg['kv_policy']),
                '--kv-keep-tokens', str(cfg.get('kv_keep_tokens', 512)),
                '--kv-sink-tokens', str(cfg.get('kv_sink_tokens', 4)),
                '--kv-recent-tokens', str(cfg.get('kv_recent_tokens', 64)),
                '--kv-pool-window', str(cfg.get('kv_pool_window', 5)),
                '--kv-tail-tokens', str(cfg.get('kv_tail_tokens', 16)),
            ]
    else:
        cmd += ['--logs-only']
    run(cmd, out / f'{cond_name}-service-{action}.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/20261114-qwen3-8b-kv-toolsandbox')
    parser.add_argument('--run-tag', default='20261114-qwen3-8b-kv')
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wins_out = '/home/wins/ai-workspace/outputs/a2o-gap-profiling'
    workspace = '/workspace/projects/a2o-gap-profiling'
    py = '/workspace/tools/a2o-toolsandbox-py311-env/bin/python'

    sync_workspace_to_wins(out)
    conditions = [
        ('baseline', ROOT / 'configs/toolsandbox-qwen3-8b-kv-baseline.json'),
        ('kv-reduce', ROOT / 'configs/toolsandbox-qwen3-8b-kv-reduce.json'),
    ]
    campaign_start = time.monotonic()
    comparisons = {}
    try:
        for cond_name, cfg_path in conditions:
            cfg = json.loads(cfg_path.read_text())
            cond_start = time.monotonic()
            save(out / 'status.json', dict(stage='start_service', condition=cond_name, elapsed_s=time.monotonic() - campaign_start))
            phone_service('start', cfg, cond_name, args.binary.resolve(), out, args.registry.resolve() if args.registry else None)
            try:
                val_name = f'{args.run_tag}-{cond_name}-validation'
                run_name = f'{args.run_tag}-{cond_name}'
                save(out / 'status.json', dict(stage='validate_service', condition=cond_name, elapsed_s=time.monotonic() - campaign_start))
                run(['ssh', 'wins', f'docker exec a2o-toolsandbox-py311 {py} {workspace}/scripts/validate/phone_server.py --config {workspace}/configs/{cfg_path.name} --output /workspace/outputs/a2o-gap-profiling/{val_name}'], out / f'{cond_name}-validation.log')
                save(out / 'status.json', dict(stage='collect_toolsandbox', condition=cond_name, elapsed_s=time.monotonic() - campaign_start))
                run(['ssh', 'wins', f'docker exec a2o-toolsandbox-py311 {py} {workspace}/scripts/collect/toolsandbox.py --config {workspace}/configs/{cfg_path.name} --output /workspace/outputs/a2o-gap-profiling/{run_name}'], out / f'{cond_name}-collect.log')
                run(['rsync', '-a', f'wins:{wins_out}/{val_name}/', str(out / f'{cond_name}-validation') + '/'], out / f'{cond_name}-val-download.log')
                run(['rsync', '-a', f'wins:{wins_out}/{run_name}/', str(out / cond_name) + '/'], out / f'{cond_name}-download.log')
                run([sys.executable, 'scripts/analyze/toolsandbox.py', str(out / cond_name)], out / f'{cond_name}-analysis.log')
                summary = json.loads((out / cond_name / 'summary.json').read_text())
                metrics = json.loads((out / cond_name / 'metrics-summary.json').read_text())
                comparisons[cond_name] = dict(
                    elapsed_s=time.monotonic() - cond_start,
                    completed=summary['completed'],
                    trace_complete=summary['trace_complete'],
                    full_score=summary['full_score'],
                    groups=metrics['groups'],
                )
                save(out / 'comparison-partial.json', comparisons)
            finally:
                save(out / 'status.json', dict(stage='stop_service', condition=cond_name, elapsed_s=time.monotonic() - campaign_start))
                phone_service('stop', cfg, cond_name, args.binary.resolve(), out, args.registry.resolve() if args.registry else None)

        save(out / 'comparison.json', comparisons)
        save(out / 'status.json', dict(stage='completed', elapsed_s=time.monotonic() - campaign_start))
        save(out / 'campaign-completion.json', dict(completed=True, conditions=list(comparisons.keys()), elapsed_s=time.monotonic() - campaign_start))
    except BaseException as exc:
        save(out / 'status.json', dict(stage='failed', error=repr(exc), elapsed_s=time.monotonic() - campaign_start))
        save(out / 'campaign-failure.json', dict(error=repr(exc), note='若手机服务未正常停止，锁及现场保留'))
        raise


if __name__ == '__main__':
    main()
