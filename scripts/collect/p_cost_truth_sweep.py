#!/usr/bin/env python3
"""在同一前缀、工具返回时刻和冻结 Decode 上对照 CPU/HTP 路径。"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.collect.tau2_e2e_worker import _http_post

BASE = os.environ.get('A2O_PHONE_HTTP_BASE', 'http://10.29.131.50:18780').rstrip('/')
LAUNCHER = ROOT / 'scripts/run/route_phone.py'
DEFAULT_P_POINTS = list(range(16, 65)) + [66, 68, 72, 80, 96, 128]
PHASES = ('prepare_ms', 'wait_ms', 'restore_ms', 'prefill_ms', 'compact_ms', 'handoff_ms', 'decode_ms')


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def load_prompt_slice(fixtures_path, tokens):
    fixtures = read(fixtures_path)
    if 'prefix_ids' in fixtures and len(fixtures['prefix_ids']) >= tokens:
        return dict(case_id=fixtures_path.stem, turn_index=None, ids=fixtures['prefix_ids'][:tokens])
    for case in fixtures.get('cases', []):
        for turn in case['turns']:
            ids = turn['request']['frozen_prompt_ids']
            if len(ids) >= tokens:
                return dict(case_id=case['id'], turn_index=turn['index'], ids=ids[:tokens])
    raise ValueError(f'夹具中没有至少 {tokens} token 的输入')


def build_identity(build):
    manifest = read(build / 'manifest.json')
    for name in ('agent', 'libggml-cpu.so', 'libllama.so'):
        if sha256(build / name) != manifest['artifacts'][name]:
            raise ValueError(f'构建产物校验失败: {name}')
    return {key: manifest[key] for key in ('build_id', 'git_commit', 'artifacts', 'hexagon_sha256')}


def ensure_plan(output, plan):
    path = output / 'plan.json'
    if path.exists():
        if read(path) != plan:
            raise ValueError('续跑的构建、模型、夹具或参数不同，请使用新的结果目录')
    elif any(item.name != 'driver.lock' for item in output.iterdir()):
        raise ValueError('结果目录已有内容但缺少采集计划，请使用新的结果目录')
    else:
        save(path, plan)
    return sha256(path)


def sample_path(output, mode, p, rep):
    return output / 'raw' / mode / f'p{p:03d}-r{rep:02d}.json'


def read_sample(path, plan_sha, mode, h, p, d, rep):
    sample = read(path)
    expected = dict(plan_sha256=plan_sha, mode=mode, H=h, P=p, D=d, rep=rep)
    if any(sample.get(key) != value for key, value in expected.items()):
        raise ValueError(f'样本与采集计划不符: {path}')
    return sample['row']


def service(action, mode, args, directory, plan):
    command = [sys.executable, str(LAUNCHER), action, '--condition', 'route',
               '--output', str(directory), '--logs-only', '--retry-network']
    shared = mode == 'shared'
    if action == 'start':
        m = plan['model']
        command += [
            '--runtime-overlay', str(args.build), '--model', m['model_path'],
            '--model-id', m['model_id'], '--model-sha256', m['model_sha256'],
            '--context-size', str(plan['context_size']), '--cpu-repack', '0',
            '--kv-offload', '0', '--verify-kv-state', str(args.verify_kv_state),
            '--kv-hwm-trim', str(args.kv_hwm_trim), '--device', args.device, '--gap-ms', str(args.gap_ms),
        ]
        if shared:
            command += ['--shared-weights']
        if args.force_hc_compact:
            command += [
                '--kv-reduce', '1', '--kv-policy', 'prefill_prior',
                '--kv-keep-tokens', str(args.keep_tokens), '--kv-recent-tokens', '32',
                '--kv-tail-tokens', '16', '--compact-anchor', str(args.compact_anchor),
                '--kv-layer-stride', str(args.kv_layer_stride),
                '--kv-min-evict-tokens', str(args.kv_min_evict_tokens),
            ]
    subprocess.run(command, cwd=ROOT, check=True)
    if action == 'start':
        health = read(directory / 'health.json')
        launch = read(directory / 'manifest.json')
        if (launch['runtime_overlay'] != plan['build']['artifacts'] or
                launch['hexagon_sha256'] != plan['build']['hexagon_sha256'] or
                health.get('build_id') != plan['build']['build_id'] or
                health.get('git_commit') != plan['build']['git_commit'] or
                health.get('shared_weights') is not shared or
                health.get('model') != plan['model']['model_id'] or
                health.get('verify_kv_state') is not bool(args.verify_kv_state) or
                health.get('kv_hwm_trim') is not bool(args.kv_hwm_trim) or
                health.get('kv_reduce') is not args.force_hc_compact or
                (not args.verify_kv_state and health.get('paired_truth_version', 1) < 2)):
            raise RuntimeError(f'手机实际配置与计划不符: {directory}')


def validate_truth(row, paths, decode_ids, gap_ms, verify_kv_state=True):
    truth = row['paired_truth']
    version = truth['version']
    if version not in (1, 2) or (truth['D'], truth['gap_ms']) != (len(decode_ids), gap_ms):
        raise ValueError('设备返回的三路计时协议或形状不符')
    if version < 2 and not verify_kv_state:
        raise ValueError('轻量 KV 核验需要提供三路交接字节数的计时协议')
    if not math.isfinite(row['cosine']) or row['cosine'] < 0.99:
        raise ValueError('CPU/HTP Prefill 数值校验失败')
    if 'HC-Compact' in paths and (row['hc_compact_evicted_count'] <= 0 or row['hc_compact_handoff_bytes'] <= 0):
        raise ValueError('HC-Compact 缺少裁剪或状态迁移')
    if not row.get('passed'):
        raise ValueError('设备未通过 CPU/HTP 配对校验')
    for path in paths:
        measured = truth[path]
        if version >= 2:
            handoff = measured['handoff_bytes']
            if type(handoff) is not int or (handoff != 0 if path == 'CC' else handoff <= 0):
                raise ValueError(f'{path} 的实际交接字节数不符')
        if measured['decode_ids'] != decode_ids:
            raise ValueError(f'{path} 的冻结 Decode token 不一致')
        if any(not math.isfinite(measured[k]) or measured[k] < 0 for k in (*PHASES, 'post_return_ms')):
            raise ValueError(f'{path} 的阶段耗时无效')
        charged = sum(measured[k] for k in PHASES if k != 'prepare_ms')
        if measured['post_return_ms'] + 0.1 < charged:
            raise ValueError(f'{path} 的返回后总耗时小于各阶段之和')


def calibrate(args, prefix, input_ids, decode_ids, repetitions, raw_reply):
    result = _http_post(BASE + '/a2o/calibrate', dict(
        prefix_ids=prefix, input_ids=input_ids, truth_decode_ids=decode_ids,
        truth_gap_ms=args.gap_ms, repetitions=repetitions, include_hc_compact=args.force_hc_compact,
        keep_tokens=args.keep_tokens,
    ), timeout=1800 if args.batch_reps else 600)
    expected = (len(prefix), len(input_ids), len(decode_ids))
    if ((result['H'], result['P'], result['D']) != expected
            or [row['repetition'] for row in result['rows']] != list(range(repetitions))):
        save(raw_reply, result)
        raise ValueError('设备返回的 H/P/D 或重复次数不符')
    return result['rows']


def measurement_batches(p_points, reps, seed, batch_reps):
    rng = random.Random(seed)
    if batch_reps or seed is None:
        order = p_points if seed is None else rng.sample(p_points, len(p_points))
        return [[(p, rep) for rep in range(reps)] for p in order] if batch_reps else [
            [(p, rep)] for p in order for rep in range(reps)
        ]
    return [[(p, rep)] for rep in range(reps) for p in rng.sample(p_points, len(p_points))]


def trimmed_mean(values):
    values = sorted(values)
    return statistics.mean(values[1:-1] if len(values) >= 4 else values)


def summarize_mode(output, plan, plan_sha, mode):
    rows = []
    for p in plan['p_points']:
        samples = [
            read_sample(path, plan_sha, mode, plan['H'], p, plan['D'], rep)
            for rep in range(plan['reps'])
            if (path := sample_path(output, mode, p, rep)).exists()
        ]
        if not samples:
            continue
        row = dict(mode=mode, H=plan['H'], P=p, D=plan['D'], gap_ms=plan['metadata']['gap_ms'],
                   count=len(samples), min_cosine=min(sample['cosine'] for sample in samples))
        medians, trimmed = {}, {}
        for path in plan['paths']:
            key = path.lower().replace('-', '_')
            stages = [sample['paired_truth'][path] for sample in samples]
            for phase in PHASES:
                row[f'{key}_{phase}_med'] = round(statistics.median(s[phase] for s in stages), 2)
            post_return = [s['post_return_ms'] for s in stages]
            medians[path] = statistics.median(post_return)
            trimmed[path] = trimmed_mean(post_return)
            row[f'{key}_post_return_ms_med'] = round(medians[path], 2)
            row[f'{key}_post_return_ms_trim'] = round(trimmed[path], 2)
        row['winner_med'] = min(medians, key=medians.get)
        row['winner_trim'] = min(trimmed, key=trimmed.get)
        rows.append(row)
    if rows:
        csv_path = output / f'{mode}-truth.csv'
        pending = csv_path.with_suffix('.csv.tmp')
        with pending.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(csv_path)
        save(output / f'{mode}-truth.json', dict(plan_sha256=plan_sha, mode=mode, rows=rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'results/mainline-eval/main-4b-ablation/fixtures.json')
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'configs/model-default.json')
    parser.add_argument('--modes', nargs='+', choices=('dual', 'shared'))
    parser.add_argument('--h', type=int, default=4096)
    parser.add_argument('--p-points', nargs='+', type=int, default=DEFAULT_P_POINTS)
    parser.add_argument('--decode-tokens', type=int, default=16)
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--force-hc-compact', action='store_true')
    parser.add_argument('--keep-tokens', type=int, default=1536)
    parser.add_argument('--device', default='xiaomi14pro')
    parser.add_argument('--compact-anchor', type=int, choices=[0, 1], default=1)
    parser.add_argument('--kv-hwm-trim', type=int, choices=[0, 1], default=0)
    parser.add_argument('--verify-kv-state', type=int, choices=[0, 1], default=0,
                        help='诊断模式 1 启用全量 KV 回读，默认只核对导入长度和末位置')
    parser.add_argument('--kv-layer-stride', type=int, default=4)
    parser.add_argument('--kv-min-evict-tokens', type=int, default=256)
    parser.add_argument('--gap-ms', type=int, default=0)
    parser.add_argument('--context-size', type=int, help='固定跨切片上下文和夹具取样范围')
    parser.add_argument('--order-seed', type=int, help='每次重复独立打乱 P 点顺序')
    parser.add_argument('--batch-reps', action='store_true', help='同一 P 的重复在一个请求内共享历史预填充')
    args = parser.parse_args()
    args.modes = args.modes or (['shared'] if args.force_hc_compact else ['dual', 'shared'])
    if args.force_hc_compact and args.modes != ['shared']:
        parser.error('三路配对要求 --modes shared')
    p_points = sorted(set(args.p_points))
    if min(args.h, args.decode_tokens, args.reps, p_points[0]) < 1 or not 0 <= args.gap_ms <= 10000:
        parser.error('H、P、Decode token 数和重复次数须为正，Gap 须在 0–10000 ms 内')
    if args.force_hc_compact and not 0 < args.keep_tokens < args.h + p_points[0]:
        parser.error('keep_tokens 须小于最小 H+P，确保三路样本实际裁剪')
    for key in ('output', 'fixtures', 'build', 'model'):
        setattr(args, key, getattr(args, key).resolve())
    args.output.relative_to(ROOT / 'results')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, p_points)


def run(args, p_points):
    build = build_identity(args.build)
    model = read(args.model)
    tokens = args.h + p_points[-1] + args.decode_tokens
    context_size = args.context_size or math.ceil((tokens + 1) / 256) * 256
    if tokens >= context_size:
        raise ValueError('固定 context_size 必须大于最大 H+P+D')
    prompt = load_prompt_slice(args.fixtures, args.context_size or tokens)
    prefix = prompt['ids'][:args.h]
    # Reuse the same continuation for every path and P point. The source is
    # frozen text for the model under test, not an independently generated answer.
    decode_ids = prompt['ids'][-args.decode_tokens:]
    paths = ['CC', 'HC-Full'] + (['HC-Compact'] if args.force_hc_compact else [])
    metadata = dict(
        git_commit=build['git_commit'], build_id=build['build_id'],
        device=args.device, model=model['model_id'], shared_weights='shared' in args.modes,
        compact_anchor=bool(args.compact_anchor) if args.force_hc_compact else False,
        kv_hwm_trim=bool(args.kv_hwm_trim), kv_layer_stride=args.kv_layer_stride,
        kv_min_evict_tokens=args.kv_min_evict_tokens, gap_ms=args.gap_ms,
    )
    plan = dict(protocol='paired-post-return', build=build, model=model, metadata=metadata,
                H=args.h, D=args.decode_tokens, p_points=p_points, reps=args.reps,
                paths=paths, modes=args.modes, context_size=context_size,
                keep_tokens=args.keep_tokens if args.force_hc_compact else None,
                verify_kv_state=bool(args.verify_kv_state), htp_kv_offload=False,
                fixture_sha256=sha256(args.fixtures), source_case=prompt['case_id'],
                source_turn=prompt['turn_index'], decode_ids=decode_ids,
                collector_sha256=sha256(Path(__file__)), launcher_sha256=sha256(LAUNCHER))
    if args.order_seed is not None:
        plan['order_seed'] = args.order_seed
    if args.batch_reps:
        plan['batch_reps'] = True
    plan_sha = ensure_plan(args.output, plan)
    print('A2O_META', json.dumps(metadata, ensure_ascii=False), flush=True)
    save(args.output / 'metadata.json', metadata)

    status = dict(status='running', completed={})

    def update_status(**items):
        status.update(items)
        save(args.output / 'status.json', status)

    try:
        for mode in args.modes:
            pending = []
            for batch in measurement_batches(p_points, args.reps, args.order_seed, args.batch_reps):
                missing = []
                for p, rep in batch:
                    path = sample_path(args.output, mode, p, rep)
                    if path.exists():
                        old = read_sample(path, plan_sha, mode, args.h, p, args.decode_tokens, rep)
                        validate_truth(old, paths, decode_ids, args.gap_ms, bool(args.verify_kv_state))
                    else:
                        missing.append((rep, path))
                if missing:
                    pending.append((batch[0][0], missing))
            if pending:
                directory = args.output / f'{args.output.name}-{mode}-{time.time_ns()}'
                try:
                    update_status(stage='starting_service', mode=mode)
                    service('start', mode, args, directory, plan)
                    for p, missing in pending:
                        update_status(stage='measuring', mode=mode, P=p, reps=[rep for rep, _ in missing])
                        stem = f'p{p:03d}-r{missing[0][0]:02d}'
                        rows = calibrate(args, prefix, prompt['ids'][args.h:args.h + p], decode_ids,
                                         len(missing), directory / f'{stem}-bad-shape.json')
                        for (rep, path), row in zip(missing, rows):
                            try:
                                validate_truth(row, paths, decode_ids, args.gap_ms, bool(args.verify_kv_state))
                            except (ValueError, KeyError) as exc:
                                save(directory / f'{stem}-failed.json', dict(error=str(exc), rows=rows))
                                raise
                            save(path, dict(plan_sha256=plan_sha, mode=mode, H=args.h, P=p,
                                            D=args.decode_tokens, rep=rep, row=row))
                finally:
                    service('stop', mode, args, directory, plan)
            status['completed'][mode] = len(summarize_mode(args.output, plan, plan_sha, mode))
        save(args.output / 'summary.json', dict(plan_sha256=plan_sha, H=args.h, D=args.decode_tokens,
                                                p_points=p_points, modes=args.modes, paths=paths))
        update_status(status='passed', stage='complete')
    except Exception as exc:
        update_status(status='failed', error=str(exc))
        traceback.print_exc()
        raise
    except KeyboardInterrupt:
        update_status(status='interrupted')
        raise


if __name__ == '__main__':
    main()
