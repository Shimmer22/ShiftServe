#!/usr/bin/env python3
"""在 Xiaomi 14 Pro (Qwen3-8B Q4_0) 上先做线程寻优，再对 H∈{1024,2048,4096}、P∈[16,64]
执行 独立CPU / 共享CPU / 独立HTP 三路交错 Prefill 测量，并同步测量 M(H) Handoff 开销。
"""

import argparse
import csv
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling.pd_sweep import save
from a2o.runtime.device_transport import DeviceTransport
from a2o.runtime.network_retry import retry_network

MAIN = ROOT / 'results/mainline-eval'
DEFAULT_BUILD = MAIN / 'prefill-3way-h1k2k4k-build'
DEFAULT_OUT = MAIN / 'prefill-3way-h1k2k4k-8b'
DEFAULT_FIXTURE = MAIN / 'prefill-motivation-h4096-8b/fixture.json'
DEFAULT_MODEL = ROOT / 'configs/model-default.json'

HISTORIES = (1024, 2048, 4096)
THREAD_CANDIDATES = (2, 3, 4, 6, 8)
THREAD_P_POINTS = (64,)
MAIN_P_POINTS = (
    16, 17, 20, 22, 24, 28, 31, 32, 33, 36, 40, 48, 56, 64,
)
HANDOFF_P_POINTS = (0, 16, 32, 64)
LATIN_ORDERS = (
    ('independent_cpu', 'shared_cpu', 'independent_htp'),
    ('independent_htp', 'shared_cpu', 'independent_cpu'),
    ('shared_cpu', 'independent_htp', 'independent_cpu'),
    ('independent_cpu', 'independent_htp', 'shared_cpu'),
)


def read(path: Path):
    return json.loads(path.read_text())


def post_ab(request: dict, timeout: int = 900) -> dict:
    base = os.environ['A2O_PHONE_HTTP_BASE']
    req = Request(
        base + '/a2o/prefill-weight-ab',
        json.dumps(request).encode(),
        {'Content-Type': 'application/json'},
    )
    def send():
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    return retry_network(send)


def start_service(build: Path, service: Path, model: dict, arm: str, cpu_threads: int) -> dict:
    shared = (arm == 'shared_cpu')
    condition = {'independent_cpu': 'cpu', 'shared_cpu': 'route', 'independent_htp': 'htp'}[arm]
    common = ['--condition', condition, '--output', str(service), '--logs-only', '--retry-network']
    launch = [
        '--runtime-overlay', str(build),
        '--model', model['model_path'],
        '--model-id', model['model_id'],
        '--model-sha256', model['model_sha256'],
        '--context-size', '8192',
        '--cpu-threads', str(cpu_threads),
        '--cpu-repack', '0',
        '--kv-hwm-trim', '0',
        '--verify-kv-state', '0',
        '--gap-ms', '0',
        '--replay-cache',
        '--memory-ledger',
    ]
    if shared:
        launch.append('--shared-weights')
    entry = [sys.executable, 'scripts/run/route_phone.py']
    service.parent.mkdir(parents=True, exist_ok=True)
    with (service.parent / 'service-launch.log').open('a') as log:
        subprocess.run([*entry, 'start', *common, *launch], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    health = read(service / 'health.json')
    manifest = read(build / 'manifest.json')
    if (health['build_id'] != manifest['build_id']
            or health['shared_weights'] is not shared
            or health['context_size'] != 8192
            or health['cpu_threads'] != cpu_threads
            or health['htp_host_threads'] != 4
            or health['htp_kv_offload']):
        raise RuntimeError(f'Service health mismatch for {arm}: {health}')
    return health


def stop_service(service: Path, arm: str) -> None:
    condition = {'independent_cpu': 'cpu', 'shared_cpu': 'route', 'independent_htp': 'htp'}[arm]
    common = ['--condition', condition, '--output', str(service), '--logs-only', '--retry-network']
    entry = [sys.executable, 'scripts/run/route_phone.py']
    with (service.parent / 'service-stop.log').open('a') as log:
        subprocess.run([*entry, 'stop', *common], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def verify_runtime_log(service: Path, arm: str) -> None:
    runtime = (service / 'device/runtime.log').read_text(errors='replace')
    if arm == 'independent_cpu':
        if 'CPU_Mapped model buffer size' not in runtime or 'HTP0-REPACK model buffer size' in runtime:
            raise RuntimeError(f'independent_cpu weight layout mismatch in {service}')
    elif arm == 'shared_cpu':
        if 'HTP0-REPACK model buffer size' not in runtime or 'CPU-only scheduler, shared packed weights' not in runtime:
            raise RuntimeError(f'shared_cpu weight layout mismatch in {service}')
    elif arm == 'independent_htp':
        if 'HTP0-REPACK model buffer size' not in runtime or 'CPU_Mapped model buffer size' in runtime:
            raise RuntimeError(f'independent_htp weight layout mismatch in {service}')


def stage_snapshots(
    transport: DeviceTransport,
    target_service: Path,
    source_service: Path,
    histories,
) -> None:
    remote = transport.remote_root
    src_dir = f'{remote}/{source_service.name}/replay-cache'
    dst_dir = f'{remote}/{target_service.name}/replay-cache'
    guard = 'test "$(cat ' + shlex.quote(remote + '/model.lock/owner') + ')" = ' + shlex.quote(target_service.name)
    cmds = [shlex.join(['mkdir', '-p', dst_dir])]
    for h in histories:
        name = f'prefill-ab-h{h}.json'
        cmds.append(shlex.join(['ln', '-sfn', f'{src_dir}/{name}', f'{dst_dir}/{name}']))
        cmds.append(shlex.join(['ln', '-sfn', f'{src_dir}/{name}.state', f'{dst_dir}/{name}.state']))
    retry_network(lambda: transport.call('shell', guard + ' && ' + ' && '.join(cmds)))


def ensure_snapshots(
    out: Path,
    build: Path,
    model: dict,
    prompt: list[int],
    histories,
    snapshot_source_dir: Path | None = None,
) -> tuple[dict, Path]:
    snap_meta_path = out / 'snapshots.json'
    if snapshot_source_dir is not None:
        src_meta_path = snapshot_source_dir / 'snapshots.json'
        src_service = snapshot_source_dir / 'snapshots' / f'{snapshot_source_dir.name}-snapshots-service'
        if src_meta_path.exists():
            src_meta = read(src_meta_path)
            if all(str(h) in src_meta for h in histories):
                filtered = {str(h): src_meta[str(h)] for h in histories}
                save(snap_meta_path, filtered)
                return filtered, src_service
    default_service = out / 'snapshots' / f'{out.name}-snapshots-service'
    if snap_meta_path.exists():
        meta = read(snap_meta_path)
        if all(str(h) in meta for h in histories):
            return meta, default_service
    snap_dir = out / 'snapshots'
    service = default_service
    snap_dir.mkdir(parents=True, exist_ok=True)
    meta = read(snap_meta_path) if snap_meta_path.exists() else {}
    try:
        start_service(build, service, model, 'independent_cpu', 4)
        for h in histories:
            key = str(h)
            if key in meta:
                continue
            save(out / 'status.json', {'phase': 'preparing_snapshot', 'H': h})
            res = post_ab({
                'prepare_snapshot': True,
                'snapshot_name': f'prefill-ab-h{h}.json',
                'prefix_ids': prompt[:h],
            })
            if res.get('mode') != 'cpu_prefill_ab_prepare_v1' or res.get('H') != h:
                raise RuntimeError(f'Snapshot prepare failed for H={h}: {res}')
            meta[key] = res
            save(snap_meta_path, meta)
    finally:
        if (service / 'pid').exists() or (service / 'health.json').exists():
            try:
                stop_service(service, 'independent_cpu')
            except Exception:
                pass
    return meta, default_service


def measure_sample(
    file: Path,
    request: dict,
    expected_arm: str,
    expected_threads: int,
    *,
    measure_handoff: bool = False,
) -> dict:
    if file.exists():
        recorded = read(file)
        if recorded['request'] != request:
            raise RuntimeError(f'Existing request mismatch at {file}')
        result = recorded['response']
    else:
        result = post_ab(request)
        save(file.with_suffix('.raw.json'), {'request': request, 'response': result})
    expected_backend = 'htp' if (expected_arm == 'independent_htp' or measure_handoff) else 'cpu'
    expected_shared = (expected_arm == 'shared_cpu') or measure_handoff
    if (result.get('mode') != 'cpu_prefill_ab_v1'
            or result.get('backend') != expected_backend
            or result.get('shared_weights') is not expected_shared
            or bool(result.get('measure_handoff', False)) is not measure_handoff
            or result.get('snapshot_checksum') != request['snapshot_checksum']
            or result.get('H') != len(request['prefix_ids'])
            or result.get('P') != len(request.get('input_ids', []))
            or not result.get('restore_verified')
            or not math.isfinite(result.get('restore_ms', -1))):
        raise RuntimeError(f'Sample validation failed ({file}): {result}')
    if not measure_handoff:
        if result['n_threads'] != expected_threads or result['n_threads_batch'] != expected_threads:
            raise RuntimeError(f'Thread count mismatch at {file}: {result["n_threads"]}/{result["n_threads_batch"]} != {expected_threads}')
        if not math.isfinite(result.get('prefill_ms', -1)) or result['prefill_ms'] <= 0:
            raise RuntimeError(f'Invalid prefill_ms at {file}: {result}')
    else:
        if result.get('handoff_ms', -1) <= 0 or result.get('handoff_bytes', 0) <= 0:
            raise RuntimeError(f'Invalid handoff at {file}: {result}')
    if not file.exists():
        file.with_suffix('.raw.json').replace(file)
    return result


def summarize_clocks(res: dict) -> tuple[int, int]:
    cb = res.get('clocks_before', {})
    ca = res.get('clocks_after', {})
    p7_b = cb.get('cpu_policy_khz', {}).get('policy7', 0)
    p7_a = ca.get('cpu_policy_khz', {}).get('policy7', 0)
    return int(p7_b or 0), int(p7_a or 0)


def run_thread_audit(
    out: Path,
    build: Path,
    model: dict,
    prompt: list[int],
    snaps: dict,
    snap_service: Path,
    transport: DeviceTransport,
) -> dict:
    audit_dir = out / 'thread-audit'
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary_path = audit_dir / 'summary.json'
    if summary_path.exists():
        prev = read(summary_path)
        if prev.get('complete') and prev.get('threads') == list(THREAD_CANDIDATES):
            return prev

    h = 2048 if '2048' in snaps else sorted(int(k) for k in snaps)[0]
    snap = snaps[str(h)]
    blocks = [
        ('independent_cpu', (2, 3, 4, 6, 8)),
        ('shared_cpu', (8, 6, 4, 3, 2)),
        ('shared_cpu', (2, 3, 4, 6, 8)),
        ('independent_cpu', (8, 6, 4, 3, 2)),
    ]
    reps_per_block = 1
    rows = []
    completed = 0
    total = len(blocks) * len(THREAD_CANDIDATES) * len(THREAD_P_POINTS) * reps_per_block

    for b_idx, (arm, t_order) in enumerate(blocks):
        block_dir = audit_dir / f'block-{b_idx}-{arm}'
        service = block_dir / f'{out.name}-ta-b{b_idx}-{arm}-service'
        block_dir.mkdir(parents=True, exist_ok=True)
        save(out / 'status.json', {'phase': 'thread_audit_start', 'block': b_idx, 'arm': arm, 'completed': completed, 'total': total})
        try:
            start_service(build, service, model, arm, 4)
            stage_snapshots(transport, service, snap_service, [h])
            # Warmup
            measure_sample(
                block_dir / 'warmup.json',
                {
                    'snapshot_name': snap['snapshot_name'],
                    'snapshot_checksum': snap['snapshot_checksum'],
                    'prefix_ids': prompt[:h],
                    'input_ids': prompt[h:h + 32],
                    'cpu_prefill_threads': 4,
                },
                arm, 4,
            )
            for t in t_order:
                for rep in range(reps_per_block):
                    p_seq = list(THREAD_P_POINTS)
                    random.Random(2026112400 + b_idx * 100 + t * 10 + rep).shuffle(p_seq)
                    for p in p_seq:
                        save(out / 'status.json', {
                            'phase': 'thread_audit_measuring', 'block': b_idx, 'arm': arm,
                            'threads': t, 'round': rep, 'P': p, 'completed': completed, 'total': total,
                        })
                        req = {
                            'snapshot_name': snap['snapshot_name'],
                            'snapshot_checksum': snap['snapshot_checksum'],
                            'prefix_ids': prompt[:h],
                            'input_ids': prompt[h:h + p],
                            'cpu_prefill_threads': t,
                        }
                        res = measure_sample(block_dir / f't{t}-r{rep}-p{p:03d}.json', req, arm, t)
                        p7_b, p7_a = summarize_clocks(res)
                        rows.append({
                            'block': b_idx, 'arm': arm, 'threads': t, 'round': rep, 'H': h, 'P': p,
                            'restore_ms': res['restore_ms'], 'prefill_ms': res['prefill_ms'],
                            'top_token': res['top_token'],
                            'swap_kib': res['resources_after']['VmSwap_kib'],
                            'policy7_before_khz': p7_b, 'policy7_after_khz': p7_a,
                        })
                        completed += 1
        finally:
            stop_service(service, arm)
        verify_runtime_log(service, arm)

    with (audit_dir / 'stage-times.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {'complete': True, 'H': h, 'P': list(THREAD_P_POINTS), 'threads': list(THREAD_CANDIDATES), 'arms': {}}
    for arm in ('independent_cpu', 'shared_cpu'):
        per_t = {}
        for t in THREAD_CANDIDATES:
            pts = {}
            for p in THREAD_P_POINTS:
                vals = [r['prefill_ms'] for r in rows if r['arm'] == arm and r['threads'] == t and r['P'] == p]
                pts[p] = statistics.median(vals)
            geomean_all = math.exp(statistics.mean(math.log(pts[p]) for p in THREAD_P_POINTS))
            cross_ps = [p for p in (16, 24, 32) if p in pts]
            geomean_cross = math.exp(statistics.mean(math.log(pts[p]) for p in cross_ps)) if cross_ps else geomean_all
            per_t[str(t)] = {
                'medians_ms': {str(p): pts[p] for p in THREAD_P_POINTS},
                'geomean_all_ms': geomean_all,
                'geomean_16_32_ms': geomean_cross,
            }
        best_t = min(THREAD_CANDIDATES, key=lambda t: per_t[str(t)]['geomean_all_ms'])
        best_cross_t = min(THREAD_CANDIDATES, key=lambda t: per_t[str(t)]['geomean_16_32_ms'])
        summary['arms'][arm] = {
            'per_thread': per_t,
            'optimal_threads_all': best_t,
            'optimal_threads_16_32': best_cross_t,
        }
    summary['selected'] = {
        arm: summary['arms'][arm]['optimal_threads_all'] for arm in ('independent_cpu', 'shared_cpu')
    }
    save(summary_path, summary)
    return summary


def run_main_sweep(
    out: Path,
    build: Path,
    model: dict,
    prompt: list[int],
    snaps: dict,
    snap_service: Path,
    transport: DeviceTransport,
    ind_threads: int,
    shr_threads: int,
    *,
    histories: tuple[int, ...] = HISTORIES,
    p_points: tuple[int, ...] = MAIN_P_POINTS,
    handoff_p_points: tuple[int, ...] = HANDOFF_P_POINTS,
    inter_block_cool_s: float = 0.0,
) -> None:
    prefill_rows = []
    handoff_rows = []
    total_prefill = len(LATIN_ORDERS) * 3 * len(histories) * len(p_points)
    total_handoff = len(LATIN_ORDERS) * len(histories) * len(handoff_p_points)
    completed_prefill = 0
    completed_handoff = 0

    for rep_idx, arm_order in enumerate(LATIN_ORDERS):
        for arm_idx, arm in enumerate(arm_order):
            threads = {'independent_cpu': ind_threads, 'shared_cpu': shr_threads, 'independent_htp': 4}[arm]
            block_dir = out / f'rep-{rep_idx}-{arm}'
            service = block_dir / f'{out.name}-r{rep_idx}-{arm}-service'
            block_dir.mkdir(parents=True, exist_ok=True)
            grid = [(h, p) for h in histories for p in p_points]
            random.Random(2026112450 + rep_idx * 17 + {'independent_cpu': 1, 'shared_cpu': 2, 'independent_htp': 3}[arm]).shuffle(grid)
            hnd_grid = [(h, p) for h in histories for p in handoff_p_points] if arm == 'shared_cpu' else []
            random.Random(2026112490 + rep_idx).shuffle(hnd_grid)
            all_cached = (
                (block_dir / 'warmup.json').exists()
                and (service / 'device/runtime.log').exists()
                and all((block_dir / f'h{h}-p{p:03d}.json').exists() for h, p in grid)
                and all((block_dir / f'handoff-h{h}-p{p:03d}.json').exists() for h, p in hnd_grid)
            )
            if not all_cached and inter_block_cool_s > 0 and (rep_idx > 0 or arm_idx > 0):
                save(out / 'status.json', {
                    'phase': 'cooling_between_blocks', 'rep': rep_idx, 'arm': arm,
                    'cool_s': inter_block_cool_s,
                    'completed_prefill': completed_prefill, 'total_prefill': total_prefill,
                    'completed_handoff': completed_handoff, 'total_handoff': total_handoff,
                })
                time.sleep(inter_block_cool_s)
            save(out / 'status.json', {
                'phase': 'main_start_service', 'rep': rep_idx, 'arm': arm, 'threads': threads,
                'completed_prefill': completed_prefill, 'total_prefill': total_prefill,
                'completed_handoff': completed_handoff, 'total_handoff': total_handoff,
            })
            started = False
            try:
                if not all_cached:
                    start_service(build, service, model, arm, threads)
                    started = True
                    stage_snapshots(transport, service, snap_service, histories)
                # Warmup at H=2048 (if available) else first history, P=32
                w_h = 2048 if '2048' in snaps else histories[0]
                w_snap = snaps[str(w_h)]
                measure_sample(
                    block_dir / 'warmup.json',
                    {
                        'snapshot_name': w_snap['snapshot_name'],
                        'snapshot_checksum': w_snap['snapshot_checksum'],
                        'prefix_ids': prompt[:w_h],
                        'input_ids': prompt[w_h:w_h + 32],
                    },
                    arm, threads,
                )
                # Interleaved (H, P) prefill measurements
                for h, p in grid:
                    snap = snaps[str(h)]
                    if not all_cached:
                        save(out / 'status.json', {
                            'phase': 'main_prefill', 'rep': rep_idx, 'arm': arm, 'threads': threads,
                            'H': h, 'P': p,
                            'completed_prefill': completed_prefill, 'total_prefill': total_prefill,
                            'completed_handoff': completed_handoff, 'total_handoff': total_handoff,
                        })
                    req = {
                        'snapshot_name': snap['snapshot_name'],
                        'snapshot_checksum': snap['snapshot_checksum'],
                        'prefix_ids': prompt[:h],
                        'input_ids': prompt[h:h + p],
                    }
                    res = measure_sample(block_dir / f'h{h}-p{p:03d}.json', req, arm, threads)
                    p7_b, p7_a = summarize_clocks(res)
                    prefill_rows.append({
                        'rep': rep_idx, 'arm': arm, 'threads': threads, 'H': h, 'P': p,
                        'restore_ms': res['restore_ms'], 'prefill_ms': res['prefill_ms'],
                        'top_token': res['top_token'],
                        'swap_kib': res['resources_after']['VmSwap_kib'],
                        'rss_kib': res['resources_after']['VmRSS_kib'],
                        'policy7_before_khz': p7_b, 'policy7_after_khz': p7_a,
                    })
                    completed_prefill += 1

                # Synchronized M(H) / M(H+P) handoff measurement inside each shared block (after CPU prefill finishes)
                if arm == 'shared_cpu':
                    for h, p in hnd_grid:
                        snap = snaps[str(h)]
                        if not all_cached:
                            save(out / 'status.json', {
                                'phase': 'main_handoff', 'rep': rep_idx, 'H': h, 'P': p,
                                'completed_prefill': completed_prefill, 'total_prefill': total_prefill,
                                'completed_handoff': completed_handoff, 'total_handoff': total_handoff,
                            })
                        req = {
                            'snapshot_name': snap['snapshot_name'],
                            'snapshot_checksum': snap['snapshot_checksum'],
                            'prefix_ids': prompt[:h],
                            'input_ids': [] if p == 0 else prompt[h:h + p],
                            'measure_handoff': True,
                            'handoff_only_h': (p == 0),
                            'verify_handoff_bytes': False,
                        }
                        res = measure_sample(block_dir / f'handoff-h{h}-p{p:03d}.json', req, arm, threads, measure_handoff=True)
                        handoff_rows.append({
                            'rep': rep_idx, 'H': h, 'P': p, 'total_tokens': h + p,
                            'restore_ms': res['restore_ms'],
                            'htp_prefill_ms': res['prefill_ms'],
                            'handoff_ms': res['handoff_ms'],
                            'handoff_bytes': res['handoff_bytes'],
                            'top_token': res['top_token'],
                            'swap_kib': res['resources_after']['VmSwap_kib'],
                        })
                        completed_handoff += 1
            finally:
                if started:
                    stop_service(service, arm)
            verify_runtime_log(service, arm)

    with (out / 'stage-times.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(prefill_rows[0]))
        w.writeheader()
        w.writerows(prefill_rows)
    with (out / 'handoff-times.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(handoff_rows[0]))
        w.writeheader()
        w.writerows(handoff_rows)
    save(out / 'status.json', {
        'phase': 'complete',
        'completed_prefill': len(prefill_rows), 'total_prefill': total_prefill,
        'completed_handoff': len(handoff_rows), 'total_handoff': total_handoff,
        'ind_threads': ind_threads, 'shr_threads': shr_threads,
        'peak_prefill_swap_kib': max(r['swap_kib'] for r in prefill_rows),
        'peak_handoff_swap_kib': max(r['swap_kib'] for r in handoff_rows),
    })


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(',') if x.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['snapshots', 'thread-audit', 'main', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--build', type=Path, default=DEFAULT_BUILD)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--model-config', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--histories', type=parse_int_list, default=HISTORIES)
    parser.add_argument('--p-points', type=parse_int_list, default=MAIN_P_POINTS)
    parser.add_argument('--handoff-p-points', type=parse_int_list, default=HANDOFF_P_POINTS)
    parser.add_argument('--snapshot-source-dir', type=Path)
    parser.add_argument('--inter-block-cool-s', type=float, default=0.0)
    parser.add_argument('--ind-threads', type=int)
    parser.add_argument('--shr-threads', type=int)
    args = parser.parse_args()

    out = args.output.resolve()
    build = args.build.resolve()
    model = read(args.model_config.resolve())
    prompt = read(args.fixture.resolve())['choices'][0]['prompt_token_ids']
    if len(prompt) < max(args.histories) + max(args.p_points):
        raise ValueError(f'Fixture prompt too short: {len(prompt)} < {max(args.histories) + max(args.p_points)}')

    snap_src = args.snapshot_source_dir.resolve() if args.snapshot_source_dir else None
    out.mkdir(parents=True, exist_ok=True)
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    with (out / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            snaps, snap_service = ensure_snapshots(out, build, model, prompt, args.histories, snap_src)
            if args.phase == 'snapshots':
                save(out / 'status.json', {'phase': 'snapshots_ready', 'snapshots': snaps})
                return
            if args.phase in ('thread-audit', 'all'):
                ta = run_thread_audit(out, build, model, prompt, snaps, snap_service, transport)
                ind_t = args.ind_threads or int(ta['selected']['independent_cpu'])
                shr_t = args.shr_threads or int(ta['selected']['shared_cpu'])
                if args.phase == 'thread-audit':
                    save(out / 'status.json', {'phase': 'thread_audit_complete', 'selected': {'independent_cpu': ind_t, 'shared_cpu': shr_t}})
                    print(json.dumps(ta, indent=2, ensure_ascii=False))
                    return
            else:
                ta_path = out / 'thread-audit/summary.json'
                ta = read(ta_path) if ta_path.exists() else {}
                ind_t = args.ind_threads or int(ta.get('selected', {}).get('independent_cpu', 4))
                shr_t = args.shr_threads or int(ta.get('selected', {}).get('shared_cpu', 4))
            run_main_sweep(
                out, build, model, prompt, snaps, snap_service, transport, ind_t, shr_t,
                histories=args.histories,
                p_points=args.p_points,
                handoff_p_points=args.handoff_p_points,
                inter_block_cool_s=args.inter_block_cool_s,
            )
        except Exception as exc:
            save(out / 'status.json', {'phase': 'failed', 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()
