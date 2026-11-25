#!/usr/bin/env python3
"""准备 Table 1 的冻结轨迹，然后串行采集六组真机对照。"""

import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.collect.frozen_ablation_corpus import normalize
from scripts.collect.host_kv_accuracy_bfcl_worker import _to_oai_tools
from scripts.run.frozen_ablation import Campaign, main as run_campaign, read, request, save
from scripts.run.t6_table1_summary import TABLE1_ARMS

SOURCE_CASES = (
    'get_wifi', 'send_message_with_contact_content_cellular_off',
    'retail-7688b6ef5255-3', 'telecom-52b4b12a5622-10',
)
BFCL_CASES = ('multi_turn_base_123', 'multi_turn_base_151')
PRIOR = ROOT / 'results/mainline-eval/t6-8b-table1-gap0'


def freeze_bfcl(source, output):
    messages_file = source / 'messages.json'
    entry_file = source / 'entry.json'
    messages = read(messages_file)
    tools = _to_oai_tools(read(entry_file)['function'])
    start = next(i for i, msg in enumerate(messages) if msg['role'] == 'assistant')
    records = []
    for i in range(start, len(messages), 2):
        if messages[i]['role'] != 'assistant' or (i != start and messages[i-1]['role'] != 'tool'):
            break
        body = {'messages': messages[:i], 'tools': tools}
        text = messages[i]['content']
        stored = output / f'{source.name}-{i:02d}.request.json'
        save(stored, body)
        prompt = request('/apply-template', body)['prompt']
        prompt_ids = request('/tokenize', {'content': prompt})['tokens']
        decode_ids = request('/tokenize', {'content': text})['tokens']
        records.append((i, stored, prompt_ids, decode_ids, None))
    fixture = normalize('bfcl', source.name, source, 'qwen3.5-9b-bf16', records)
    fixture['source_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (messages_file, entry_file)
    }
    return fixture


def prepare_fixtures(args):
    original = read(ROOT / 'results/mainline-eval/main-4b-ablation/fixtures.json')
    cases = [c for c in original['cases'] if c['id'] in SOURCE_CASES]
    if set(c['id'] for c in cases) != set(SOURCE_CASES):
        raise ValueError('缺少指定的 ToolSandbox 或 τ²-bench 轨迹')
    pending = {'cases': cases, 'calibration_cases': []}
    save(args.fixtures, pending)
    campaign = Campaign(args, pending)
    service = args.output / f'bfcl-renderer-{time.time_ns()}'
    source = ROOT / 'results/_archive/20261108-bfcl-text-pilot'
    audit = {}
    try:
        campaign.service('start', 'full_cpu', service)
        pending['cases'].extend(freeze_bfcl(source / name, service) for name in BFCL_CASES)
        for case in pending['cases']:
            audit[case['id']] = campaign.verify_renderer(case)
        save(service / 'renderer-audit.json', audit)
        if any(audit.values()):
            raise ValueError(f'冻结输入渲染不匹配 {audit}')
    finally:
        campaign.service('stop', 'full_cpu', service)
    save(args.output / 'renderer-audit.json', audit)
    save(args.output / 'fixtures.json', pending)


DEFAULT_ROUTE_COSTS = ROOT / 'configs/qwen3-8b-route-costs.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--route-costs', type=Path, default=DEFAULT_ROUTE_COSTS)
    parser.add_argument('--cpu-threads', type=int, default=None, choices=range(1, 17))
    parser.add_argument('--max-quiet-c', type=float, default=37.0)
    parser.add_argument('--min-prime-khz', type=int, default=2400000)
    parser.add_argument('--min-mem-avail-kib', type=int, default=4000000)
    parser.add_argument('--cooldown-timeout-s', type=int, default=240)
    cli_args = parser.parse_args()
    args = cli_args
    args.output = args.output.resolve()
    args.build = args.build.resolve()
    route_src = args.route_costs.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cpu_registry = ROOT / 'results/20261118-qwen3-8b-strict-cpu-frozen/libggml.so'
    args.route_costs = args.output / 'route-costs.json'
    args.kv_costs = args.output / 't5-kv-priors.json'
    route = read(args.route_costs if args.route_costs.exists() else route_src)
    route_threads = route.get('model', {}).get('cpu_threads')
    if args.cpu_threads is None:
        args.cpu_threads = int(route_threads) if route_threads is not None else 4
    elif route_threads is not None and int(route_threads) != args.cpu_threads:
        raise ValueError(f'route_costs cpu_threads={route_threads} 与 --cpu-threads={args.cpu_threads} 不一致')
    if not args.route_costs.exists():
        route['model']['routing_p_min'] = 16
        save(args.route_costs, route)
    if not args.kv_costs.exists():
        save(args.kv_costs, {**read(PRIOR / 't5-kv-priors.json'), 'handoff_saving_enabled': True})
    args.model = ROOT / 'configs/model-default.json'
    args.context_size = 9472
    args.conditions = list(TABLE1_ARMS)
    args.device = 'xiaomi14pro'
    args.verify_kv_state = args.kv_hwm_trim = args.gap_ms = 0
    args.compact_anchor = 1
    args.memory_ledger = args.m1_explorer = True
    args.kv_layer_stride, args.kv_min_evict_tokens = 4, 256
    args.case, args.single_turn, args.force_path = None, False, 'none'
    args.fixtures = args.output / 'fixtures.json'
    if not args.fixtures.exists():
        args.fixtures = args.output / 'fixtures-pre-render.json'
        prepare_fixtures(args)
        args.fixtures = args.output / 'fixtures.json'
    sys.argv = [sys.argv[0], '--output', str(args.output), '--build', str(args.build),
                '--cpu-registry', str(args.cpu_registry), '--fixtures', str(args.fixtures),
                '--route-costs', str(args.route_costs), '--kv-costs', str(args.kv_costs),
                '--gap-ms', '0', '--verify-kv-state', '0', '--memory-ledger', '--kv-hwm-trim', '0',
                '--compact-anchor', '1', '--context-size', str(args.context_size),
                '--cpu-threads', str(args.cpu_threads),
                '--max-quiet-c', str(args.max_quiet_c),
                '--min-prime-khz', str(args.min_prime_khz),
                '--min-mem-avail-kib', str(args.min_mem_avail_kib),
                '--cooldown-timeout-s', str(args.cooldown_timeout_s),
                '--conditions', *TABLE1_ARMS]
    run_campaign()


if __name__ == '__main__':
    main()
