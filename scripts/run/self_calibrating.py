#!/usr/bin/env python3
"""无手机演示：虚拟时钟驱动三个模块，不加载模型、不执行动作。"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling import CostKey, SamplingProfiler
from a2o.runtime.self_calibrating import SelfCalibratingRuntime
from a2o.scheduling.online_cost import OnlineCostModel
from a2o.scheduling.offline_cost import OfflineCostTable
from a2o.warmup import RequestWarmup, WarmupConfig, WarmupController


class VirtualClock:
    def __init__(self):
        self.ns = 0

    def advance(self, ms):
        self.ns += round(ms * 1e6)

    def clock_ns(self):
        return self.ns

    def seconds(self):
        return self.ns / 1e9


def run(config, scenario, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({
        'execution': 'synthetic', 'timing_valid_for_performance': False,
        'config': config, 'scenario': scenario,
        'scope': '无手机、虚拟时间；只演示机制，不代表实际模型耗时或性能收益'
    }, indent=2, ensure_ascii=False) + '\n')
    try:
        if config['execution'] != 'synthetic':
            raise ValueError('this entry supports synthetic execution only')
        clock = VirtualClock()
        runtime = SelfCalibratingRuntime(
            warmup=RequestWarmup(**config['warmup']),
            profiler=SamplingProfiler(**config['profiling'], clock_ns=clock.clock_ns),
            costs=OnlineCostModel(**config['online_cost'], clock=clock.seconds))
        if config.get('offline_table'):
            runtime.offline = OfflineCostTable.load(
                ROOT / config['offline_table'], signature=config['signature'],
                mode=config['mode'], allow_fixture=True)
        base_key = CostKey(**{name: config[name] for name in (
            'signature', 'mode', 'path', 'history_tokens', 'new_tokens', 'decode_tokens')}, phase='prefill')
        key = base_key
        warmup_trace = iter(scenario['warmup_ms'])
        last_ms = scenario['warmup_ms'][-1]
        with (out / 'events.jsonl').open('w') as log:
            def emit(event):
                log.write(json.dumps(event, ensure_ascii=False) + '\n')
                log.flush()

            def warmup_step():
                with runtime.profiler.span(key, warmup=True):
                    clock.advance(next(warmup_trace, last_ms))

            for phase in ('model_init', 'vision_init'):
                if phase in scenario['phases_ms']:
                    ms = scenario['phases_ms'][phase]
                    with runtime.profiler.span(replace(key, phase=phase)):
                        clock.advance(ms)
                    emit({'event': phase, 'duration_ms': ms})
            runtime.flush()
            initialization = WarmupController(
                WarmupConfig(**config.get('initialization_warmup', {})),
                clock_ns=clock.clock_ns).run(warmup_step)
            emit({'event': 'initialization_warmup', **asdict(initialization)})
            runtime.flush()
            first_ready_request = None
            first_online_request = None
            for i in range(scenario['requests']):
                shapes = scenario.get('shapes', [{}])
                key = replace(base_key, **shapes[i % len(shapes)])
                factor = scenario['drift_factor'] if i >= scenario['drift_after'] else 1.
                phases = {}
                for phase, ms in scenario['phases_ms'].items():
                    if phase in {'model_init', 'vision_init'}:
                        continue
                    phase_key = replace(key, phase=phase)
                    # 合成轨迹显式假设桶内成本率相同；并非设备模型的实测规律。
                    def units(k):
                        return max(1, {'entry': k.history_tokens,
                                       'switch': k.history_tokens + k.new_tokens,
                                       'prefill': k.new_tokens, 'decode': k.decode_tokens}.get(k.phase, 1))
                    duration = ms * factor * units(phase_key) / units(replace(base_key, phase=phase))
                    phases[phase] = duration
                    with runtime.span(phase_key):
                        clock.advance(duration)
                used = runtime.flush()
                prediction = runtime.path_prediction(key, scenario['gap_ms'])
                if prediction is not None and first_ready_request is None:
                    first_ready_request = i + 1
                if (prediction is not None and prediction['samples'] >= runtime.costs.min_samples
                        and first_online_request is None):
                    first_online_request = i + 1
                emit({'event': 'request', 'request': i, 'updated': used,
                      'key': asdict(key), 'phases_ms': phases,
                      'warmup_state': runtime.state(key), 'path_prediction': prediction,
                      # 只回放配置指定的路径；建议不驱动额外候选推理。
                      'router_recommendation': runtime.choose(key, 'ready') if runtime.offline else None})
        result = {'execution': 'synthetic', 'timing_valid_for_performance': False,
                  'initialization_warmup': asdict(initialization),
                  'warmup_state': runtime.state(key), 'first_ready_request': first_ready_request,
                  'first_online_request': first_online_request,
                  'accepted': runtime.costs.accepted,
                  'ignored': runtime.costs.ignored, 'dropped': runtime.profiler.dropped,
                  'path_cost_ms': runtime.path_cost(key, scenario['gap_ms']),
                  'costs': runtime.costs.snapshot()}
        (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        return result
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps({'error': repr(exc)}, indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/self-calibrating-local.json')
    parser.add_argument('--scenario', type=Path, default=ROOT / 'scenarios/self-calibrating-local.json')
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'results' / (stamp + '-self-calibrating-local')
    result = run(json.loads(args.config.read_text()), json.loads(args.scenario.read_text()), out)
    print(json.dumps({'directory': str(out), 'warmup_state': result['warmup_state'],
                      'first_ready_request': result['first_ready_request'],
                      'first_online_request': result['first_online_request'],
                      'initialization_steps': result['initialization_warmup']['steps'],
                      'path_cost_ms': result['path_cost_ms'],
                      'timing_valid_for_performance': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
