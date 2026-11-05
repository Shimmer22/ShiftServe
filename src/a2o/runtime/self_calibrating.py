"""三个独立模块的可选组合；请求边界批量更新，不隐式执行预热。"""
from dataclasses import replace
import math

from ..profiling import SamplingProfiler
from ..scheduling.online_cost import OnlineCostModel
from ..warmup import RequestWarmup


class SelfCalibratingRuntime:
    def __init__(self, *, warmup=None, profiler=None, costs=None):
        self.warmup = warmup if warmup is not None else RequestWarmup()
        self.profiler = profiler if profiler is not None else SamplingProfiler()
        self.costs = costs if costs is not None else OnlineCostModel()

    def state(self, key):
        return self.warmup.state(self.costs.sample_count(key), self.costs.min_samples)

    def span(self, key):
        """包住一次正常阶段执行；冷桶与过期桶优先采，不执行额外请求。"""
        count = self.costs.sample_count(key) if self.profiler.enabled else self.costs.min_samples
        return self.profiler.span(
            key, sampling_key=self.costs.group_key(key),
            force=self.costs.enabled and self.warmup.needs_sample(count, self.costs.min_samples))

    def flush(self):
        samples = self.profiler.drain()
        return sum(self.costs.update(sample) for sample in samples)

    def path_cost(self, key, gap_ms=0.):
        prediction = self.path_prediction(key, gap_ms)
        return None if prediction is None else prediction['total_ms']

    def path_prediction(self, key, gap_ms=0.):
        """对齐已有 path_cost：Gap 仅隐藏 entry；初始化/视觉单独计时。"""
        if not math.isfinite(gap_ms) or gap_ms < 0:
            raise ValueError('gap_ms must be finite and nonnegative')
        values = {phase: self.costs.prediction(replace(key, phase=phase))
                  for phase in ('entry', 'prefill', 'switch', 'decode')}
        if any(value is None for value in values.values()):
            return None
        return {'total_ms': max(0., values['entry']['duration_ms'] - gap_ms) + sum(
                    values[phase]['duration_ms'] for phase in ('prefill', 'switch', 'decode')),
                'source': 'shape_bucket' if any(v['source'] == 'shape_bucket' for v in values.values()) else 'exact',
                'samples': min(v['samples'] for v in values.values())}
