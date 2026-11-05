"""三个独立模块的可选组合；请求边界批量更新，不隐式执行预热。"""
from dataclasses import replace
import math

from ..profiling import SamplingProfiler
from ..scheduling.online_cost import OnlineCostModel
from ..warmup import RequestWarmup


class SelfCalibratingRuntime:
    def __init__(self, *, warmup=None, profiler=None, costs=None, offline=None):
        self.warmup = warmup if warmup is not None else RequestWarmup()
        self.profiler = profiler if profiler is not None else SamplingProfiler()
        self.costs = costs if costs is not None else OnlineCostModel()
        self.offline = offline

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
        if self.offline is not None:
            baseline = self.offline.lookup(key)
            if baseline is None:
                return None  # 在线样本不扩张原表的候选覆盖。
            entry = values['entry']
            future = [values[p] for p in ('prefill', 'switch', 'decode')]
            online_future = all(v is not None for v in future)
            # 旧表仅有 future 合计；阶段不齐时保留该合计，避免重复计费。
            future_ms = sum(v['duration_ms'] for v in future) if online_future else baseline.future_ms
            entry_ms = entry['duration_ms'] if entry is not None else baseline.entry_ms
            used = ([entry] if entry is not None else []) + (future if online_future else [])
            return {'total_ms': max(0., entry_ms - gap_ms) + future_ms,
                    'entry_ms': entry_ms, 'future_ms': future_ms,
                    'source': 'online_corrected' if used else 'offline',
                    'entry_source': entry['source'] if entry is not None else 'offline',
                    'future_source': ('shape_bucket' if any(v['source'] == 'shape_bucket' for v in future)
                                      else 'exact') if online_future else 'offline',
                    'offline_source': self.offline.source,
                    'samples': min(v['samples'] for v in used) if used else 0}
        if any(value is None for value in values.values()):
            return None
        return {'total_ms': max(0., values['entry']['duration_ms'] - gap_ms) + sum(
                    values[phase]['duration_ms'] for phase in ('prefill', 'switch', 'decode')),
                'source': 'shape_bucket' if any(v['source'] == 'shape_bucket' for v in values.values()) else 'exact',
                'samples': min(v['samples'] for v in values.values())}

    def choose(self, key, availability):
        """host 适配现有 Router 的 CPU 权威、CC/HC、固定 D 选路语义。"""
        if self.offline is None:
            raise ValueError('router requires an offline cost table')
        cc = self.path_prediction(replace(key, path='CC'))
        hc = self.path_prediction(replace(key, path='HC'))
        result = {'backend': 'cpu', 'reason': 'coverage_missing',
                  'cost_covered': cc is not None and hc is not None,
                  'availability': availability, 'CC': cc, 'HC': hc}
        if not result['cost_covered']:
            return result
        if availability != 'ready':
            result['reason'] = 'candidate_' + availability
            return result
        # 历史准备已完成；只比较 future，不把 entry 重复计入响应。
        if hc['future_ms'] < cc['future_ms']:
            result.update(backend='htp', reason='htp_lower_cost')
        else:
            result['reason'] = 'cpu_lower_cost'
        return result
