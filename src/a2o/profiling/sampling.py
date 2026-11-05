"""仅在命中抽样时读取两次时钟；有界队列，批量消费。"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class CostKey:
    # signature 由调用方标识模型、量化、上下文配置、运行时和设备配置。
    signature: str
    mode: str
    path: str
    history_tokens: int
    new_tokens: int
    decode_tokens: int
    phase: str

    def __post_init__(self):
        if not self.signature or self.mode not in {'cpu', 'htp', 'hybrid'}:
            raise ValueError('explicit signature and cpu/htp/hybrid mode required')
        if self.path not in {'CC', 'HC', 'CH', 'HH'}:
            raise ValueError('unknown path')
        if ((self.mode == 'cpu' and self.path != 'CC')
                or (self.mode == 'htp' and self.path != 'HH')):
            raise ValueError('path conflicts with mode')
        if min(self.history_tokens, self.new_tokens, self.decode_tokens) < 0:
            raise ValueError('token counts must be nonnegative')
        if self.phase not in {'model_init', 'vision_init', 'observation',
                              'entry', 'prefill', 'switch', 'decode'}:
            raise ValueError('unknown phase')


@dataclass(frozen=True)
class Sample:
    key: CostKey
    duration_ms: float
    warmup: bool = False
    failed: bool = False


class SamplingProfiler:
    def __init__(self, *, enabled=True, every=8, cold_samples=3, capacity=256,
                 clock_ns=perf_counter_ns):
        if every < 1 or capacity < 1 or cold_samples < 0:
            raise ValueError('every and capacity must be positive')
        self.enabled, self.every = enabled, every
        self.cold_samples = cold_samples
        self.clock_ns = clock_ns
        self._samples = deque(maxlen=capacity)
        self._seen = {}
        self._capacity = capacity
        self.dropped = 0

    @contextmanager
    def span(self, key, *, warmup=False, force=False, sampling_key=None):
        """同步完成的阶段边界计时；异步后端应在已有完成点调用 record。"""
        if not self.enabled:
            yield
            return
        # 分阶段抽样，避免固定阶段顺序与 every 共振而只采到 entry。
        group = key if sampling_key is None else sampling_key
        count, collected = self._seen.get(group, (0, 0))
        if group not in self._seen and len(self._seen) >= self._capacity:
            del self._seen[next(iter(self._seen))]
        selected = force or collected < self.cold_samples or count == 0
        self._seen[group] = ((count + 1) % self.every, collected)
        if not selected:
            yield
            return
        start = self.clock_ns()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(Sample(key, (self.clock_ns() - start) / 1e6, warmup, failed))
            # 初始化和失败不能让正常请求提前进入稀疏采样。
            if not warmup and not failed and collected < self.cold_samples:
                self._seen[group] = (1 % self.every, collected + 1)

    def record(self, sample):
        """接收已有阶段计时，不重复读时钟；也用于无手机轨迹回放。"""
        if not self.enabled:
            return
        if len(self._samples) == self._samples.maxlen:
            self.dropped += 1
        self._samples.append(sample)

    def drain(self):
        samples = list(self._samples)
        self._samples.clear()
        return samples
