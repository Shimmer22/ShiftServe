"""分阶段的小桶成本率 EMA；配置/路径隔离，不额外执行推理。"""
from dataclasses import dataclass, replace
import math
from time import monotonic


@dataclass
class Estimate:
    mean_ms: float
    samples: int
    updated_at: float
    first_key: object
    mixed_shapes: bool = False


class OnlineCostModel:
    def __init__(self, *, alpha=0.2, min_samples=3, max_keys=256,
                 ttl_s=300., enabled=True, shape_reuse=True,
                 h_bucket=128, p_bucket=8, d_bucket=4, clock=monotonic):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('alpha must be in (0, 1]')
        if min_samples < 1 or max_keys < 1 or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError('invalid online cost limits')
        if min(h_bucket, p_bucket, d_bucket) < 1:
            raise ValueError('bucket widths must be positive')
        self.alpha, self.min_samples, self.max_keys = alpha, min_samples, max_keys
        self.ttl_s, self.enabled, self.clock = ttl_s, enabled, clock
        self.shape_reuse = shape_reuse
        self.h_bucket, self.p_bucket, self.d_bucket = h_bucket, p_bucket, d_bucket
        # 插入顺序即最近更新时间；达到上限时淘汰最久未更新项。
        self._costs = {}
        self.accepted = self.ignored = 0

    @staticmethod
    def _bucket(value, width):
        # 零与小形状独立；floor 桶默认不会把 P31/P32 合并。
        return value if value < width else value // width * width

    def group_key(self, key):
        if not self.shape_reuse:
            return key
        h, p, d = key.history_tokens, key.new_tokens, key.decode_tokens
        if key.phase == 'entry':
            p = d = 0
        elif key.phase == 'switch':
            h, p, d = h + p, 0, 0
        elif key.phase in {'prefill', 'observation'}:
            d = 0
        elif key.phase == 'decode':
            h, p = h + p, 0
        elif key.phase in {'model_init', 'vision_init'}:
            h = p = d = 0
        return replace(key, history_tokens=self._bucket(h, self.h_bucket),
                       new_tokens=self._bucket(p, self.p_bucket),
                       decode_tokens=self._bucket(d, self.d_bucket))

    def _units(self, key):
        if not self.shape_reuse:
            return 1
        units = {'entry': key.history_tokens,
                 'switch': key.history_tokens + key.new_tokens,
                 'prefill': key.new_tokens, 'decode': key.decode_tokens}
        return max(1, units.get(key.phase, 1))

    def sample_count(self, key):
        value = self._costs.get(self.group_key(key))
        if (not self.enabled or value is None
                or self.clock() - value.updated_at > self.ttl_s):
            return 0
        return value.samples

    def update(self, sample):
        if not self.enabled:
            return False
        if (sample.warmup or sample.failed or not math.isfinite(sample.duration_ms)
                or sample.duration_ms < 0):
            self.ignored += 1
            return False
        now = self.clock()
        group = self.group_key(sample.key)
        observed = sample.duration_ms / self._units(sample.key)
        old = self._costs.pop(group, None)
        if old is None or now - old.updated_at > self.ttl_s:
            value = Estimate(observed, 1, now, sample.key)
        else:
            value = Estimate(old.mean_ms + self.alpha * (observed - old.mean_ms),
                             old.samples + 1, now, old.first_key,
                             old.mixed_shapes or old.first_key != sample.key)
        if len(self._costs) >= self.max_keys:
            del self._costs[next(iter(self._costs))]
        self._costs[group] = value
        self.accepted += 1
        return True

    def predict(self, key):
        if not self.enabled:
            return None
        value = self._costs.get(self.group_key(key))
        if (value is None or value.samples < self.min_samples
                or self.clock() - value.updated_at > self.ttl_s):
            return None
        return value.mean_ms * self._units(key)

    def prediction(self, key):
        """请求边界读取来源；桶内成本率是近似，不能当作精确形状实测。"""
        ms = self.predict(key)
        if ms is None:
            return None
        value = self._costs[self.group_key(key)]
        return {'duration_ms': ms, 'samples': value.samples,
                'source': 'shape_bucket' if value.mixed_shapes or value.first_key != key else 'exact'}

    def snapshot(self):
        """只在请求边界导出；不在计时热路径做 JSON 序列化。"""
        from dataclasses import asdict
        return [{'group_key': asdict(key), 'ema_cost_per_unit': value.mean_ms,
                 'samples': value.samples, 'updated_at': value.updated_at,
                 'first_key': asdict(value.first_key), 'mixed_shapes': value.mixed_shapes}
                for key, value in self._costs.items()]
