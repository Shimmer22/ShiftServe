"""过去 P 的滑动直方图与投机准备期望价值；不读取本轮返回长度。"""
from collections import Counter, deque
import math


class PDistribution:
    def __init__(self, window=32):
        if window < 1:
            raise ValueError('window must be positive')
        self.history = deque(maxlen=window)

    def observe(self, p):
        if type(p) is not int or p < 1:
            raise ValueError('P must be a positive integer')
        self.history.append(p)

    def predict(self):
        counts = Counter(self.history)
        return {p: n / len(self.history) for p, n in counts.items()}


def expected_value(distribution, costs, prepare_ms, gap_ms, *, weight=1., min_gain_ms=0.):
    """costs[P]=(CC, ready HC, qualified)。目标=响应ms+weight*投机准备预算ms（不等同总设备时间）。

    固定 Gap 的 ready-only 基线；未覆盖质量不外推，返回 abstain。
    不准备对照允许返回后按需准备，与现有 post-tool routing 一致。
    """
    if any(not math.isfinite(x) or x < 0 for x in (prepare_ms, gap_ms, weight, min_gain_ms)):
        raise ValueError('invalid cost parameters')
    if not distribution:
        return dict(prepare=False, reason='cold_start', expected_gain_ms=None)
    if any(p not in costs for p in distribution):
        return dict(prepare=False, reason='coverage_missing', expected_gain_ms=None)
    if any(not math.isfinite(v) or v < 0 for v in distribution.values()) or not math.isclose(sum(distribution.values()), 1.):
        raise ValueError('invalid probability mass')
    gain = 0.
    for p, prob in distribution.items():
        cc, hc, qualified = costs[p]
        if any(not math.isfinite(x) or x < 0 for x in (cc, hc)):
            raise ValueError('invalid route costs')
        late = min(cc, prepare_ms + hc) if qualified else cc
        early = min(cc, hc) if qualified and gap_ms >= prepare_ms else cc
        gain += prob * (late - early)
    net = gain - weight * prepare_ms
    return dict(prepare=net > min_gain_ms, reason='positive_value' if net > min_gain_ms else 'nonpositive_value',
                expected_gain_ms=gain, prepare_cost_ms=prepare_ms, weight=weight,
                expected_net_ms=net, gap_estimate_ms=gap_ms, distribution=distribution)
