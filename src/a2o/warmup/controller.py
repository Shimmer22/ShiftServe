"""可选初始化收敛工具，与正常请求自预热分开；默认关闭。"""
from collections import deque
from dataclasses import dataclass
import math
from time import perf_counter_ns


@dataclass(frozen=True)
class WarmupConfig:
    enabled: bool = False
    max_steps: int = 8
    budget_ms: float = 500
    window: int = 3
    relative_spread: float = 0.05

    def __post_init__(self):
        if self.max_steps < 1 or self.window < 2 or self.window > self.max_steps:
            raise ValueError('require max_steps >= window >= 2')
        if (not math.isfinite(self.budget_ms) or self.budget_ms < 0
                or not math.isfinite(self.relative_spread) or self.relative_spread < 0):
            raise ValueError('invalid warmup budget or spread')


@dataclass(frozen=True)
class WarmupResult:
    reason: str
    steps: int
    elapsed_ms: float
    durations_ms: tuple
    overrun_ms: float = 0


class WarmupController:
    def __init__(self, config=None, *, clock_ns=perf_counter_ns):
        self.config = config or WarmupConfig()
        self.clock_ns = clock_ns

    def run(self, step):
        """step 在同一常驻上下文中完成一轮预热；异常直接传播，不切换后端。"""
        cfg = self.config
        if not cfg.enabled:
            return WarmupResult('disabled', 0, 0, ())
        start = self.clock_ns()
        durations, recent = [], deque(maxlen=cfg.window)
        reason = 'max_steps'
        elapsed = 0.
        for _ in range(cfg.max_steps):
            elapsed = (self.clock_ns() - start) / 1e6
            # API 不可抢占；按最近最慢步估算下一步，首步也可能超预算。
            if elapsed >= cfg.budget_ms or (recent and elapsed + max(recent) > cfg.budget_ms):
                reason = 'budget'
                break
            before = self.clock_ns()
            step()
            after = self.clock_ns()
            duration = (after - before) / 1e6
            durations.append(duration)
            recent.append(duration)
            elapsed = (after - start) / 1e6
            if elapsed >= cfg.budget_ms:
                reason = 'budget'
                break
            if len(recent) == cfg.window:
                mean = sum(recent) / cfg.window
                if max(recent) - min(recent) <= cfg.relative_spread * mean:
                    reason = 'stable'
                    break
        return WarmupResult(reason, len(durations), elapsed, tuple(durations),
                            max(0., elapsed - cfg.budget_ms))
