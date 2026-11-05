"""从正常请求逐步建立成本；不额外执行模型，不持有第二份成本表。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestWarmup:
    enabled: bool = True

    def state(self, samples, required):
        if not self.enabled:
            return 'disabled'
        if samples == 0:
            return 'cold'
        return 'ready' if samples >= required else 'collecting'

    def needs_sample(self, samples, required):
        return self.enabled and samples < required
