"""阶段级抽样计时，不读取张量、日志或设备计数器。"""
from .sampling import CostKey, Sample, SamplingProfiler

__all__ = ['CostKey', 'Sample', 'SamplingProfiler']
