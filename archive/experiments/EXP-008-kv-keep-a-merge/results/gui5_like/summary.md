# EXP-008 KV keep(A)+任务关注度 + history merge — 结果摘要

- 时间: 2026-11-03 10:57:16 CST（Asia/Shanghai）
- **合成 KV**: `True`（无真实 dump；特征维=64 代理；字节按 Qwen2.5-VL-3B 公式）
- tokens in→out: 1064 → 344 （压缩比 0.323）
- MiB in→out: 37.406 → 12.094 （压缩比 0.323）
- action_prev 未破坏: `True`
- 策略耗时中位: 1.070 ms （p90 1.168）
- + host memcpy(out bytes): ≈ 1.281 ms
- NPU DMA: stub（未实现）

## vs A2O 预算（EXP-005）

| 框架 | 预算 ms | prep+memcpy ms | 落入空档 |
|---|---:|---:|---|
| UI-TARS-desktop (ADB) | 1345.4 | 1.281 | 是 |
| Mobile-Agent-v1 | 2546.6 | 1.281 | 是 |
| Mobile-Agent-v3 | 3546.6 | 1.281 | 是 |
| AndroidLab | 5703.6 | 1.281 | 是 |
| Mobile-Agent-v2/E | 6546.6 | 1.281 | 是 |
| AppAgent | 12703.6 | 1.281 | 是 |
| AndroidWorld M3A (wait only) | 2000.0 | 1.281 | 是 |

## 输出文件
- `kv_keep_merge_summary.json`
- `dispositions.csv`
- `timings.csv`
- `a2o_comparison.csv`
