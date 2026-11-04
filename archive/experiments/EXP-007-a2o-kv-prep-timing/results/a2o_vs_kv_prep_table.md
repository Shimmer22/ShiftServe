# EXP-007 对照表: A2O 预算 vs KV prep 耗时（上界）

- 测量时间: 2026-11-03 10:47:05 CST
- Host: macOS-26.6.2-arm64-arm-64bit（numpy page-aligned memcpy；**非** Hexagon DMA）
- Phone: adb `10.29.11.213:42629`；用户态 memcpy microbench； Hexagon DMA=**否**；安全注记: llama_running=False MemAvailable_MiB=5239.4
- 模型体量公式: `2 * 36 * 2 * 128 * 2 * n_tok` bytes（f16）；ctx4096 → 144.00 MiB，与 EXP-006 `HTP0 KV buffer size = 144.00 MiB` 一致。
- prep = `organize_trunc`（按 token 截断的逐层前缀拷贝）+ `move_page_aligned`（搬到页对齐目标 buffer）。 这是拷贝上界，**不是**真实 GUI-KV prune/merge 算法。
- HTP0-REPACK≈1488 MiB 是权重 repack，不是 KV；本表不拿它当搬移体量。

## Host prep 摘要

| 场景 | src MiB | keep MiB | trunc 中位 ms | move 中位 ms | prep 中位 ms | prep p90 ms |
|---|---:|---:|---:|---:|---:|---:|
| gui_1step_M | 21.094 | 21.094 | 0.3815 | 0.3616 | 0.743 | 0.8233 |
| gui_5step_keep_2 | 105.469 | 42.188 | 0.7345 | 0.7277 | 1.4622 | 1.5238 |
| ctx_2048_full | 72.0 | 72.0 | 1.31 | 1.3294 | 2.6394 | 2.8115 |
| ctx_4096_full | 144.0 | 144.0 | 2.712 | 2.6928 | 5.4048 | 5.638 |

## 各框架 A2O 预算 vs prep（是否落在预算内）

| 框架 | A2O 预算 ms | vs gui_1step prep | vs gui_5step_keep_2 prep | vs ctx4096 prep | 备注 |
|---|---:|---|---|---|---|
| UI-TARS-desktop (ADB) | 1345.4 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |
| Mobile-Agent-v1 | 2546.6 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |
| Mobile-Agent-v3 | 3546.6 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |
| AndroidLab | 5703.6 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |
| Mobile-Agent-v2/E | 6546.6 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |
| AppAgent | 12703.6 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |
| AndroidWorld M3A (wait only) | 2000.0 | 是 (0.743ms) | 是 (1.4622ms) | 是 (5.4048ms) | EXP-005 推算预算 |

## Phone memcpy（同尺寸量级探测；非 DMA）

| 场景 | MiB | per_ms | GiB/s | skipped | 说明 |
|---|---:|---:|---:|---|---|
| phone_1MiB | 1.0 | 0.108638 | 8.989 | False | userspace_memcpy_page_aligned |
| phone_4MiB | 4.0 | 0.318724 | 12.256 | False | userspace_memcpy_page_aligned |
| phone_16MiB | 16.0 | 1.669031 | 9.362 | False | userspace_memcpy_page_aligned |
| phone_36MiB_approx_1k_tok | 36.0 | 2.680937 | 13.113 | False | userspace_memcpy_page_aligned |
| phone_72MiB_approx_2k_tok | 72.0 | 5.769218 | 12.188 | False | userspace_memcpy_page_aligned |
| phone_144MiB_ctx4096 | 144.0 | 11.226875 | 12.526 | False | userspace_memcpy_page_aligned |

## Phone 推导的 prep 上界（单次 memcpy 实测 ×2；仍非 DMA）

- 手机 144MiB 页对齐 memcpy 实测 per_ms=11.226875（GiB/s=12.526）。
- 72MiB per_ms=5.769218；36MiB per_ms=2.680937。
- 假设 organize_trunc≈拷贝 keep 字节、move≈再拷贝 keep 字节 → prep_est ≈ 2 × memcpy(keep)。

| 场景 | keep MiB | phone prep_est ms | host prep 中位 ms | 最短 A2O(UI-TARS) 1345ms 内? |
|---|---:|---:|---:|---|
| gui_1step_M | 21.094 | 3.289 | 0.743 | 是 |
| gui_5step_keep_2 | 42.188 | 6.578 | 1.4622 | 是 |
| ctx_2048_full | 72.0 | 11.227 | 2.6394 | 是 |
| ctx_4096_full | 144.0 | 22.454 | 5.4048 | 是 |

## 关键路径增量（解释）

若把 prep 放进动作后空档：在 host 拷贝上界下，gui 单步/多步截断 prep 远小于最短 A2O（UI-TARS≈1.3s）；满 ctx4096 双倍拷贝量级仍通常远小于多数框架预算。真正上关键路径时还会叠加调度/锁/与截图并发，以及真实 prune 算分与 Hexagon 侧布局转换——那些**未测**，故本表只回答「纯整理+搬移拷贝上界 vs A2O」。
