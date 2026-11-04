# EXP-007: A2O 空档内 KV 整理+搬移耗时 vs A2O 预算

## 问题
GUI Agent 动作后固定等待+截图（A2O，见 EXP-005）期间 NPU 常空闲。想用这段空档做历史 KV 的整理（选/并/淘汰）并把必要 KV 搬成 NPU 可读布局，使下一张图到达即可 NPU prefill。本实验只回答：**KV 整理+搬移的拷贝上界耗时，是否落在各框架 A2O 预算内**；不把 A2O 去和整段 NPU 推理比。

## 方法
- **体量**：Qwen2.5-VL-3B LLM KV（f16）  
  `bytes = 2(K+V) × 36 × 2(kv_heads) × 128 × 2 × n_tok`  
  ctx4096 → **144.00 MiB**，与 EXP-006 日志 `HTP0 KV buffer size = 144.00 MiB` 一致。  
  `HTP0-REPACK≈1488 MiB` 是权重 repack，不是 KV，不作搬移体量。
- **organize 上界（模拟，非真实 prune 算法）**  
  - 整段 memcpy  
  - 按 token 截断：逐层拷贝前缀（`keep_frac`）  
  - 隔层选择：偶数层全量拷贝（半量字节）
- **move 上界**：页对齐目标 buffer 的 host/phone userspace `memcpy`（**非 Hexagon DMA**）。
- **prep**：`organize_trunc + move_page_aligned`。
- **场景**：GUI 1/3/5 步历史量级、ctx 512/1024/2048/4096。
- **设备**：Mac host（conda `visiontrim-qwen25vl`）；手机 Xiaomi 15，adb `10.29.11.213:42629`，短跑 `memcpy_bench`（NDK PIE）。跑前检查 `llama` 进程与 `MemAvailable`；有长作业则跳过大 buffer。
- **未改**：CPU 频率、手机系统设置；未重编译 llama.cpp；未下载大模型；大文件未拷入本目录。

## 结果（摘要）
详见 `results/a2o_vs_kv_prep_table.md`、`host_kv_prep_timings.csv`、`phone_memcpy_timings.csv`、`kv_prep_summary.json`。

| 场景 | keep MiB | Host prep 中位 ms | Phone prep_est≈2×memcpy ms |
|---|---:|---:|---:|
| gui_1step_M (~600 tok) | 21.1 | ~0.7 | ~3.3 |
| gui_5step_keep_2 | 42.2 | ~1.5 | ~6.6 |
| ctx_2048_full | 72.0 | ~2.6 | ~11.2 |
| ctx_4096_full | 144.0 | ~5.4 | ~22.5 |

A2O 预算（EXP-005）：约 **1345–12704 ms**。上表 prep 均 **远小于** 最短预算（UI-TARS≈1.3s）。手机 144MiB 单次页对齐 memcpy ≈ **11.2 ms**（~12.5 GiB/s）。

## 复现
```bash
conda run -n visiontrim-qwen25vl python scripts/kv_prep_bench.py \
  --out-dir results --repeats 15 --device 10.29.11.213:42629
# 可选跳过手机: --skip-phone
# 手机 binary: scripts/native/memcpy_bench_android （源码 memcpy_bench.c，NDK aarch64 clang -fPIC -pie）
```

## 局限 / 未验证
1. **未接入真实 GUI-KV prune/merge 算分**；organize 仅为字节拷贝上界。
2. **未测真实 Hexagon DMA / fastrpc / HTP 侧布局转换**；phone 仅为 bionic userspace memcpy。
3. 未测与截图/UI 树采集并发时的调度干扰；未测放进 agent 关键路径后的端到端增量。
4. KV 布局假设为按层连续、按 token 前缀可截断；与 llama.cpp 实际 KV 槽位布局若有差异，拷贝次数可能更多。
5. 手机大 buffer 次数少（144MiB 仅 1 次迭代）；内存紧张或存在 llama 长作业时会自动跳过。

## 清理
手机仅短暂推送 `/data/local/tmp/a2o_exp007_memcpy_bench`，脚本结束已删除。
