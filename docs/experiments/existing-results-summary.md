# 已有实验结果梳理与数据基线

> 整理日期：2026-11-19 · 所有数值均直接提取自本地 `results/` 原始 JSON/CSV 实测产物。

本文档按论文论证结构整理当前已完成的真机与主机实验，标明数据来源、关键结论与已知口径限制，供撰文、绘图及后续补测直接引用。

---

## 一、动机与硬件特性证据（Motivation）

### 1.1 CPU 短输入锯齿反转与 HTP 门槛
- **锯齿机理建模**：[sawtooth summary](../../results/20261111-cpu-prefill-sawtooth-audit/summary.json)
  - `Q4_0` ARM `I8MM` 双行 `vec_dot`：当 `ne11`（或实际切块行列）为奇数时回退单行路径。
  - 回归拟合 `time_s = 0.1590 + 0.06378 * P + 0.02905 * single_row_tokens`（$R^2 = 0.9984$，`MAPE = 1.09%`，留出测试 `P > 48` 处 `MAPE = 1.13%`）。
- **SM8650 `H=4096` 实测（CPU6/HTP4）**：[p24-128 geometric](../../results/20261111-prefill-p24-128-geometric-cpu6-t10/summary.json)
  - `P=24`：CPU 中位数 `1,720.2 ms`，HTP `4,235.8 ms`（CPU 占优）；
  - `P=31 / 32 / 33`：CPU 呈现剧烈锯齿 `3,165.4 ms → 2,265.1 ms → 3,306.9 ms`；
  - `P=96`：CPU `6,383.5 ms` vs HTP `6,434.9 ms`；`P=128`：HTP `6,436.5 ms` 显著反超 CPU `8,369.0 ms`。
- **SM8650 Qwen3-4B 多 `H` Oracle 真值网格（`H=6144` 示例）**：[raw_multi_h_truth.json](../../results/20261117-m1-explorer-large-campaign/oracle-grid/raw_multi_h_truth.json)
  - CPU Prefill 随 `P` 剧烈起伏：`P=24: 18,027.8 ms` → `P=32: 3,236.4 ms` → `P=40: 13,711.2 ms` → `P=64: 5,006.3 ms` → `P=80: 21,163.2 ms` → `P=96: 8,242.0 ms` → `P=128: 23,838.2 ms`；
  - 同点 HTP 含回迁总耗时分别为 `5,376.6 / 4,982.6 / 6,244.9 / 6,093.2 / 8,992.7 / 8,133.2 / 11,199.8 ms`，最优后端随 `(H, P)` 频繁翻转。

### 1.2 长历史推高 Decode 与跨后端代价
- **Qwen3-8B `H` 扫描**（`ctx=8704, D=16, CPU6/HTP4`）：[qwen3-history-decode](../../results/20261114-qwen3-history-decode/summary.json)

| 历史长度 `H` | CPU Decode (`ms/tok`) | HTP Decode (`ms/tok`) |
|---:|---:|---:|
| 128 | 129.31 | 199.09 |
| 256 | 144.52 | 222.85 |
| 512 | 154.03 | 251.73 |
| 1024 | 177.07 | 259.92 |
| 2048 | 192.71 | 304.75 |
| 4096 | 224.24 | 340.52 |
| 8192 | 277.32 | 398.37 |

### 1.3 双份权重 + 长上下文触发 Swap 反噬 `M1-only`
- **`v3` τ²-bench 4 题 M1-only 复测**：[tau2-qwen3-4b-m1-v3-retest](../../results/20261117-tau2-qwen3-4b-m1-v3-retest/summary.json)
  - 4 题通过 `2/4`，服务耗时 `4,605.2 s`（66 轮均为 `H>=4k` 长上下文，平均 `restore = 1,303.5 ms`、`handoff = 1,233.1 ms`、`decode = 258.8 ms/t`）；实验结束后手机进程 `VmSwap` 高达 **`2,854,992 kB (~2,788 MiB)`**（启动前 `0 kB`）。
- **Retail 精确冻结回放（含内存页错采样）**：[retail-memory summary](../../results/20261118-qwen3-4b-exact-frozen-retail-memory/summary.json)
  - `cpu_baseline`：Warm `182.13 s`，`357.49 ms/t`，`0` major faults；
  - `cpu_adaptive_compact`：Warm `138.14 s`，`238.56 ms/t`，`0` major faults；
  - `route_m1_only`：因双权重压力触发 **`4,599` 次 major faults**，Decode 恶化至 `437.77 ms/t`，Warm 拖慢至 **`213.70 s`（慢于纯 CPU）**；
  - `route_m1_m2_compact`：major faults 降至 `1,697` 次，Warm **`133.71 s`（全场最快）**。
- **串行上下文（`serial_context`）补充对照**：[context-cc-frozen](../../results/20261118-qwen3-4b-context-cc-frozen/summary.json)
  - `retail-short`：`cpu_baseline 43.42 s (0 faults)` vs `serial_context 42.56 s (0 faults)` vs `route_m1_only 60.24 s (2,154 faults)`；
  - `telecom-long`：`cpu_baseline 72.22 s (0 faults)` vs `serial_context 87.95 s (1,669 faults)` vs `route_m1_only 95.74 s (7,464 faults)`。

---

## 二、4B 多 Benchmark 冻结回放主消融 + M2 采集税优化验收

### 2.1 19 条轨迹 / 64 个 Warm 轮次 5 路消融（旧版先验采集口径）
来源：[20261118-multibench-frozen-ablation/summary.json](../../results/20261118-multibench-frozen-ablation/summary.json)（Qwen3-4B Q4\_0，`ctx=9472`，`gap_ms=1000`，共 19 条轨迹、64 个 warm 轮次）。

| 条件 | 工具返回后合计 | 含首轮合计 | Prefill | Decode (`ms/t`) | Handoff | Wait | Restore | 峰值 RSS / Swap (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_cpu` | 1,514.1 s | 9,424.4 s | 783.3 s | 661.6 s (216.1) | 0.0 s | 0.0 s | 43.6 s | 5,393.7 / 0.0 |
| `shared_cc` | 1,395.8 s (−7.8%) | 10,001.9 s | 746.1 s | 581.9 s (190.1) | 0.0 s | 0.0 s | 41.8 s | 3,483.4 / 0.0 |
| `shared_m1` | 1,264.8 s (−16.5%) | 10,653.2 s | 544.7 s | 614.3 s (200.7) | 63.0 s | 13.5 s | 4.3 s | 3,463.1 / 0.0 |
| **`shared_recent`** | **1,175.3 s (−22.4%)** | **9,614.3 s** | **513.3 s** | **569.2 s (185.9)** | **59.3 s** | 6.9 s | 3.8 s | 3,336.7 / 0.0 |
| `shared_prior` *(旧版未优化)* | 1,353.3 s (−10.6%) | 10,606.0 s | 637.6 s | 616.8 s (201.5) | 62.5 s | 9.0 s | 3.7 s | 3,134.0 / **389.6** |

分 Benchmark 工具返回后合计（`post_return_s` / 总 Handoff 量）：

| Benchmark (轨迹 / Warm 轮) | `full_cpu` | `shared_cc` | `shared_m1` | `shared_recent` | `shared_prior` *(旧版)* |
|---|---:|---:|---:|---:|---:|
| `tau2-airline` (5 / 10) | 263.9 s | 237.5 s | 201.0 s (7,352 MB) | 187.6 s (6,869 MB) | 214.1 s (6,869 MB) |
| `tau2-retail` (3 / 6) | 255.1 s | 247.5 s | 197.5 s (5,702 MB) | 192.6 s (4,665 MB) | 216.4 s (4,665 MB) |
| `tau2-telecom` (6 / 39) | 911.4 s | 839.7 s | 813.8 s (41,831 MB) | 742.7 s (39,436 MB) | 861.1 s (39,436 MB) |
| `toolsandbox` (5 / 9) | 83.8 s | 71.1 s | 52.5 s (743 MB) | 52.3 s (604 MB) | 61.6 s (604 MB) |

> **旧版 `shared_prior` 落后原因诊断**：(1) `evaluate_with_prior_tail` 强制拆出尾段 `16` 个 token 多跑一遍 36 层小 batch HTP 图；(2) 全 36 层同步回调 `kq_soft_max` 带来每轮 ~1.3–3.6 s 采集税（64 轮 Prefill 多花 `+124.3 s`）；(3) 57/64 轮在 `pinned_tokens` 限制下仅可裁剪 `<256` tokens，无准入过滤；(4) 排在末棒唯一触发 `389.6 MiB` Swap（Decode 多花 `+47.6 s`）。

### 2.2 `179acb1` 优化后真机验证（单 batch 不拆图 + `kv_layer_stride=4` + `kv_min_evict_tokens=256`）
- **成对双用例复测**：[20261119-prior-opt-verify](../../results/20261119-prior-opt-verify/summary.json)
  - `airline-c6f3ac57944a-9`：T01（可淘汰 `558` tokens，实际开启先验）Prefill 采集税从 `+1,621 ms` 降至 **`+229 ms`（−85.9%）**；T02（仅 `57` tokens）自动退化 `recent`；2 轮 Warm 合计 **`28.67 s` vs 同机 `shared_recent` `28.56 s`（+0.38%）**。
  - `send_message_with_contact_content_cellular_off`：4 轮均 `<256` 自动退化 `recent`，Prefill 从 `15.96 s` 降至 **`10.50 s`（−34.2%）**，Warm 合计 `20.97 s`（同机 `shared_recent` `22.68 s`）。
- **三路定点冻结回放**：[20261119-prior-opt-targeted](../../results/20261119-prior-opt-targeted/summary.json)（`retail` 末路因 worktree 合并删除在 T0 中断，前两条三路完整）：
  - `telecom-52b4b12a5622-10`（混合场）：T01（`P=293, evict=848` 触发 `prefill_prior`）`38.01 s` 优于 `shared_recent 38.30 s` 与 `shared_m1 40.32 s`；T02–T04 自动退化 `recent`；4 轮 Warm 合计 **`92.97 s`**（`shared_m1 95.57 s`，`shared_recent 89.54 s`）。
  - `send_message_with_contact_content_cellular_off`：4 轮 Warm 合计 **`18.85 s`**（`shared_m1 19.06 s`，`shared_recent 22.53 s`，较优化前 `28.21 s` 提速 **−33.2%**）。

---

## 三、4B 早期成对冻结 & 端到端对照（含跨模型 9B）

### 3.1 HTP 逻辑裁剪融合回迁微观验证（6 样本，`v3` 双权重）
来源：[20261117-qwen3-4b-frozen-replay/summary.json](../../results/20261117-qwen3-4b-frozen-replay/summary.json)（`gap_ms=2500`，`n=6`）。

| 条件 | HTP 选路 | 平均 Decode KV | Restore | Prefill | KV 裁剪 | Handoff | Decode (`ms/t`) | **平均 Post-Return** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu_baseline` | 0/6 | 7,951.3 | 741.6 ms | 83.74 s | 0.0 ms | 0.0 ms | 12.70 s (262.8) | **97.19 s** |
| `cpu_adaptive_compact` | 0/6 | 2,133.3 | 886.4 ms | 84.41 s | **1,267.6 ms** | 0.0 ms | 7.51 s (155.3) | **94.10 s (−3.2%)** |
| `route_m1_only` | 5/6 | 7,951.3 | 358.7 ms | 58.37 s | 0.0 ms | **3,726.0 ms** | 13.68 s (283.1) | **76.53 s (−21.3%)** |
| **`route_m1_m2_compact`** | **6/6** | **2,133.3** | 0.0 ms | **57.78 s** | **4.2 ms** | **845.3 ms (−77.3%)** | **8.69 s (179.7)** | **67.64 s (−30.4%)** |

> 关键发现：M2 将逻辑裁剪（`4.2 ms`）搭在 HC 状态回迁顺风车上，避免 CPU 侧额外物理重排（`1,267.6 ms`），同时将回迁耗时削减 **77.3%**。

### 3.2 τ²-bench 9 题端到端实测（Qwen3-4B，`ctx=14336`）
来源：[20261116-tau2-qwen3-4b-joint-v2-e2e](../../results/20261116-tau2-qwen3-4b-joint-v2-e2e/summary.json)（及 `conditions/*/worker-summary.json`、`phone-memory.json`）。

| 条件 | 完成 / 通过 | Agent 轮次 (Warm) | 总服务耗时 | Warm 整轮均值 | Warm Prefill 均值 | Decode (`ms/t`) | 结束时 `VmSwap` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cpu_baseline` | 9 / 4 | 114 (105) | 8,840.0 s | 47.7 s | 29.4 s | 262.2 | 0 kB |
| `cpu_adaptive_compact` | 9 / 3 | 156 (147) | 9,961.2 s | 42.2 s | 25.3 s | 229.6 | 0 kB |
| `route_m1_only` *(未完结)* | — | — | — | — | — | — | **4,707,168 kB** |
| **`route_m1_m2_compact`** | **9 / 5** | 138 (129) | **8,542.3 s** | **35.7 s** | **15.3 s** | 295.8 | 387,228 kB |

### 3.3 ToolSandbox 10 题跨模型路由实测（Qwen3.5-9B，`ctx=4096`，关闭 KV 裁剪）
来源：[20261107-toolsandbox-route-e2e/report.md](../../results/20261107-toolsandbox-route-e2e/report.md)
- `cpu`：完成 `10/10`，逐题 E2E 中位数 **`97.23 s`**；
- `htp`：轨迹完整 `9/10`，逐题 E2E 中位数 **`92.91 s`**；
- `route`：轨迹完整 `9/10`，逐题 E2E 中位数 **`86.44 s`**（全部 9 个完整题上 `route/CPU` 均为 `0.736–0.978`）。

---

## 四、M1-Explorer 在线贝叶斯路由

来源：[oracle_summary.json](../../results/20261117-m1-explorer-large-campaign/oracle-grid/oracle_summary.json) 与 [e2e_oracle_comparison.json](../../results/20261117-m1-explorer-large-campaign/e2e_oracle_comparison.json)

1. **49 点 `(H, P)` Oracle 网格逼近率**（30 个多 `H∈{2048,6144,9216}` 点 + 19 个 `H=4096` 点）：
   - `static_tiny`：**84.95%** (`0.8495`)
   - `online_greedy`：**88.35%** (`0.8835`)
   - **`m1_explorer`**：**89.08%** (`0.8908`)
2. **τ²-bench 9 题 E2E 与 `full_htp` / `Oracle` 对照**（实测总计）：

| 条件 | 任务数 | Reward 合计 | 总服务耗时 | Warm 整轮均值 | Warm Prefill 均值 | Decode (`ms/t`) |
|---|---:|---:|---:|---:|---:|---:|
| `full_cpu_baseline` | 9 | 4.0 | 8,840.0 s | 45.41 s | 28.63 s | 250.1 |
| `full_cpu_compact` | 9 | 3.0 | 9,961.2 s | 39.77 s | 24.60 s | 225.7 |
| `full_htp` | 9 | 3.0 | 8,641.3 s | 43.68 s | 18.78 s | 327.8 |
| `m1_explorer_m2` | 9 | **5.0** | 9,031.6 s | **39.06 s** | **15.06 s** | 338.2 |
| `oracle_per_task_best` | 9 | — | 5,716.7 s | 31.24 s | — | — |

---

## 五、主机大规模 KV 准确率评测（`20261118-host-kv-accuracy`）

注：以下已完成组均使用优化前主干二进制在 `wins` RTX 4090 上运行（`kv_layer_stride=1`）；其中 `tau2-ratio-8b` 当前**仍在后台运行中**（`m2_prior_low` 进行中）。

| 评测集 | 模型 | 指标口径 | `full_kv` | `recent_only` | `m2_prior_low` | `m2_prior_mid` | `m2_prior_high` | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---|
| [bfcl](../../results/20261118-host-kv-accuracy/bfcl/summary.json) | 4B | 比例 `0.25/0.5/0.75`，60 题 `accuracy` | **0.1000** (6/60) | 0.0500 (3/60) | 0.0167 (1/60) | 0.0667 (4/60) | **0.0833** (5/60) | 已完成 |
| [bfcl-8b](../../results/20261118-host-kv-accuracy/bfcl-8b/summary.json) | 8B | 比例 `0.25/0.5/0.75`，60 题 `accuracy` | **0.1000** (6/60) | 0.0500 (3/60) | 0.0667 (4/60) | 0.0667 (4/60) | **0.0833** (5/60) | 已完成 |
| [toolsandbox](../../results/20261118-host-kv-accuracy/toolsandbox/summary.json) | 4B | 比例 `0.25/0.5/0.75`，28 题 `mean_similarity` | **0.1161** | 0.1157 | 0.1157 | 0.1156 | 0.1156 | 已完成 |
| [toolsandbox-8b](../../results/20261118-host-kv-accuracy/toolsandbox-8b/summary.json) | 8B | 比例 `0.25/0.5/0.75`，28 题 `mean_similarity` | 0.2357 | 0.2233 | 0.2333 | 0.2291 | **0.2527** | 已完成 |
| [tau2](../../results/20261118-host-kv-accuracy/tau2/summary.json) | 4B | 固定 `1536/3072/4608`，36 题 `reward_mean` | **0.1111** (4/36) | 0.0556 (2/36) | 0.0556 (2/36) | 0.0556 (2/36) | 0.0556 (2/36) | 已完成 |
| [tau2-ratio](../../results/20261118-host-kv-accuracy/tau2-ratio/summary.json) | 4B | 比例 `0.25/0.5/0.75`，36 题 `reward_mean` | 0.0556 (2/36) | 0.0833 (3/36) | 0.0278 (1/36) | 0.0556 (2/36) | 0.0000 (0/36) | 已完成 |
| [tau2-ratio-8b](../../results/20261118-host-kv-accuracy/tau2-ratio-8b/summary.json) | 8B | 比例 `0.25/0.5/0.75`，36 题 `reward_mean` | 0.1111 (4/36) | *运行中* | *运行中* | *排队中* | *排队中* | **后台运行中** |

> **待对齐项**：需在 `tau2-ratio-8b` 完成后用 `03ca4e4` / `179acb1` 重建主机版 `phone_service_host`（启用 `kv_layer_stride=4` 与 `kv_min_evict_tokens=256`），在 BFCL 上补验稀疏层采样是否保持准确率不掉点。

---

## 六、8B 共享权重与内存底座专项

- **Qwen3-8B Q4\_0 共享权重冻结短链（2 轮对照）**：[8b-strict-cpu](../../results/20261118-qwen3-8b-strict-cpu-frozen/summary.json) / [8b-joint-retry](../../results/20261118-htp-shared-weights-vector-8b-joint-retry/summary.json)
  - `full_cpu`（单份 CPU）：Post-return **`43,881.9 ms`**，`198.2 ms/t`，进程 `VmRSS = 4,993 MiB`（`Swap = 0`）；
  - `shared_cc`（共享权重 CPU）：Post-return **`34,743.9 ms`**，`170.8 ms/t`，`VmRSS = 1,320 MiB`（`Swap = 0`）；
  - `shared_m1`（共享权重 + M1）：Post-return **`21,337.4 ms`**，`173.7 ms/t`，`VmRSS = 1,322 MiB`（`Swap = 0`）；
  - `shared_joint`（共享权重 + M1 + 旧版 M2）：Post-return `23,634.9 ms`（因旧版先验采集税导致 prefill `5,744 ms` 慢于 `shared_m1` `3,884 ms`），Decode 速度最优 **`160.1 ms/t`**，`VmRSS = 1,232 MiB`（`Swap = 0`）。需用 `179acb1` 封版在长历史 8B 轨迹上重测。
- **HWM 脏页回收（`kv_hwm_trim`）与 Telecom 3-way 探测**：
  - [20261118-telecom-swap-3way](../../results/20261118-telecom-swap-3way/m1_m2_no_hwm_trim.json) 已保留 `cpu_baseline`（T0–T2 `VmRSS 5,712 → 6,371 MiB, VmSwap=0`）和 `m1_m2_no_hwm_trim`（`VmRSS 6,122 → 6,191 MiB, VmSwap=0`）；第三组 `with_hwm_trim` 当时因服务端连接重置尚未写入汇总 JSON。
