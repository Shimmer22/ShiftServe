# Qwen3-1.7B 机制 2：时间-历史压缩曲线、成本表抽象与端到端实测

- 模型与资产：`Qwen3-1.7B-Q4_0`（`/data/data/com.termux/files/home/.local/a2o/models/Qwen3-1.7B-Q4_0.gguf`，SHA-256：`5c71ea792028698598657d8f3e13fcf1980fcb2e1887c4931c5a129a5c8b488f`）
- 性质边界说明：多轮对话中 `<think>\n\n</think>\n\n` 与尾部标记导致的 `H_anchor` KV 前缀失效属于工程 Bug 修复，不计入机制 2 核心贡献；本文档评估机制 2（`PrefillPrior` 指导的 KV Reduce + 成本准入 + 状态紧凑锚点）的定量收益与端到端表现。

## 1. 第 1 步：时间 / 历史长度 / 保留预算 / 质量曲线扫描

数据来源：[summary.json](../../results/20261115-qwen3-1_7b-kv-curve/summary.json)（72 次网格计时 + 36 次质量边界测试 + CPU 探针）。

![时间-历史长度曲线与成本准入边界](../../results/20261115-qwen3-1_7b-kv-curve/figures/fig1-time-deadzone-curves.png)

![质量拐点与保留比例-速度权衡](../../results/20261115-qwen3-1_7b-kv-curve/figures/fig2-quality-budget-pareto.png)

### 1.1 收益与死区：固定历史与保留长度 `K` 对 Decode / Handoff 的影响

固定 `P = 256`，比较无裁剪直传（`hc_nocb`）与启用 `PrefillPrior` 裁剪至 `K = 1024`（`hc_prior`）：

| `H` | CPU Decode (`ms/tok`) | `hc_nocb` Handoff (`ms`) | `hc_nocb` Decode (`ms/tok`) | `hc_prior(K=1024)` Prefill 增税 (`ms`) | `hc_prior` Handoff (`ms`) | `hc_prior` Decode (`ms/tok`) | 盈亏平衡 `D` (tokens) |
|---|---|---|---|---|---|---|---|
| 512 | 39.66 | 69.2 | 39.79 | -269.3 | 77.7 | 52.95 | —（负收益） |
| 1024 | 44.36 | 112.4 | 44.64 | +467.9 | 174.0 | 51.10 | None（死区） |
| 2048 | 50.47 | 194.8 | 50.61 | +1167.1 | 185.4 | 51.49 | None（死区） |
| 4096 | 61.14 | 402.3 | 72.45 | +1689.8 | 272.4 | 51.92 | 76.1 |
| 6144 | 80.82 | 616.5 | 80.17 | +1352.9 | 253.7 | 55.94 | 40.9 |
| 8192 | 93.44 | 851.7 | 96.44 | +1526.9 | 194.4 | 54.54 | 20.8 |

在固定大历史下的 `K` 扫描（`k_sweep`）同样显示单调线性关系（`decode_ms_per_tok = 44.55 + 0.00587 * decode_n_kv`，\(R^2 = 0.809\)；`handoff_ms = 117.03 + 0.811 * MiB`，\(R^2 = 0.765\)）：

- `K=512`：`48.42 ms/tok`，handoff `150.3 ms` (`56.0 MiB`)
- `K=1024`：`51.52 ms/tok`，handoff `193.8 ms` (`112.0 MiB`)
- `K=1536`：`54.52 ms/tok`，handoff `308.9 ms` (`168.0 MiB`)
- `K=2048`：`61.32 ms/tok`，handoff `422.5 ms` (`224.0 MiB`)
- `K=3072`：`64.14 ms/tok`，handoff `560.0 ms` (`336.0 MiB`)
- `K=4096`：`65.44 ms/tok`，handoff `755.1 ms` (`448.0 MiB`)
- `K=6144`：`80.20 ms/tok`，handoff `806.1 ms` (`672.1 MiB`)

### 1.2 `PrefillPrior` (`cb_eval`) 额外税与 CPU 对照探针

- 在 `H = 4096` 扫描 `P = 64 / 128 / 512`（`p_sweep`）：HTP 上附加先验回调税分别为 `+180.0 ms` / `+1119.1 ms` / `+988.6 ms`；
- 在 CPU 单驻留多轮探针（`cpu_probes`）中：
  - `H=1024`：`t1_prm = 265.8 ms` (`kv=0`) vs `270.1 ms` (`kv=1`)，`t1_dec = 50.18` vs `51.35 ms/tok`；
  - `H=4096`：`t1_prm = 538.8 ms` (`kv=0`) vs `1157.7 ms` (`kv=1`，含 `~619 ms` 回调税)，`t1_kvr = 218.3 ms`，但 `t1_dec` 从 `80.42 ms/tok` 降至 `63.95 ms/tok`；
  - `H=8192`：`t1_prm = 954.4 ms` (`kv=0`) vs `1560.8 ms` (`kv=1`)，`t1_kvr = 213.3 ms`，`t1_dec` 从 `111.98 ms/tok` 降至 `64.49 ms/tok`。

结论：**`H <= 2048` 是明确的负收益死区**（`cb_eval` 回调税高于回迁与解码节省）；只有当 `H` 跨过阈值并结合预期输出长度 `D` 后，启用 KV Reduce 才产生净收益。

### 1.3 质量退化拐点（6 个 `τ²-bench` 边界 × 6 档 `K`）

`quality_k_curve`（36 组）：

- `K=512`：`1/6` 通过关键事实（均值 LCP `4.33`，`48.36 ms/tok`），仅 b5 通过，`b0..b4` 全失败；
- `K=1024`：`4/6` 通过（均值 LCP `18.00`，`51.21 ms/tok`），`b1`（`H=6478, r=0.1417`）与 `b2`（`H=7102, r=0.1375`）失败；
- `K=1536`：`5/6` 通过（均值 LCP `18.17`，`55.32 ms/tok`），仅 `b1`（`r=0.2125`）失败；
- `K=2048`：`5/6` 通过（均值 LCP `29.00`，`62.07 ms/tok`），仍仅 `b1`（`r=0.2833`）失败；
- `K=3072 / 4096`：`4/6` 通过（`b0` 与 `b1` 在该高比例候选下反而因不同中间序列选出缺失核心字段）。

据此拟合质量安全下界：`quality_k_min = 1536`，`quality_keep_ratio_min = 0.25`，按 `align_pad = 256` 向上对齐，候选集 `[1536, 2048, 2560, 3072, 4096]`。

---

## 2. 第 2 步：KV Reduce 成本表抽象与双端实现验证

### 2.1 成本表 `kv_reduce_cost_v1` 与决策公式

由 [kv_reduce_cost.py](../../src/a2o/scheduling/kv_reduce_cost.py) 拟合生成 [qwen3-1_7b-kv-reduce-cost.json](../../configs/qwen3-1_7b-kv-reduce-cost.json)：

- `decode_per_tok`: \(41.742 + 0.006360 \cdot n_{\text{kv}}\) ms/tok
- `handoff`: \(102.907 + 0.097792 \cdot n_{\text{kv}}\) ms（HTP→CPU 状态回迁）
- `evict`: \(0.339 + 0.000275 \cdot H_{\text{total}}\) ms
- `htp_prior_tax`: \(655.688 + 1.144891 \cdot P\) ms
- `cpu_prior_tax`: \(612.705 + 0 \cdot P\) ms（CPU 侧无随 `P` 显著增长项）
- 约束与阈值：`quality_k_min = 1536`，`quality_keep_ratio_min = 0.25`，`align_pad = 256`，`min_evict_savings_tokens = 512`，`min_gain_ms = 100.0`，默认 `expected_decode_tokens = 96`。

准入规则：先计算 \(K_{\text{floor}} = \text{pad\_align\_up}(\max(K_{\min}, \lceil H_{\text{total}} \cdot r_{\min} \rceil), 256)\)；若 \(H_{\text{total}} - K_{\text{floor}} < 512\) 直接拒绝（`below_quality_or_evict_floor`）；否则枚举候选中 \(\ge K_{\text{floor}}\) 的最小安全保留档（当前单调成本下最小合法 `K` 最优），计算净收益：

\[
\Delta_{\text{gain}} = \Delta_{\text{handoff}}(\text{若需 HTP→CPU}) + D \cdot (t_{\text{dec}}(H_{\text{total}}) - t_{\text{dec}}(K)) - (\text{tax}(P) + t_{\text{evict}}(H_{\text{total}}))
\]

仅当 \(\Delta_{\text{gain}} \ge 100.0\text{ ms}\) 时准入裁剪（`cost_table_admitted`），否则拒绝（`negative_net_gain`）并跳过 `PrefillPrior` 回调以避免无意义税负。

### 2.2 原生运行时改动与单测一致性

- C++ 头与运行时：
  - [kv_reduce_cost.h](../../src/a2o/native/scheduling/kv_reduce_cost.h) 实现 `KvReduceCostTable`、`plan_kv_reduce`、JSON 解析/序列化；
  - [kv_reduce_probe.h](../../src/a2o/native/profiling/kv_reduce_probe.h) 与 [gap_replay.h](../../src/a2o/native/profiling/gap_replay.h) 支持 `adaptive_cost` 与 CPU 侧 `PrefillPrior`；
  - [phone_service.cpp](../../src/a2o/native/session/phone_service.cpp) 与 [route_phone.py](../../scripts/run/route_phone.py) 接入 `--kv-policy adaptive_cost`、`--kv-costs`、`--compact-anchor`（`compact_anchor=1` 时将本轮末尾之外已驱逐位置的锚点物理回滚重导，`H_effective` 按活跃锚点数累计）及 `/a2o/costs` 运行时开关。
- 单元测试（[kv_reduce_cost_test.cpp](../../tests/kv_reduce_cost_test.cpp)、[test_kv_reduce_cost.py](../../tests/test_kv_reduce_cost.py)）：C++ 与 Python `plan_kv_reduce` 对同一成本表输出逐字段对齐（误差 \(< 10^{-6}\)）。

### 2.3 真机实测验证（Step 2 Smoke）

来源：[step2-verify/summary.json](../../results/20261115-qwen3-1_7b-adaptive-kv/step2-verify/summary.json)（验证脚本：[kv_reduce_cost_phone.py](../../scripts/validate/kv_reduce_cost_phone.py)）：

1. **`/a2o/gap-replay` (`route`)**：
   - `H=1024`（`H_total=1254`）：拒绝（`below_quality_or_evict_floor`），`decode_n_kv=1254`；
   - `H=2048`（`H_total=2279`）：拒绝（`negative_net_gain`，预期净收益 `-421.2 ms`），`decode_n_kv=2279`；
   - `H=4096`（`H_total=4329`）：准入 `K=1536`（预期净收益 `+1030.4 ms`），实测 `evict=1.77 ms`、`decode_n_kv=1536`；
   - `H=6144`（`H_total=6379`）：准入 `K=1792`（按 `ceil(6379*0.25)` 向上对齐 256；预期 `+2300.5 ms`），`decode_n_kv=1792`。
2. **CPU `compact_anchor=1` 五轮 `/v1/chat/completions`**：
   - Turn 0 (`H_total=1025`)、Turn 1 (`H_total=2577`, 预期净收益 `+21.8 ms < 100 ms`) 均正确拒绝；
   - Turn 2 (`H_total=4429`) 准入 `K=1536`，生成后将锚点紧凑保存至 `1532` 个活跃 token（`175.7 MB`），保留 `max_pos=4428`；
   - Turn 3 (`H_anchor=4425`, `H_effective=1532`, `P=355`, `H_total=1887`)：由于复用紧凑锚点，`restore_ms=118.3 ms`，并正确判定无需重复裁剪（`below_quality_or_evict_floor`）；
   - Turn 4 (`H_anchor=4776`, `H_effective=1883`, `P=2555`, `H_total=4438`)：再次跨阈值准入 `K=1536`，`max_pos_preserved=7330`，生成连贯文本。

---

## 3. 第 3 步：`τ²-bench` 多轮端到端速度与质量实测

实验配置：[tau2-qwen3-1_7b-adaptive-e2e.json](../../configs/tau2-qwen3-1_7b-adaptive-e2e.json)（`context_size=12288`，`expected_decode_tokens=96`，Agent 在手机 CPU 运行 `Qwen3-1.7B-Q4_0`，User 模拟器与 `NLAssertions` 裁判在 `wins` RTX 运行 `Qwen3.5-9B BF16` 并显式关闭思考）；共 9 题（`retail` 3 题、`airline` 3 题、`telecom` 3 题）。结果汇总：[summary.json](../../results/20261115-tau2-qwen3-1_7b-adaptive-kv-e2e/summary.json)，脚本：[tau2_adaptive_kv_e2e.py](../../scripts/collect/tau2_adaptive_kv_e2e.py)、[tau2_e2e_worker.py](../../scripts/collect/tau2_e2e_worker.py)、[tau2_adaptive_kv_e2e.py](../../scripts/analyze/tau2_adaptive_kv_e2e.py)。

### 3.0 三个测试条件分别测什么（方法与执行路径）

三组均运行在**手机单驻留 CPU**（`condition=cpu`, `route=cpu`）并统一启用跨轮锚点复用（含 `<think>\n\n</think>\n\n` 前缀修复），差异完全在于 KV Reduce 是否启用与跨轮锚点保存路径（见 [phone_service.cpp](../../src/a2o/native/session/phone_service.cpp)）：

1. **`cpu_baseline` (`kv_reduce=0`)**：无任何历史 KV 裁剪与先验回调；每轮 `restore` 全量历史快照 → 增量 Prefill（无 `cb_eval` 回调税）→ 保存完整历史快照 → 用全量 `n_kv = H+P+D` 做 Decode。
2. **`cpu_adaptive_full` (`kv_policy=adaptive_cost, compact_anchor=0`)**：成本表准入 + **单轮即时裁剪（全量锚点保存）**。当 `plan_kv_reduce` 判为净正收益时，当轮挂载 `PrefillPrior` 回调（交约 `613 ms` CPU 先验税），但**在裁剪前先把完整前缀导出为下一轮快照 `anchor`**，然后仅压缩本轮上下文用于当前 Decode；下一轮进入时仍恢复全量历史 KV（`h_eff == H_anchor` 持续膨胀）。
3. **`cpu_adaptive_compact` (`kv_policy=adaptive_cost, compact_anchor=1`)**：成本表准入 + **紧凑锚点跨轮持久化**。本轮先跳过全量快照导出，在 KV 裁剪与 Decode 结束后执行 `llama_memory_seq_rm(mem, 0, anchor_toks.size(), -1)` 回滚本轮生成后缀，将**压缩后的紧凑 KV 与 `anchor_live_positions` 物理回写为下一轮 `anchor`**（保持逻辑 RoPE 位置 ID 不变）。后续进入新轮次时只恢复稀疏活槽位（`h_eff ~ 1.5k–2.6k`），仅在累积再次跨越收益阈值时才稀疏触发重裁剪。

![三条件端到端对比](../../results/20261115-qwen3-1_7b-kv-curve/figures/fig3-tau2-e2e-three-conditions.png)

### 3.1 全局指标（9 题三条件对照）

| 指标 | `cpu_baseline` (`kv=0`) | `cpu_adaptive_full` (`compact=0`) | `cpu_adaptive_compact` (`compact=1`) |
|---|---|---|---|
| 完成题数 | 7 / 9 | 7 / 9 | **9 / 9 (100%)** |
| 未完成原因 | `airline:26`、`telecom:data_saver` 在读取 Agent HTTP 响应时隧道 `IncompleteRead` | `retail:110` 读取 Agent HTTP 响应隧道 `IncompleteRead`；`telecom:break_apn...` 真实 `context overflow`（Turn 16，`max_prompt=11371`） | 无 |
| 总得分 (`reward_sum` / 均值) | 3.0 / 0.333 | 3.0 / 0.333 | **4.0 / 0.444** |
| E2E 总 wall 时长 (`s`) | 2151.78 | 1551.42 | **1238.22** |
| Agent 服务累计耗时 (`s`) | 1381.70 | 1389.73 | **1109.96** (相对 baseline **`1.245×`**，且多完成 2 题) |
| Prefill / Decode / Restore / KV-Reduce (`s`) | 860.64 / 448.91 / 36.38 / 0.0 | 1027.87 / 266.47 / 34.85 / 24.97 | **778.65 / 297.92 / 12.15 / 4.10** |
| 生成总 token / Decode 速率 (`ms/tok`) | 4744 / 94.63 | 4050 / 65.79 (`1.438×`) | 4603 / **64.72** (**`1.462×`**) |
| 长轮（`>=4k`）次数 / 实际准入裁剪次数 | 92 / 0 | 87 / 87 (100%) | 87 / **16 (18.4%)** |
| 长轮均值 `restore_ms` / `prompt_ms` / `kv_reduce_ms` / `total_ms` | 395.40 / 9354.81 / 0 / 15018.53 | 400.60 / 11814.63 / 286.99 / 15973.94 | **139.64 / 8949.96 / 47.09 / 12758.11** |
| 手机内存峰值 `VmHWM` | `5936596 kB` (`5.66 GiB`) | `6680052 kB` (`6.37 GiB`) | **`4099916 kB` (`3.91 GiB`)**，下降 **`2.46 GiB`** |

### 3.2 严格公共完成子集（三条件均无故障完成的 5 题）

公共 5 题：`retail:105`、`retail:84`、`airline:0`、`airline:48`、`telecom:airplane_mode_on|break_apn_settings[PERSONA:Hard]`：

| 指标（5 题合计） | `cpu_baseline` | `cpu_adaptive_full` | `cpu_adaptive_compact` |
|---|---|---|---|
| `reward_sum` | 3.0 | 2.0（`airline:48` 得 0） | **3.0** |
| 总 wall 时长 (`s`) | 881.97 | 782.05 | **573.46** (**`1.538×`**) |
| Agent 服务耗时 (`s`) | 788.16 | 703.12 | **521.12** (**`1.512×`**) |
| Prefill (`s`) | 488.82 | 495.40 | **375.13** |
| Decode (`s`) / tok / `ms/tok` | 254.11 / 2739 / 92.78 | 153.23 / 2372 / 64.60 | **133.21 / 2060 / 64.67** (**`1.435×`**) |
| Restore (`s`) | 22.91 | 19.91 | **4.50** (**`5.09×` 缩短**) |
| KV Reduce (`s`) | 0.00 | 14.02 | **1.80** (**`7.79×` 缩短**) |

### 3.3 结论分析

1. **为什么 `adaptive_full` 在服务端总耗时上几乎无法打平 `baseline`**：
   - 未物理收缩锚点时，`H_anchor` 在每轮进入时仍持续单调膨胀（最大 `11165`），导致 **100% (87/87) 长轮** 都反复触发 `cb_eval` 先验计算与重排压缩（`prompt_s` 增加 `+167.2s`，`kv_reduce_s = 24.97s`），且超长多轮对话仍会在第 16 轮撞上 `context overflow`。
2. **`adaptive_compact` 的端到端优势**：
   - **稀疏化重裁剪与税负抑制**：一轮将历史紧凑为约 `1536–2048` 活跃 token 后，后续中间轮次 `H_effective` 从 `~2k` 起步而非 `~7k`，因此长轮实际触发 KV Reduce 比例从 `87/87` 降至 `16/87`，同步把 `restore_ms` 均值从 `395–401 ms` 压低至 `139.6 ms`，且省去大量 `cb_eval` 回调税（长轮均值总耗时 `12.76s` vs `baseline 15.02s` / `full 15.97s`）。
   - **内存峰值大幅下降 & 上下文防溢出**：峰值 `VmHWM` 从 `5.66 GiB` (`baseline`) / `6.37 GiB` (`full`) 降至 **`3.91 GiB`**（节省 `2.46 GiB`），并在 `telecom:break_apn...`（原始文本长度相当于 `10383` anchor token）上保持有效锚点 `<= 2654` 顺利跑完，消除 `context overflow`。
   - **质量无损**：全集 `reward_sum = 4.0 > 3.0`（`retail:105`、`airline:0/26/48` 全通过），公共 5 题严格与无裁剪 `baseline` 持平（`3.0/5`），未出现因 `adaptive_cost` + `compact_anchor` 导致的准确率损失。
