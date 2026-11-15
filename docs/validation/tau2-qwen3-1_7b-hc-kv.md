# Qwen3-1.7B τ²-bench 双驻留 `HC + PrefillPrior KV Reduce` 联合验证

结论：在 `Qwen3-1.7B-Q4_0` 与 `τ²-bench` 6 个真实长历史边界（`H=5046–7960, P=210–750, D=74–688`）的双驻留联合实测中，机制 2（`PrefillPrior` KV 裁剪与紧凑重排，`keep=1024`）使 `HC` 回迁状态体积平均缩小 **`7.15×`**（`840.35 MB → 117.45 MB`）、回迁耗时缩短 **`3.08×`**（`4.374 s → 1.419 s`）、CPU Decode 提速 **`1.774×`**（`115.637 s → 65.183 s`），端到端累计 `post_return_ms` 相比机制 1 `HC Baseline` 加速 **`1.321×`**（`198.530 s → 150.322 s`，6/6 边界全部更快），相比全量 `CPU Baseline` 累计仍快 **`1.170×`**（`175.911 s → 150.322 s`），且在长生成边界 `b3`（`H=7960, D=688`）单轮节省 **`32.0 s`**（`89.523 s → 57.503 s`）。同时如实记录了 `1.7B` 小模型与固定 `keep=1024` 压缩率下的两项边界特征（见第 4 节）。

## 1. 前置修复与模型选择背景

- **多轮 `H_anchor` 锚点修复**（[phone_service.cpp](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/src/a2o/native/session/phone_service.cpp)）：修复 Qwen3 `enable_thinking=false` 模板在回合末尾追加 `<think>\n\n</think>\n\n`、下一轮作为历史渲染时剥离该空思考块导致跨轮前缀错位 4 token（`H_anchor=0`）的问题。双轮对话验证（`CEDAR5837`）实测 Turn 1 命中 `H_anchor=22` 并准确召回。
- **双驻留内存容量边界**：`Qwen3-8B-Q4_0`（约 `4.5 GB` 权重）双驻留加载 `CPU + HTP`（`ctx=6144`）在 `H=5120` 时触发进程 `VmRSS ≈ 7.02 GB` 并引发 Android LMK 杀进程；改用 `Qwen3-1.7B-Q4_0`（`1.2 GB` 权重，SHA-256 `5c71ea792028698598657d8f3e13fcf1980fcb2e1887c4931c5a129a5c8b488f`）后，在 `ctx=12288` 下双驻留 `HC + PrefillPrior` 全程 `max VmRSS = 7.18 GiB`、`VmSwap = 0 kB`、新增 `major_faults = 1`。
- **`cb_eval` 零开销隔离与 tail 先验采集**（[prefill_prior.h](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/src/a2o/native/profiling/prefill_prior.h)）：`route` 模式下 `cpu` Engine 保持 `cb_eval = nullptr`；仅开启 `A2O_KV_REDUCE=1` 且策略为 `prefill_prior` 时在 `htp` Engine 挂载回调，且仅在末 `kv_tail_tokens=64` 收集 `kq_soft_max` 先验。对比未挂回调的 `HC + Recent` 消融组（Prefill 合计 `78.338 s` vs `83.697 s`），`HTP` 上挂载 `cb_eval` 仅引入 `+6.8%` prefill 额外开销。

## 2. 实验配置与范围

- 配置文件：[tau2-qwen3-1_7b-hc-kv.json](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/configs/tau2-qwen3-1_7b-hc-kv.json)
- 采集与分析入口：[tau2_qwen3_hc_kv.py](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/scripts/collect/tau2_qwen3_hc_kv.py)、[tau2_hc_kv.py](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/scripts/analyze/tau2_hc_kv.py)
- 工件目录：`results/20261115-tau2-qwen3-1_7b-hc-kv/campaign/`（共 `54` 组有效测量：`4` 组对比 × `6` 边界 × `2` 轮 `ready (2000 ms)`，外加第 0 轮 `HC + PrefillPrior` 在 `pilot-six` 真实工具等待 `2.09–3.17 ms` 下的 `6` 组 `real_gap` 实测）。
- 运行时参数：`context_size=12288`，`cpu_threads=6`，`cpu_repack=1`，`htp_host_threads=4`，`opbatch=1024`，`opqueue=16`，`kv_offload=0`，`flash_attention=0`（前期探针确认 `1.7B` 无 FA 下开启 `kv_offload=1` 会使 `b0/b1` HTP prefill 从 `6.73s/22.68s` 恶化至 `9.24s/27.43s`）；KV 裁剪参数：`kv_keep_tokens=1024`，`kv_sink_tokens=4`，`kv_recent_tokens=256`（`recent` 消融组为 `1020`），`kv_pool_window=5`，`kv_tail_tokens=64`。
- 作用域说明：本轮测试使用 `Qwen3-1.7B` 导出的冻结边界序列（[tau2-1_7b-requests.json](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/results/20261115-tau2-qwen3-1_7b-hc-kv/tau2-1_7b-requests.json)）执行 `teacher-forced` 延时测量，并在第 0 轮 decode 后通过 `seq_rm` 回滚 `[H+P .. -1)` 执行 `128 token` 贪心自由生成以检验输出质量；强制启用 `HC` 路径以隔离验证机制 2 在 `HC` 上的增益，不代表完整多轮 E2E 任务成功率。

## 3. 核心实测结果（6 边界累计均值，2 轮）

| 条件 | Prefill (s) | Evict (ms/边界) | Handoff (s) | Handoff MB | `decode_n_kv` | Decode (s) | `post_return` (s) | Max Swap (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CPU Baseline` | `55.407` | `0.0` | `0.000` | `0` | `7326.5` | `117.934` | `175.911` | `203508` |
| `HC Baseline` | `78.515` | `0.0` | `4.374` | `840.35` | `7326.5` | `115.637` | `198.530` | `203508` |
| `HC + PrefillPrior` | `83.697` | `2.0–2.8` | `1.419` | `117.45` | `1024` | `65.183` | `150.322` | `0` |
| `HC + Recent`（消融） | `78.338` | `1.1–2.1` | `0.815` | `117.45` | `1024` | `66.098` | `145.267` | `0` |

分边界 `post_return_ms`（`ready` 组两轮均值，单位秒）与速度分解：

| 边界 | `H / P / D` | `CPU Base` (s) | `HC Base` (s) | `HC + Prior` (s) | `HC+Prior` Decode `ms/tok` (`CPU Base`) | 对比 `HC Base` |
|---|---|---:|---:|---:|---:|---:|
| `b0 retail` | `5046 / 264 / 115` | `13.542` | `16.102` | `15.046` | `58.54` (`79.63`) | `1.070×` |
| `b1 retail` | `6478 / 750 / 74` | `26.538` | `30.517` | `29.238` | `58.63` (`93.55`) | `1.044×` |
| `b2 airline` | `7102 / 343 / 102` | `18.117` | `22.394` | `19.344` | `60.66` (`94.19`) | `1.158×` |
| `b3 airline` | `7960 / 470 / 688` | `89.523` | `93.462` | `57.503` | `57.06` (`112.54`) | **`1.625×`**（比 CPU 也快 **`1.557×`**） |
| `b4 telecom` | `7696 / 210 / 78` | `13.131` | `16.327` | `13.136` | `54.98` (`96.43`) | `1.243×` |
| `b5 telecom` | `7335 / 305 / 76` | `15.061` | `19.729` | `16.056` | `57.64` (`96.03`) | `1.229×` |

- **`real_gap` 补充**：在 `pilot-six` 真实本地工具间隙（`2.09–3.17 ms`）下，`HC + PrefillPrior` 的 `prepare_ms` 为 `303.9–488.4 ms`，`wait_ms` 为 `301.9–486.0 ms`（6 边界合计约 `2.40 s`，占总 `post_return` 约 `1.6%`），未改变上述加速方向。

## 4. 质量对比与发现边界

- **自由生成首 token 一致率**：`6/6` 边界上 `HC Baseline`、`HC + PrefillPrior`、`HC + Recent` 的 `top1_first` 与 `CPU Baseline` **`100%` 一致**。
- **`PrefillPrior` 对长程工具定义的保护（`b3`）**：在 `b3 (H=7960)`，`CPU Baseline`（文本询问未调用）以外，`HC Baseline` 与 `HC + PrefillPrior` 均正确生成 `<tool_call> {"name": "get_reservation_details", "arguments": {"reservation_id": "Q0ZF0J"}}`；而 **`HC + Recent`（仅保留 4 个 sink 与末尾 1020 个 token）完全丢失早先系统区工具定义，停止输出且未能生成 `<tool_call>`**。
- **固定 `keep=1024`（`~7×` 压缩）下的细节退化边界（`b1`）**：`b0` 与 `b4` 上 `HC + PrefillPrior` 完整保留了邮箱、地址、信用卡后四位、订单号 `#W1603792`/`#W1092119`、电话 `555-123-2002` 与套餐 `P1001`；但在 `b1`（`7228 → 1024`，前文并列存在多个相似订单结果），`CPU Baseline` 与 `HC Baseline` 准确继续查询 `#W6231698`，而 `HC + PrefillPrior` 与 `HC + Recent` 均幻觉编造了不存在的订单号（`#W3456789` / `#W7896397433`）。表明在未针对关键早期实体提高保留预算时，单一全局 `keep=1024` 在 `7×` 压缩下可能丢失相邻未聚焦实体的精确编号。
- **`1.7B` 与大模型的架构启示**：`1.7B` 在开启 `cpu_repack=1 (6 线程)` 后 `CPU prefill`（`55.41 s`）本身快于 `HTP prefill`（`78.51–83.70 s`），所以短输出轮次（`D < 120`）上 `CPU + KV Reduce` 会比 `HC + KV Reduce` 更具优势；而在 `8B/9B` 或未来可容纳双驻留的大模型上，`HTP prefill` 显著快于 `CPU prefill`，`HC + PrefillPrior KV Reduce` 将同时收割 Prefill 卸载、`7×` 状态回迁瘦身与 `~1.8×` CPU Decode 加速三项红利。

手机端已停机并核验：`LOCK_FREE`、`NO_ACTIVE_MODEL`。
