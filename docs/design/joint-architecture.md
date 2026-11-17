# 双机制协同架构：Gap 选路（M1）与 NPU 先验指导解码（M2）

本文档澄清本仓库两条核心机制的设计初衷、互补相图，以及当前实测与理想目标之间的工程边界。

## 1. 为什么需要两条机制：`(H, P, G)` 相图接力

在端侧 CPU/HTP 异构 Agent 推理中，单一静态后端或单一机制无法覆盖完整工作负载：

1. **全 CPU**：无跨后端迁移开销，但长输入（大 `H` / 大 `P`）Prefill 慢，且长历史 `H` 下 CPU Decode 带宽受限、内存膨胀。
2. **全 HTP（Full HTP）**：Prefill 快，但 HTP Decode 慢且算子/长度受限（如 `τ²-bench` 9B 冻结回放中，Full HTP 累计耗时比 CPU 还慢，比 `HC` 慢 `1.608×`）。
3. **固定 `HC`（HTP Prefill → CPU Decode）**：大 `H`/大 `P` 时占优，但在小 `P` 或短 `G` 时会被 HTP 固定启动与 `HTP→CPU` 回迁成本反杀。

因此，系统按 `(H, P, G)` 空间分为两层接力：

| 机制 | 定位与适用区间 | 解决的核心问题 |
|---|---|---|
| **M1：Gap 异步准备 + 返回后选路** | **浅水区 / 边界不确定区**<br>（如 ToolSandbox / BFCL 中 `P` 跨交叉点、`G` 波动） | 工具发出时 `P, G` 未知：Gap 内按预算异步准备 HTP 状态；返回后依据实测 `P` 与剩余准备时间在 `CC`（CPU）与 `HC`（HTP prefill → CPU decode）间动态选路，避免小 `P`/短 `G` 被 HTP 开销反杀。 |
| **M2：NPU Prefill 先验指导 KV 裁剪（`PrefillPrior` + `compact_anchor`）** | **深水区 / `HC` 收敛区**<br>（如 `τ²-bench` 等长历史 `H`、大 `P` 多轮任务，M1 稳定收敛到 `HC`） | 当 M1 收敛为 `HC` 时，纯选路退化为静态 `Always-HC`。M2 的设计初衷是在 **`HC` 管道内部建立跨后端协同**：利用 NPU Prefill 顺带产出的注意力先验（`kq_soft_max`）在 HTP 侧裁剪冗余 KV 并紧凑持久化（`compact_anchor`），同时压低 **HTP→CPU 回迁量**、**CPU Decode 每 token 耗时** 与 **双驻留内存峰值**。 |

---

## 2. M2 的异构协同逻辑（为什么需要 NPU 指导 CPU Decode）

单纯在 CPU 上做 KV 裁剪虽有工程收益（Qwen3-1.7B 单驻留 CPU 实测公共 5 题服务耗时 `1.512×` 加速、`VmHWM` 降 `2.46 GiB`），但从异构协同视角看，**在 CPU 上计算先验代价高昂**：

- 以 Qwen3-4B 成本表（[qwen3-4b-kv-reduce-cost.json](../../configs/qwen3-4b-kv-reduce-cost.json)）为例：CPU 提取先验的 `cb_eval` 固定税约 `1030.6 ms`，而 HTP 侧约 `291.1 ms`（低约 `3.5×`），且 HTP Prefill 本身在大 `P` 下快于 CPU。
- **理想闭环**：NPU 在执行 Prefill 的同时以低边际代价产出 `PrefillPrior` → 在 HTP→CPU 回迁前完成物理紧凑化 → CPU 仅接收并恢复紧凑后的活跃 KV 槽位执行更快 Decode，并将紧凑锚点跨轮持久化（保持逻辑 RoPE 位置不变）。

---

## 3. 当前实测与理想闭环的差距（诚实边界）

以下边界基于已有实测，不得在结论中越界宣称：

1. **M1 在 `τ²-bench` 单独退化为 `Always-HC`，尚未在混合分布上证明“动态分岔优于静态 `Always-HC`”**
   - [tau2-greedy-return.md](../validation/tau2-greedy-return.md) 中 18/18 次 Route 均选 HC；证明了 `Route > CPU (1.258×)` 与 `Route > Full HTP (1.608×)`，但尚缺含小 `P`/短 `G` 的混合回放来证明 `Router` 严格优于静态 `Always-HC`。
   - 真实 Gap 需区分：BFCL 内存沙箱 `G` 中位数仅 `0.193 ms`（[tool-return-structure.md](../validation/tool-return-structure.md)），`τ²-bench` 本地工具调用 Gap 趋近于 0（当前隐藏准备时间来自远端 27B user-sim 网络延迟）。
2. **M2 在 Qwen3-4B 上的联合闭环尚未成立（待修复项）**
   - **`k_floor` 基数与前缀保护**：[kv_reduce_cost.h](../../src/a2o/native/scheduling/kv_reduce_cost.h) 当前按已压缩的 `h_eff + P` 计算下限，多轮中第 0 轮即裁剪系统/工具前缀，后期保留率跌至约 `17%`；需改为按 `H_anchor + P` 计算并对系统/工具前缀加 pin。
   - **裁剪后 Handoff 慢路径**：4B 实测裁剪后 `state_bytes` 虽降为 `226.5 MB`（1536 token），但 `handoff_ms` 反从 `232–271 ms` 升至 `329–470 ms`，导致“NPU 先验缩小回迁耗时”的协同假设在当前导出路径上尚未兑现。
   - **`htp_prior_tax` 仍偏高**：当前通过 `cb_eval` 获取 `kq_soft_max` 在 1.7B（`655.7 + 1.145·P` ms）与 4B（`~291 ms`）上仍有显式回调税，需通过成本门控避开短历史死区，尚未实现“近零额外税”的算子内聚合导出。
3. **E2E 联合对照尚不完整**：4B 上 `cpu_baseline`、`route_m1_only`、`m2_only`、`joint` 四个条件尚未在统一任务集、统一驻留模式与同批 user-sim 下完整跑齐并做配对比较。

---

## 4. 关键文档索引

- M1 设计与边界：[Tool Gap 中面向不确定需求的状态准备](gap-aware-state-preparation.md)、[P 分布与投机准备期望价值](p-distribution-value.md)、[τ² 贪心继续准备](../validation/tau2-greedy-return.md)
- M2 历史探索（已暂停）：[NPU 草稿、CPU 验证探索](second-mechanism.md)
- M2 当前主线（KV Reduce + `compact_anchor`）：[Qwen3-1.7B 压缩曲线与端到端实测](../validation/qwen3-1_7b-kv-cost-and-e2e.md)、[Qwen3-1.7B τ² 双驻留 HC + KV Reduce](../validation/tau2-qwen3-1_7b-hc-kv.md)
