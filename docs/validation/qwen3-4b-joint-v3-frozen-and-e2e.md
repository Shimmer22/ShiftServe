# Qwen3-4B-Q4_0 联合机制（M1 + M2）端到端与成对冻结回放验收

**结论**：在 SM8650（Xiaomi 14 Pro, 12GB RAM）上，联合开启 **M1（Tool-Gap 异步准备 + 线性代价选路）** 与 **M2（自适应 KV 裁剪与紧凑锚点）** 的 `route_m1_m2_compact`，在 τ²-bench 9 题端到端实测和 6 样本成对冻结回放中均稳定优于纯 CPU 基线与单机制方案。单开 M1 在长上下文下会受制于全量 KV handoff 与双常驻内存抖动，只有与 M2 联合才能同时实现 prefill 加速、轻量 handoff 与快速 decode。

---

## 一、端到端实测（τ²-bench 9 题）

- **配置**：[tau2-qwen3-4b-joint-v2-e2e.json](../../configs/tau2-qwen3-4b-joint-v2-e2e.json)，`context_size=14336`，高 H 路由标定 [qwen3-4b-route-costs-highh.json](../../configs/qwen3-4b-route-costs-highh.json)（`H ∈ [1536, 5120]`），结果目录 `results/20261116-tau2-qwen3-4b-joint-v2-e2e/`。
- **内存优化（`v3`）**：在 `htp -> cpu` 交接或本轮选路走 CPU 后释放 HTP 计算上下文，并在生成下一轮检查点前清理上一轮锚点。`route_m1_m2_compact` 的 `VmSwap` 从 `v2` 的 `1173 MB` 降至 `378 MB`，`airline` 任务 decode 速度恢复至 `210–237 ms/tok`。

| 条件 | 完成 / 通过题数 | 总轮次 (Warm) | 总服务耗时 | Warm 整轮均值 | Warm Prefill 均值 | Decode 速度 |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: |
| `cpu_baseline` | 9 / **4** | 114 (105) | 8840.0 s | 47.7 s | 29.4 s | 262.2 ms/t |
| `cpu_adaptive_compact` | 9 / **3** | 156 (147) | 9961.2 s | 42.2 s (-11.5%) | 25.3 s | 229.6 ms/t |
| `route_m1_only` | 9 / **2** | 97 (88) | 8909.1 s | 57.3 s | 22.3 s | 377.3 ms/t |
| `route_m1_m2_compact` | 9 / **5** | 138 (129) | **8542.3 s** | **35.7 s (-25.2%)** | **15.3 s (-48.0%)** | 295.8 ms/t |

> 注：`route_m1_m2_compact` 的 `retail`/`telecom` 6 题运行于 `v2`（`Swap=1173 MB`），`airline` 3 题运行于 `v3`（`Swap=378 MB`）；`route_m1_only` 因不压缩 KV，在高上下文下承受严重换页与交接开销。

---

## 二、同机同轨迹成对冻结回放（6 样本，`v3`）

- **脚本与数据**：[qwen3_4b_frozen_replay.py](../../scripts/validate/qwen3_4b_frozen_replay.py)，从 `cpu_baseline` 真实轨迹中按输入长度 `P` 分层抽取 6 个样本（`H ∈ [4881, 9773]`，`P ∈ [56, 3069]`），统一 `gap_ms = 2500`，在同一台手机 `v3` 服务上成对比较 4 个条件。
- **汇总表**（`results/20261117-qwen3-4b-frozen-replay/summary.json`）：

| 条件 | HTP 选路 | 平均 KV 槽位 | Restore | Prefill | KV 裁剪 | Handoff | Decode (速度) | **平均 Post-Return** | 相对基线 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu_baseline` | 0/6 | 7,951 | 742 ms | 83.74 s | 0 ms | 0 ms | 12.70 s (262.8 ms/t) | **97.19 s** | baseline |
| `cpu_adaptive_compact` | 0/6 | 2,133 | 886 ms | 84.41 s | 1,268 ms | 0 ms | 7.51 s (155.3 ms/t) | **94.10 s** | -3.2% |
| `route_m1_only` | 5/6 | 7,951 | 359 ms | 58.37 s | 0 ms | 3,726 ms | 13.68 s (283.1 ms/t) | **76.53 s** | -21.3% |
| `route_m1_m2_compact` | **6/6** | **2,133** | **0 ms** | **57.78 s** | **4.2 ms** | **845 ms** | **8.69 s (179.7 ms/t)** | **67.64 s** | **-30.4%** |

---

## 三、核心发现与边界

1. **单开 M1 的结构性短板**：当历史上下文达到 7k–10k 时，HTP prefill 结束后若将全量 KV 原封不动搬回 CPU，交接本身耗时高达 2–14 秒，且随后 CPU decode 也被大 KV 拖慢（`283.1 ms/t`），在 `2/6` 冻结样本中反而不如纯 CPU 基线。
2. **M1 + M2 的互补闭环**：在 HTP prefill 后就地按注意力先验裁剪 KV 仅需 **3–5 ms**（比 CPU 侧裁剪整理快两个数量级），将平均 handoff 压低 **77.3%**（`3726 ms -> 845 ms`），同时把后续 CPU decode 速度压回 `179.7 ms/t`，使 `route_m1_m2_compact` 在 6/6 冻结样本中全部第一。
3. **已知边界与后续方向**：
   - `route_m1_only` 在 `~9.85k` tokens 处曾触发一次 `seq export failed`，长上下文导出边界待单独排查；
   - 在 `Qwen3-4B` 上，通过 `v3` 上下文释放已将双常驻权重压力压至可稳定运行的区间；若后续将联合方案扩展至 `Qwen3-8B`（12GB 设备上双份权重约 9.6 GB），则需要立项单份共享权重常驻（M3）。

- 绘图脚本：[qwen3_4b_frozen_and_e2e.py](../../scripts/plot/qwen3_4b_frozen_and_e2e.py)
- 汇总图：`results/20261117-qwen3-4b-frozen-replay/figures/fig-qwen3-4b-joint-summary.png`
