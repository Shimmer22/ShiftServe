# T4 主机自由生成质量评测

用 T3 终版主机构建 `20261120-t3-final-host` 在 wins 的 RTX 4090 上运行 Qwen3-8B Q4_0 和 Qwen3-4B Q4_0。各模型运行同一套 BFCL 60 题、ToolSandbox 28 个场景和 τ²-bench 36 题，按 8B、4B 顺序采集。每个模型和基准都有 Full-KV、中预算、高预算三个条件，共 18 个对照组、744 次任务运行。

| 条件 | KV 策略 | 可排序段保留比例 | 跨轮裁剪锚点 |
| --- | --- | ---: | --- |
| Full-KV | 全量保留 | — | 关闭 |
| ShiftServe 中预算 | `prefill_prior` | 0.50 | 开启 |
| ShiftServe 高预算 | `prefill_prior` | 0.75 | 开启 |

两组 ShiftServe 均设置 `kv_layer_stride=4` 和 `kv_min_evict_tokens=256`。保留比例作用于扣除受保护前缀和最近 token 后的区间。可淘汰量低于阈值时，封版执行器会选用 `recent`，表中单列这类轮次。质量评测使用主机 CUDA 单驻留上下文，端侧路由、交接和计时来自 T6 的冻结回放。

三组统一开启 Agent thinking，使用温度 0 和同一题集、种子与生成预算。BFCL 每请求上限 2,048 token，ToolSandbox 2,048 token，τ²-bench 4,096 token。τ²-bench 统一采用 T3 的 `judge_api` 用户模拟器 `gpt-6.1-terra`，模拟器输出随交互变化。

## 运行与产物

```sh
python3 -u scripts/collect/t4_host_accuracy.py
```

默认输出 [t4-stride4](../../results/mainline-eval/t4-stride4/)。各阶段记录配置快照、二进制与模型校验、逐题结果及服务日志。同目录重跑时继续原计划。单题冒烟用 `--smoke`，默认写入 [t4-stride4-smoke](../../results/mainline-eval/t4-stride4-smoke/)。

`table3.csv` 汇总 BFCL 准确率、ToolSandbox 平均相似度及 τ²-bench 平均 reward，三项都用完整题集作分母。缺失的 reward 或相似度按 0 纳入均值，并保留缺失计数、评测失败、上下文溢出、生成截断和请求错误计数。分类可能重叠。`table3.json` 列出待采条件和缺少先验命中或锚点复用证据的条件。表中也记录实际淘汰 token 数和裁剪轮次的淘汰比例，便于核验各预算的实际作用。

## 8B 冒烟记录

BFCL 的 `multi_turn_long_context_94` 在三个条件下均跑完整条交互。Full-KV 通过官方校验，中、高预算均未通过。中预算在 3 轮使用先验、5 轮复用裁剪锚点，高预算分别为 5 轮、7 轮。ToolSandbox 的 `get_wifi` 三组均完结、相似度均为 0.8606，这条短样本未触发裁剪。单题数据用于核验运行路径。[冒烟表](../../results/mainline-eval/t4-stride4-smoke/table3.csv)
