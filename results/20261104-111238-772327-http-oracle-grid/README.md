# 真实 HTTP Gap Oracle 首批实测

完整输出 `{"action":"wait"}`；D=6（含 EOG 评估）。H=2048，三次重复。共 180 条记录，准入路径响应/状态检查通过，诊断失败 0 条；HC 准入：{31: True, 32: True, 33: True}。

两个 P 的 CC/HC 均通过数值准入。

时间单位 ms；Router 列为成本感知策略返回后到完整响应的中位数；选路 Oracle 同该准备策略。联合 Oracle 列为从 Tool 发起到完整响应的中位数。

| P | G 目标 | 实际 G | Router 路径 | ready/3 | Router 返回后 | 选路 Oracle 返回后 | 联合 Oracle 总时长 | 联合策略/路径 |
|---|---|---|---|---|---|---|---|---|
| 31 | 0 | 41.4 | HC/HC/HC | 3 | 2269.8 | 2269.8 | 2348.1 | always/HC |
| 31 | 200 | 306.7 | HC/HC/HC | 3 | 2311.8 | 2242.0 | 2622.0 | always/HC |
| 32 | 0 | 64.7 | HC/HC/HC | 3 | 1501.1 | 1500.2 | 1585.7 | always/HC |
| 32 | 200 | 300.0 | HC/HC/HC | 3 | 1528.3 | 1528.3 | 1813.9 | always/HC |
| 33 | 0 | 37.6 | HC/HC/HC | 3 | 1521.5 | 1521.5 | 1573.4 | always/HC |
| 33 | 200 | 336.0 | HC/HC/HC | 3 | 1518.6 | 1518.6 | 1888.0 | always/HC |

## 强制路径诊断（不合格 HC 不进入 Oracle）

| P | G 目标 | 策略 | CC 返回后 ms | HC 返回后 ms | HC ready/3 | HC 准入 |
|---|---|---|---|---|---|---|
| 31 | 0 | after_return | 2553.9 | 2329.7 | 0 | True |
| 31 | 0 | always | 2546.0 | 2310.1 | 3 | True |
| 31 | 0 | cost_aware | 2359.0 | 2309.9 | 3 | True |
| 31 | 200 | after_return | 2505.3 | 2342.4 | 0 | True |
| 31 | 200 | always | 2491.0 | 2308.6 | 3 | True |
| 31 | 200 | cost_aware | 2561.1 | 2334.1 | 3 | True |
| 32 | 0 | after_return | 1829.6 | 1527.2 | 0 | True |
| 32 | 0 | always | 1957.7 | 1506.1 | 3 | True |
| 32 | 0 | cost_aware | 1830.3 | 1497.4 | 3 | True |
| 32 | 200 | after_return | 1951.3 | 1533.1 | 0 | True |
| 32 | 200 | always | 1845.9 | 1502.0 | 3 | True |
| 32 | 200 | cost_aware | 1946.9 | 1512.4 | 3 | True |
| 33 | 0 | after_return | 2601.3 | 1549.6 | 0 | True |
| 33 | 0 | always | 2691.1 | 1519.6 | 3 | True |
| 33 | 0 | cost_aware | 2679.3 | 1519.1 | 3 | True |
| 33 | 200 | after_return | 2690.1 | 1544.9 | 0 | True |
| 33 | 200 | always | 2447.7 | 1532.4 | 3 | True |
| 33 | 200 | cost_aware | 2521.5 | 1539.3 | 3 | True |

## 解释边界

- 三次串行反事实重放，不能获得同一轮未选路径的真实耗时；最小值存在选择偏差，差值可为负
- HC 未准入时只有 CC，选路差距与联合收益记为 null；不能把退化的单路径 Oracle 解释为 Router 已达最优
- HC 仅在同一 P 的三次完整校准全部过 manifest 中的数值门槛后准入；失败组原样保留，强制 HC 测量仍可作诊断
- 严格选路 Oracle 总包含实际已选路径；反事实重放的准备/ready 状态不匹配则该条件差距记 null，不将其伪装成可比较 Oracle；联合 Oracle 要求 HC 三次均 ready；另列允许等待的扩展值
- 联合 Oracle 选用三次中位数，不将不同阶段的最小值拼接；没有离线 overlap 模拟
- G 是服务器延迟目标；实际 Gap 含 adb reverse 网络、线程调度及 HTTP 开销
- 固定文本微基准；无图像、无真实 GUI 操作、无能耗结论；每轮回收等待在响应计时之外并单独记录
- 返回后准备策略的 Router 遵循现有不等待 pending 的规则，因此选 CC；HC 强制重放单独测按需准备成本
- 共同 snapshot 构建/导出和 CPU 恢复在 Tool 前；提前准备包括 clear/import/字节核验，未压缩 KV
- 边界 logits 校验仅在独立校准执行；测量阶段检查真实 HTTP P、完整生成 ID、EOG 与状态迁移字节

原始数据：`checked-rows.jsonl`、`device/trials.jsonl`、`http-server.jsonl`、`protocol.log`、`runtime.log`。`summary.csv` 列出三个准备策略，`summary.json` 保留每次严格候选及扩展允许等待 Oracle。
