# 真实 HTTP Gap Oracle 首批实测

完整输出 `{"action":"wait"}`；D=6（含 EOG 评估）。H=4096，三次重复。共 300 条记录，准入路径响应/状态检查通过，诊断失败 0 条；HC 准入：{24: True, 32: True, 40: True, 48: True, 56: True}。

两个 P 的 CC/HC 均通过数值准入。

时间单位 ms；Router 列为成本感知策略返回后到完整响应的中位数；选路 Oracle 同该准备策略。联合 Oracle 列为从 Tool 发起到完整响应的中位数。

| P | G 目标 | 实际 G | Router 路径 | ready/3 | Router 返回后 | 选路 Oracle 返回后 | 联合 Oracle 总时长 | 联合策略/路径 |
|---|---|---|---|---|---|---|---|---|
| 24 | 0 | 47.0 | CC/CC/CC | 2 | 1854.7 | 状态未匹配 | 1833.1 | always/CC |
| 24 | 200 | 290.8 | CC/CC/CC | 3 | 1805.8 | 1805.8 | 2084.4 | after_return/CC |
| 32 | 0 | 43.9 | HC/CC/HC | 2 | 1689.3 | 状态未匹配 | 2162.3 | always/CC |
| 32 | 200 | 309.9 | HC/HC/HC | 3 | 1701.4 | 1701.4 | 1986.6 | always/HC |
| 40 | 0 | 43.2 | CC/HC/CC | 1 | 2693.0 | 状态未匹配 | 2702.4 | after_return/CC |
| 40 | 200 | 491.1 | HC/HC/HC | 3 | 1763.2 | 1763.2 | 2035.6 | always/HC |
| 48 | 0 | 51.5 | HC/HC/CC | 2 | 1773.9 | 1773.9 | 3143.5 | always/CC |
| 48 | 200 | 369.4 | HC/HC/HC | 3 | 1794.1 | 1794.1 | 2125.4 | always/HC |
| 56 | 0 | 39.4 | HC/CC/CC | 1 | 3281.3 | 状态未匹配 | 3352.4 | always/CC |
| 56 | 200 | 342.1 | HC/HC/HC | 3 | 1816.1 | 1816.1 | 2138.5 | always/HC |

## 强制路径诊断（不合格 HC 不进入 Oracle）

| P | G 目标 | 策略 | CC 返回后 ms | HC 返回后 ms | HC ready/3 | HC 准入 |
|---|---|---|---|---|---|---|
| 24 | 0 | after_return | 1799.8 | 3224.2 | 0 | True |
| 24 | 0 | always | 1798.1 | 3225.0 | 1 | True |
| 24 | 0 | cost_aware | 1863.8 | 3192.5 | 2 | True |
| 24 | 200 | after_return | 1775.9 | 3274.6 | 0 | True |
| 24 | 200 | always | 1801.7 | 3291.4 | 3 | True |
| 24 | 200 | cost_aware | 1810.9 | 3210.6 | 3 | True |
| 32 | 0 | after_return | 2182.7 | 1742.1 | 0 | True |
| 32 | 0 | always | 2120.8 | 1693.2 | 1 | True |
| 32 | 0 | cost_aware | 2129.7 | 1630.2 | 2 | True |
| 32 | 200 | after_return | 2284.6 | 1720.9 | 0 | True |
| 32 | 200 | always | 2204.1 | 1663.9 | 3 | True |
| 32 | 200 | cost_aware | 2232.3 | 1674.7 | 3 | True |
| 40 | 0 | after_return | 2650.0 | 1742.3 | 0 | True |
| 40 | 0 | always | 2707.5 | 1716.7 | 0 | True |
| 40 | 0 | cost_aware | 2618.3 | 1828.3 | 0 | True |
| 40 | 200 | after_return | 2685.7 | 1774.6 | 0 | True |
| 40 | 200 | always | 2604.5 | 1716.5 | 3 | True |
| 40 | 200 | cost_aware | 2543.0 | 1764.2 | 3 | True |
| 48 | 0 | after_return | 3118.1 | 1825.0 | 0 | True |
| 48 | 0 | always | 3109.0 | 1772.1 | 2 | True |
| 48 | 0 | cost_aware | 3120.5 | 1806.6 | 2 | True |
| 48 | 200 | after_return | 3026.6 | 1790.2 | 0 | True |
| 48 | 200 | always | 2888.6 | 1787.0 | 3 | True |
| 48 | 200 | cost_aware | 2974.6 | 1722.2 | 3 | True |
| 56 | 0 | after_return | 3336.3 | 1945.4 | 0 | True |
| 56 | 0 | always | 3317.2 | 1803.4 | 1 | True |
| 56 | 0 | cost_aware | 3426.1 | 1804.8 | 1 | True |
| 56 | 200 | after_return | 3531.8 | 1849.9 | 0 | True |
| 56 | 200 | always | 3543.9 | 1824.8 | 3 | True |
| 56 | 200 | cost_aware | 3401.0 | 1791.9 | 3 | True |

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
