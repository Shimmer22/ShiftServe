# 真实 HTTP Gap Oracle 首批实测

完整输出 `{"action":"wait"}`；D=6（含 EOG 评估）。H=4096，三次重复。共 180 条记录，准入路径响应/状态检查通过，诊断失败 0 条；HC 准入：{31: True, 32: True, 33: True}。

两个 P 的 CC/HC 均通过数值准入。

时间单位 ms；Router 列为成本感知策略返回后到完整响应的中位数；选路 Oracle 同该准备策略。联合 Oracle 列为从 Tool 发起到完整响应的中位数。

| P | G 目标 | 实际 G | Router 路径 | ready/3 | Router 返回后 | 选路 Oracle 返回后 | 联合 Oracle 总时长 | 联合策略/路径 |
|---|---|---|---|---|---|---|---|---|
| 31 | 0 | 37.0 | CC/CC/CC | 0 | 2716.1 | 2716.1 | 2778.8 | after_return/CC |
| 31 | 200 | 302.0 | CC/CC/CC | 3 | 2598.5 | 2598.5 | 3122.0 | always/CC |
| 32 | 0 | 48.3 | HC/CC/HC | 2 | 1731.4 | 状态未匹配 | 2203.2 | after_return/CC |
| 32 | 200 | 288.2 | HC/HC/HC | 3 | 1654.2 | 1654.2 | 2149.2 | always/HC |
| 33 | 0 | 75.5 | HC/HC/CC | 2 | 1793.8 | 状态未匹配 | 2776.8 | always/CC |
| 33 | 200 | 309.5 | HC/HC/HC | 3 | 1802.6 | 1802.6 | 2030.4 | always/HC |

## 强制路径诊断（不合格 HC 不进入 Oracle）

| P | G 目标 | 策略 | CC 返回后 ms | HC 返回后 ms | HC ready/3 | HC 准入 |
|---|---|---|---|---|---|---|
| 31 | 0 | after_return | 2736.5 | 3966.0 | 0 | True |
| 31 | 0 | always | 2862.4 | 3810.0 | 3 | True |
| 31 | 0 | cost_aware | 2685.9 | 3818.5 | 2 | True |
| 31 | 200 | after_return | 2822.2 | 3836.3 | 0 | True |
| 31 | 200 | always | 2864.2 | 3819.5 | 3 | True |
| 31 | 200 | cost_aware | 2878.5 | 3755.6 | 3 | True |
| 32 | 0 | after_return | 2161.3 | 1711.1 | 0 | True |
| 32 | 0 | always | 2228.0 | 1626.0 | 1 | True |
| 32 | 0 | cost_aware | 2172.1 | 1717.3 | 1 | True |
| 32 | 200 | after_return | 2289.4 | 1728.7 | 0 | True |
| 32 | 200 | always | 2272.8 | 1718.2 | 3 | True |
| 32 | 200 | cost_aware | 2350.7 | 1768.7 | 3 | True |
| 33 | 0 | after_return | 2879.5 | 1765.1 | 0 | True |
| 33 | 0 | always | 2739.0 | 1676.2 | 1 | True |
| 33 | 0 | cost_aware | 2850.2 | 1852.3 | 1 | True |
| 33 | 200 | after_return | 3024.8 | 1746.8 | 0 | True |
| 33 | 200 | always | 2868.5 | 1711.7 | 3 | True |
| 33 | 200 | cost_aware | 2754.7 | 1682.9 | 3 | True |

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
