# 真实 HTTP Gap Oracle 首批实测

完整输出 `{"action":"wait"}`；D=6（含 EOG 评估）。H=4096，三次重复。共 228 条响应/状态检查通过；HC 数值准入：{16: False, 64: False}。

**核心结果：HC 未通过既定数值门槛。本批只能得到单路径 CC 的退化 Oracle，无法据此判断 Router 的选路差距或联合策略的有效收益。**

时间单位 ms；Router 列为成本感知策略返回后到完整响应的中位数；选路 Oracle 同该准备策略。联合 Oracle 列为从 Tool 发起到完整响应的中位数。

| P | G 目标 | 实际 G | Router 路径 | ready/3 | Router 返回后 | 选路 Oracle 返回后 | 联合 Oracle 总时长 | 联合策略/路径 |
|---|---|---|---|---|---|---|---|---|
| 16 | 0 | 51.2 | CC/CC/CC | 0 | 1439.3 | 1347.1 | 1481.4 | after_return/CC |
| 16 | 50 | 98.3 | CC/CC/CC | 0 | 1426.0 | 1372.9 | 1503.8 | always/CC |
| 16 | 200 | 326.0 | CC/CC/CC | 0 | 1400.7 | 1456.0 | 1718.4 | after_return/CC |
| 16 | 500 | 605.0 | CC/CC/CC | 0 | 1463.3 | 1434.3 | 2061.1 | after_return/CC |
| 64 | 0 | 45.6 | CC/CC/CC | 0 | 3824.5 | 3669.6 | 3731.0 | always/CC |
| 64 | 50 | 99.2 | CC/CC/CC | 0 | 3680.8 | 3973.2 | 3844.4 | after_return/CC |
| 64 | 200 | 282.0 | CC/CC/CC | 0 | 3956.5 | 3657.2 | 4150.7 | after_return/CC |
| 64 | 500 | 642.8 | CC/CC/CC | 0 | 3749.2 | 3807.6 | 4370.6 | always/CC |

**HC 数值未准入的 P 只有 CC 候选，因此选路差距与联合收益无法辨识（JSON 中为 null）；表中的单路径时间不构成最优性或收益证据。**

## 强制路径诊断（不合格 HC 不进入 Oracle）

| P | G 目标 | 策略 | CC 返回后 ms | HC 返回后 ms | HC ready/3 | HC 准入 |
|---|---|---|---|---|---|---|
| 16 | 0 | after_return | 1442.5 | 2578.4 | 0 | False |
| 16 | 0 | always | 1395.4 | 2596.9 | 1 | False |
| 16 | 0 | cost_aware | 1347.1 | 2605.5 | 0 | False |
| 16 | 50 | after_return | 1427.3 | 2759.0 | 0 | False |
| 16 | 50 | always | 1362.4 | 2556.3 | 3 | False |
| 16 | 50 | cost_aware | 1372.9 | 2638.0 | 0 | False |
| 16 | 200 | after_return | 1446.0 | 2603.9 | 0 | False |
| 16 | 200 | always | 1456.9 | 2638.9 | 3 | False |
| 16 | 200 | cost_aware | 1456.0 | 2523.5 | 0 | False |
| 16 | 500 | after_return | 1440.3 | 2571.9 | 0 | False |
| 16 | 500 | always | 1452.8 | 2600.0 | 3 | False |
| 16 | 500 | cost_aware | 1434.3 | 2660.6 | 0 | False |
| 64 | 0 | after_return | 3791.4 | 1971.8 | 0 | False |
| 64 | 0 | always | 3688.0 | 1996.5 | 3 | False |
| 64 | 0 | cost_aware | 3669.6 | 2011.4 | 0 | False |
| 64 | 50 | after_return | 3738.6 | 1971.5 | 0 | False |
| 64 | 50 | always | 3854.5 | 1999.3 | 3 | False |
| 64 | 50 | cost_aware | 3973.2 | 1994.8 | 0 | False |
| 64 | 200 | after_return | 3864.0 | 1906.2 | 0 | False |
| 64 | 200 | always | 3878.6 | 1979.5 | 3 | False |
| 64 | 200 | cost_aware | 3657.2 | 1960.0 | 0 | False |
| 64 | 500 | after_return | 3986.3 | 1934.9 | 0 | False |
| 64 | 500 | always | 3707.0 | 1901.7 | 3 | False |
| 64 | 500 | cost_aware | 3807.6 | 1877.0 | 0 | False |

## 解释边界

- 三次串行反事实重放，不能获得同一轮未选路径的真实耗时；最小值存在选择偏差，差值可为负
- HC 未准入时只有 CC，选路差距与联合收益记为 null；不能把退化的单路径 Oracle 解释为 Router 已达最优
- HC 仅在同一 P 的三次完整校准全部过数值门槛后准入；失败组原样保留，强制 HC 测量仍可作诊断
- 严格选路 Oracle 仅把返回时 ready 的 HC 当候选；联合 Oracle 要求 HC 三次均 ready；另列允许等待的扩展值
- 联合 Oracle 选用三次中位数，不将不同阶段的最小值拼接；没有离线 overlap 模拟
- G 是服务器延迟目标；实际 Gap 含 adb reverse 网络、线程调度及 HTTP 开销
- 固定文本微基准；无图像、无真实 GUI 操作、无能耗结论；每轮回收等待在响应计时之外并单独记录
- 返回后准备策略的 Router 遵循现有不等待 pending 的规则，因此选 CC；HC 强制重放单独测按需准备成本
- 共同 snapshot 构建/导出和 CPU 恢复在 Tool 前；提前准备包括 clear/import/字节核验，未压缩 KV
- 边界 logits 校验仅在独立校准执行；测量阶段检查真实 HTTP P、完整生成 ID、EOG 与状态迁移字节

原始数据：`checked-rows.jsonl`、`device/trials.jsonl`、`http-server.jsonl`、`protocol.log`、`runtime.log`。`summary.csv` 列出三个准备策略，`summary.json` 保留每次严格候选及扩展允许等待 Oracle。

## 并发与复核

真实提前准备 72 次，72 次与 HTTP 时间区间实际重叠，返回时 ready 64 次；准备耗时中位数 44.4ms。这些是设备实际时间戳的观察，不是离线模拟。

HTTP 服务器延迟目标 0/50/200/500ms 对应的手机实际 Gap 中位数分别为 44.8/107.1/304.0/626.0ms；包括无线 adb reverse 的往返与调度开销，不能当成精确 0/50/200/500ms 的端到端 Gap。

| P | HC 校准次数 | 最小边界 cosine | 最大边界 NRMSE | 数值准入 |
|---|---|---|---|---|
| 16 | 3 | 0.998832 | 0.048636 | 不通过 |
| 64 | 3 | 0.998556 | 0.053982 | 不通过 |

阈值保持 cosine≥0.999、NRMSE≤0.05；本轮仅校验 prefill/final 边界，不冒称逐 token 数值通过。28 项本地测试、额外 5 项最终分析测试通过；HTTP 请求/正文 hash、228 份原始测量、响应发布顺序、源码/二进制 hash 和本轮设备进程/reverse 清理检查见 `final-checks.json`。旧三次启动/校准失败目录均保留。
