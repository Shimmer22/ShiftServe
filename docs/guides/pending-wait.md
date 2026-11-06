# 短 Gap 的按成本有限等待

这是原报告 Oracle 路径上的可选实验机制；原 `CostRouter::choose` 的
ready-only 基线和常驻 Tool 会话保持原有行为。本机制尚未接入常驻 Tool 会话。

## 决策

仅当提前准备的 HTP 在 Tool 完整返回时仍为 pending，且独立校准已通过时考虑等待。
校准 CC/HC 后续成本均为 prefill、交接核验和 CPU decode 的中位数；
准备估计使用同 P 的三次 HC 校准 `prepare_ms` 最大值乘 1.5。
该估计只是保守试算，不能保证覆盖所有未来准备时间。

本轮固定配置为 cap=100ms、margin=20ms。只有
`HC后续成本 + 准备估计 + margin < CC后续成本` 才准入，
预算为 `min(cap, CC后续成本 - HC后续成本 - margin)`。
不根据本轮完成时间或不可观测进度估计实时剩余准备时间。

预算内就绪则完成 future.get 并执行 HTP prefill；超时则执行 CPU。
未完成的准备线程在响应发布后回收，原有状态隔离与核验保留。
`wait_for` 是请求的等待期限，不是操作系统硬实时保证；实测等待及预算超出量均记录，
不剔除超时或慢点。完整响应继续用包含挂起的 CLOCK_BOOTTIME。

## 对照与触发率

H4096、P24/32；三次独立 CC/HC 校准后，每条件20组相邻配对，
ready-only/bounded 的 AB/BA 各半，case 顺序每块随机：

- P24/G目标0/Cost-aware：不划算输入的负对照。
- P32/G目标0、50/Always、Cost-aware：短 Gap 的正向候选。

共12条校准、200条测量、100组配对。代码、配置、原始输出独立保存，
不混入旧批次、不更新原报告。置信区间仅反映本会话配对，不代表跨会话稳定性。

准备率和选 HTP 率的分母为正式测量中 `requested_route=router` 的决策次数，
不包括校准和强制 CC/HC。等待成功率/超时率的分母为实际发起等待的次数；
分母为零时记 null。还记录 pending 时等待比例、准备后仍选 CPU 比例以及逐 H/P/G 的统计。
“准备后选 CPU”不能直接换算为能耗损失，本实验不测能耗。

## 入口

```sh
python scripts/profile/pending_wait.py --output results/<具名有限等待目录>
python scripts/analyze/pending_wait.py results/<原生运行目录> --output results/<具名目录>/summary.json
python scripts/analyze/pending_wait.py --rates-only results/<运行目录> --output results/<具名目录>/trigger-rates.json
```

设备入口默认 Termux SSH、显式 hybrid、F16 KV、ctx6144/batch128、CPU 4线程 decode，
关闭 KV 整理、通用预搬移与 profile；持锁、串行运行且验收后释放锁。
当前配置为 `configs/pending-wait-h4096-termux.json`。

旧 Termux 主矩阵54次 Cost-aware Router 决策中，提前准备54次（100%）、
选择HTP12次（22.2%）、准备后仍走CPU42次（77.8%）；统计见
`results/20261106-cost-aware-trigger-audit/previous-trigger-rates.json`。
