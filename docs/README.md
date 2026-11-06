# 文档索引

命令均从仓库根目录执行。`results/`、`archive/` 中的历史路径与原始 manifest 不改写，复现旧版本时按其 Git 提交查看。

| 分类 | 文档 |
|---|---|
| design：路线与架构 | [开发路线](design/roadmap.md)、[执行契约](design/architecture.md)、[目录与模块边界](design/layout.md) |
| guides：运行指南 | [环境与回放](guides/run.md)、[常驻 Tool 会话](guides/tool-session.md)、[真实 HTTP Oracle](guides/oracle.md)、[首批主图采样](guides/observation-main.md)、[KV tiles 预整理](guides/kv-prepack.md)、[NPU 执行计划预测准备](guides/execution-preparation.md)、[BFCL 采集](guides/bfcl.md) |
| validation：验证证据 | [选路、联合 Oracle 与 KV 预整理结论](validation/oracle-summary.md)、[结果与限制](validation/results.md)、[早期异步夹具](validation/async.md)、[多轮状态基线](validation/multiturn.md) |
| reference：参考与保留路线 | [后端评审](reference/backend-review.md)、[HTP P32 阶跃调研](reference/htp-kernel-threshold.md)、[QNN 视觉（暂停）](reference/qnn.md)、[第三方声明](reference/third-party-notices.md) |

返回 [项目首页](../README.md)。

无手机开发入口：[Self-Calibrating Runtime：自预热、抽样计时与在线成本更新](guides/self-calibrating-runtime.md)。

当前待办：[未完成工作与交付边界](design/remaining-work.md)。

最新设备机制记录：[自校准与执行计划准备：14 个短条件](validation/self-calibrating-device.md)。

主实验平台切换：[SM8650 迁移验证](validation/platform-migration.md)。

真实任务反馈：[手机单闭环测试](validation/single-loop.md)。

CPU基线补充：[图1的CPU线程实验（P≤64）](guides/cpu-thread-study.md)。

灭屏运行入口：[Termux SSH、依赖修复与持锁验收](guides/termux-ssh.md)。

短Gap机制：[按成本有限等待与Cost-aware触发率](guides/pending-wait.md)。

- [P 分布与投机准备期望价值](design/p-distribution-value.md)

- [第一张曲线补测 OpenCL GPU：实验与结果](validation/gpu-study.md)

- [CPU / HTP / OpenCL decode 独立测量与曲线](validation/gpu-study.md)
