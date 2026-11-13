# 文档索引

当前默认文本模型：**Qwen3-8B Q4_0**，见[模型准入与关键成本扫描](guides/qwen3-model-costs.md)。旧模型文档保留其实际实验身份；HTP均需按具体配置区分offload与算子覆盖。

新模型实测：[准入、双驻留换页诊断与单驻留正式扫描](validation/qwen3-model-preflight.md)。

本轮完成：[Qwen3-8B固定配置成本表验收与绘图](validation/qwen3-key-costs.md)。

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
- [BFCL 历史工具输出注意力试跑](guides/bfcl-attention.md)
- [Qwen3.5-9B Q4_0 手机服务与 ToolSandbox](guides/toolsandbox-phone.md)
- [ToolSandbox 五组手机实测](validation/toolsandbox-phone.md)

- [Qwen3.5 手机三条件端到端比较](guides/toolsandbox-route-e2e.md)

- [十题三条件审计：精度与人工干预](validation/toolsandbox-ten-audit.md)

- [H/P 线性成本路由与实验接续](guides/linear-route.md)

冻结边界实验优化：[BFCL 快照缓存、直连采集与串行接管](guides/bfcl-cached-replay.md)。

工具返回与 Gap 扩展分析：[BFCL / ToolSandbox 格式、P 分组与共享前缀](validation/tool-return-structure.md)。

PromptWeaver 对照：[长工具说明与短字段前缀的收益边界](validation/promptweaver-assessment.md)。

主机共享前缀验证：[工具说明与历史布局的四条件检查](guides/prompt-layout-host.md)。

布局共享实测：[18 个新增函数边界、四条件主机结果](validation/prompt-layout-host.md)。

机制候选与失效审计：[输入稳定性、状态检查点与缓存损失分解](design/cache-stability-mechanism.md)。

固定历史分阶段扫描：[H=4096 的 CPU/HTP P 与 D 耗时曲线](guides/pd-stage-sweep.md)。

τ²-bench：[主机并发轨迹、M(H) 补测与 always-prepare 手机 pilot](guides/tau2.md)。

τ²-bench 验收：[54 次测量、短 Gap 未就绪与成本模型边界](validation/tau2-pilot.md)。

τ²-bench 贪心接续：[18 次继续准备成功、HC 实测与阶段 Oracle](validation/tau2-greedy-return.md)。

CPU Prefill 锯齿调查：[ARM 双行内核与列切块回退](validation/cpu-prefill-sawtooth.md)。

HTP 平台探索：[HVX/HMX 切换、DeltaNet 64-token 分块与布局探针](validation/htp-platform-exploration.md)。

31行补齐验证：[真实Q4权重下的31→32补零、裁剪与数值对照](validation/pad31-matmul.md)。

NPU草稿成本检查：[固定FFN宽度、CPU验证与recurrent回滚](validation/npu-draft-key-probe.md)。

NPU decode慢因：[CPU/HTP拆图、融合DeltaNet与批量提交实测](validation/decode-attribution.md)。

NPU decode组合优化：[充分预热对照、JSON记忆与检查点验收](validation/npu-decode-combined.md)。

第二机制探索入口：[NPU草稿、CPU验证：设想、证据与未完成项](design/second-mechanism.md)。

NPU优化叠加稀疏：[同轮CPU对照、固定FFN裁剪与打包成本](validation/npu-decode-stacked-sparse.md)。

优化后decode细分：[CPU逐节点、HTP逐算子与KV历史长度对照](validation/npu-decode-fine.md)。
