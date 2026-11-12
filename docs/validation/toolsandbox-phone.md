# Qwen3.5-9B Q4_0 手机服务验收

小米 14 Pro 通过 Termux SSH 部署，手机运行显式 CPU llama.cpp HTTP server，wins 运行 ToolSandbox 并直接向手机发送推理请求。9B Q4_0 模型与程序哈希核对一致；两轮 JSON、会话码记忆通过。

5 个不同基础场景串行完成，14 次生成、9 次工具调用、534 个输出 token；全部正常 EOS，无上下文/输出截断，原始 token 计数和工具时间线检查通过。官方平均相似度 0.7554，满分为 0/5。

| 场景 | 官方分数 | 工具次数 |
|---|---:|---:|
| Wi-Fi 查询 | 0.8816 | 1 |
| 联系人电话号码查询 | 0.9817 | 1 |
| 最近消息查询 | 0 | 1 |
| 按手机号删除沙盒联系人 | 0.9654 | 2 |
| 蜂窝关闭时向联系人发信 | 0.9484 | 4 |

最近消息任务空条件调用被工具拒绝，模型转为追问，因此质量失败。发信任务从蜂窝关闭错误中恢复并重试成功。其余四组工具/状态里程碑均为 1.0，最终自然语言相似度低于满分。保留原始官方评分，不将轨迹完整解释成任务质量通过。

实测 H=300–1018，工具正文 P=1–82 token，G=43.4–78.8 ms，每次 D=9–69 token。另记录实际追加 P、公共前缀、服务器缓存量、prefill/decode、主机计量准备和请求耗时。G 是 wins 工具可用窗口，不能当成手机网络空闲窗口。

本地完整证据：[实测报告](../../results/20261106-qwen35-toolsandbox/report.md)、[验收 JSON](../../results/20261106-qwen35-toolsandbox/acceptance.json)、[逐工具 CSV](../../results/20261106-qwen35-toolsandbox/final/five-groups-final-v2/tool-metrics.csv)。远端原始目录为 `/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261106-qwen35-toolsandbox/`，失败及中断试跑另行保留，正式统计仅使用 `five-groups-final-v2`。

仅覆盖 CPU 文本推理、单用户轮基础场景与同进程历史，用户端在首个最终回答后确定性结束。未验证视觉、HTP、跨后端状态迁移或调度收益，不代表完整 ToolSandbox 多轮基准。服务已停止，实验锁/wake lock 已释放，模型与程序保留。复现和指标定义见[运行指南](../guides/toolsandbox-phone.md)。
