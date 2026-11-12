# BFCL 无图像文本 Agent 预检

本次采集以三个官方 multi-turn 类别的随机小样本检验 P 分布与任务质量，避免直接使用旧 Qwen2.5-VL-3B Q4 轨迹（旧 800 任务仅 22 题通过）推荐正式场景。

- 配置：`configs/bfcl-text-pilot-qwen35.json`，每类固定 seed 随机 20 题，共 60 题。
- 模型：复用 wins 已有 Qwen3.5-9B BF16，vLLM 0.31.0，显式 CUDA0，仅主机工作负载刻画。
- 关闭视觉：`--language-model-only`，所有请求无图像，单任务串行，prefill/decode 在同一服务进程中完成。
- 接入：官方 tokenizer 的 `apply_chat_template(..., tools=..., enable_thinking=False)`；保存精确 prompt 和 generated token IDs；离线与 vLLM 官方 `qwen3_coder` 工具解析器比对函数名、参数和调用顺序。
- 原始数据、函数文档、官方沙箱及 `multi_turn_checker` 保持不变。缺函数场景按规定轮次开放函数。
- greedy、单次 1024 token 上限、32K 上下文是本次预检设置；与官方通用非思考采样建议不同。超限保留为失败，不截断后继续计为成功。
- BFCL 是 API 型文本 Agent，消息/行程/车辆等为内存模拟，不是真实 Android UI 基准。G 是沙箱进程 dispatch/return，不能解释为真实网络等待。

`P_body` 是每条工具返回正文 token 数。`P_append` 是下一轮输入与上一轮 prompt+实际 generated IDs 的最长公共前缀之后的全部 token，包括消息模板。批量多工具共享一次返回边界，P 分布按 `(task_id,next_step)` 去重。H 为公共前缀长度。统计将全部任务、完整任务通过/未通过分开；通过样本只用于机制重放候选，不代表无偏任务准确率。

入口：`scripts/collect/bfcl_vllm.py`；宿主编排：`scripts/collect/bfcl_text_pilot_host.py`；汇总：`scripts/analyze/bfcl_text_pilot.py`。

结果目录：`results/20261108-bfcl-text-pilot/`。完整原始日志保留在 wins `/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261108-bfcl-text-pilot`；私有运行源码在 `/home/wins/ai-workspace/private/a2o-text-pilot`，不修改共享权重或模型工具链。

后续 CPU/HTP 复现必须重新核对手机 tokenizer、上下文与状态迁移，主机 P 不直接冒充设备 KV 或延迟。多数移动领域任务的 H 超过 4K，不能截断或删除文档来伪造适配；优先评估 8K 原样上下文的内存可行性。
