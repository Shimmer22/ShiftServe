# BFCL 历史工具输出注意力试跑

本次仅验证历史工具输出是否占用较多 token、但在后续 decode 中获得较少注意力。**不执行 KV retire，也不把低注意力解释为可安全删除。**

- 数据：官方 `BFCL_v4_multi_turn_long_context.json` 的全部 200 任务。
- 模型：远端已有 Qwen3.5-9B BF16；不下载、不量化、不修改权重文件。
- 设备：`ssh wins`，`10.29.243.183`，RTX 4090，CUDA0；不使用手机。
- 模型原生工具模板，关闭 thinking、greedy decode；每轮最多 20 次模型调用，单次最多生成 1024 token，完整上下文上限 65536。超限保留为失败，不截断。
- 工具复用 BFCL 内存沙箱，只执行白名单直接函数及 literal 参数。评分复用官方 `multi_turn_checker`。
- vLLM CUDA 后端，8 个并发任务，原生 prefix caching 与 chunked prefill；逻辑历史完整。没有 KV retire、压缩、预搬移或 CPU 推理回退。

## 最小测量口径

只观察 8 个 full-attention 层；24 个线性注意力层不提供这里的逐 token 注意力分布。先用 vLLM 采集精确 prompt/generated token IDs，再释放 vLLM，用同一 BF16 权重在 Transformers 中 teacher-forced 回放，每次调用均匀采样最多 16 个预测 query（包含预测第一个生成 token 的 query），沿层和头汇总。回放采用 2048-token chunked prefill，同一上下文保留全部缓存，并核对采样位置的 greedy-token 一致率。注意力属于 HF 回放诊断，不冒充 vLLM 内核直接导出的数值。

对工具结果正文 token 求和得到 attention mass，其分母是每个 query 所有可见 key 的注意力总量，包含 system、工具文档、用户、assistant、其他工具结果和已生成 token。保存每个输入 token 的平均注意力，按真实模板位置标记工具正文，避免相同返回字符串误匹配。

工具结果分别统计：所有后续模型调用，以及后续用户轮次。对同一结果跨调用按 layer-query 数加权；同时保留单次调用平均 mass 的最大值。阈值为 0.1%、0.5%、1%、5%，报告低于阈值的结果比例和它们占已观察结果 token 的比例。未被后续调用观察的结果不进入分母；通过和未通过官方评分的任务分开。

这仅验证注意力现象，不测因果贡献、退役安全性或工具返回是否超出模型预期。全任务平均和观察期最大值属于离线诊断，不能当成 Tool Gap 当时可用的在线决策信号。

## 本次运行

正式远端结果：`/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261106-qwen35-9b-bfcl-lc200-vllm`。

私有运行目录：`/home/wins/ai-workspace/private/a2o-bfcl-attention`；容器：`a2o-bfcl-lc200`；复用镜像 `docker.1ms.run/vllm/vllm-openai:v0.31.0`。vLLM 服务只监听 `127.0.0.1:18766`，`--language-model-only --max-model-len 65536 --max-num-seqs 8 --gpu-memory-utilization 0.90 --enforce-eager`。

入口：`scripts/collect/bfcl_vllm.py`；配置：`configs/bfcl-attention-qwen35.json`；回放：`scripts/analyze/bfcl_replay.py`；分析：`scripts/analyze/bfcl_attention.py`。

HF 串行启动预检保留在 `20261106-qwen35-9b-bfcl-lc200`；vLLM 32K 预检保留在正式目录的 `vllm-32k-preflight/`。均不计入正式 200 任务。

正式验收检查：200 个结果文件齐全；官方质量与异常分开统计；输入、输出、工具结果和注意力原始向量可关联；失败输出保留；不以进程退出码代替任务通过率。

已接续后台执行链：`scripts/collect/finish_bfcl_pipeline.py` 等本次采集进程退出，独立核对 200 个任务结果，停止本次私有 vLLM 服务，串行执行回放和汇总，最终写 `acceptance.json`。异常写 `pipeline-failure.json`，不删除原始输出。该执行链不创建系统计划任务或 Codex 定时唤醒。

后续离线分析：[>1024-token 工具结果的密度、块级集中度和跨轮稳定性](../../results/20261106-qwen35-9b-long-tool-results/report.md)。只复用已保存向量，没有重跑任务或模型回放。
