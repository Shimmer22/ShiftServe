# 多轮状态基线与当前四路径扩展

当前论文主线为 Tool Slack 投机准备与返回后延迟绑定；GPU ViT＋CC/HH/HC/CH 是已有正确性与成本校准组件，见[ROADMAP.md](../design/roadmap.md)。`four-path-session.json`显式指定每轮Decode后端，snapshot由实际Decode后端导出。下文记录此前固定CPU Decode的基线设计与结果。

此前优先建立多轮正确性基线。ViT 固定 OpenCL GPU，关闭视觉 FA；不继续 QNN/NPU ViT 编译。不启用 KV 整理、预搬移、模拟 A2O 等待或自动调度策略。默认模型仍为手机已有 Qwen2.5-VL-3B Q4_0 + F16 mmproj。

## 路线分析

用户提出的三个 Observation 组成一个待验证的问题：历史状态的表示及其就绪程度，是否足以改变下一轮的最优 prefill 后端。异构速度优势不能脱离输入长度、历史规模、回传和状态准备成本单独比较。Tool 空档可以隐藏准备成本，但异步工作对前台任务造成的延迟也必须计入。

以下三项是工程证据顺序，论文 Innovation 以 ROADMAP 的代价模型、两阶段决策、异步运行时为准：

1. **多轮完整状态基线**：先证明 CPU decode 后的状态能被 CPU/HTP 等价接收，历史图片不重复编码。
2. **条件化收益边界**：改变历史规模、新输入/输出长度，计入 snapshot/restore、返回及 decode，找到实际 crossover；当前 GPU 视觉保持相同。
3. **准备与调度耦合**：接真实 Tool 间隔，开始时判断是否投机准备，返回后按真实输入与准备进度选择后端，验证取消和 CPU 安全回退。压缩率与硬件布局优化留作基础机制成立后的可选扩展。

Workload-aware Cost Modeling、Speculative Preparation with Late Binding、Asynchronous State Preparation Runtime 目前是研究设计，不作为已证明的贡献。简单 recent 选择器不是当前重点。

## 借鉴 SplitLM

检查版本：`dastin359/SplitLM@0a5105bb03c643e56f8b585f1c930f665ac2e715`。

- [engine.cpp](https://github.com/dastin359/SplitLM/blob/0a5105bb03c643e56f8b585f1c930f665ac2e715/src/engine.cpp)：`cache_lookup` 找最长的严格 token prefix；cache entry 保存完整 sequence snapshot；CPU/NPU lane 在 restore 后只 prefill suffix，并用同一 sequence-state 接口交接。
- [engine.h](https://github.com/dastin359/SplitLM/blob/0a5105bb03c643e56f8b585f1c930f665ac2e715/src/engine.h)：缓存命中长度与新增 prefill 长度分开记录。
- 该工程包含针对递归状态和内核的 llama.cpp 补丁，运行平台为 Windows Snapdragon X；不将其性能或 Qwen3.5 支持直接当成当前 Android 运行时的能力。

本工程沿用 `llama_state_seq_get_data/set_data`。先维护单个活动会话的完整 prefix，不建立多用户缓存服务或重写 SplitLM 全部引擎。多模态状态除文本外还依赖历史图片身份和 M-RoPE；因此同时保留实际模板前缀、图片 SHA、真实 sequence snapshot 与独立的 `next_pos`，不把文本 token 长度当作图像的位置编码。

## 当前执行契约

每个会话在一个 Android 进程内持有 CPU 和 HTP 上下文。每轮执行：

1. 用真实上一轮 assistant 输出构造完整模板，只取精确 prefix 之后的新 suffix。
2. 新截图仅在 GPU 编码一次。保留新增 embedding 和公开 API 给出的 `[t,y,x,z]` 位置；随后释放视觉权重/图缓冲区，再进行 LLM prefill。
3. 诊断模式下，CPU 和 HTP 从同一份上一轮 CPU snapshot 开始，接收相同新增 token、embedding 和位置，比较完整末尾 logits。
4. 按显式配置选择 CPU 或 HTP prefill 结果；HTP 结果完整交接给 CPU 后再 decode。
5. CPU 实际评估生成的 Action token 和结束 token，随后 snapshot。下一轮直接恢复，不重算旧图或旧 Action。

`cached_tokens` 表示真实已评估的历史状态，`next_position` 表示 M-RoPE 下一位置，两者不可混同。每轮输出、原始 prompt/prefix、CPU/HTP logits 和阶段时间保存在结果目录。CPU/HTP restore 后重新导出，核对完整序列字节；数值比较另记 cosine、NRMSE、最大误差和 top-1。

当前采用 ctx4096、batch128、CPU mmap、不做 CPU extra repack。HTP queue=1、opbatch=1 是针对 FastRPC 停顿的串行诊断配置；尚不将其看作底层可靠性问题已被修复。`compare_prefill=true` 会执行两份 prefill，整轮时间不能当作正式性能数据。

## 入口

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run/agent.py \
  --config configs/multiturn-baseline.json \
  --scenario scenarios/multiturn-screenshots.json --tag multiturn-baseline
```

四轮选择 CPU→HTP→CPU→HTP，decode 均为 CPU；前三轮是真实已保存截图，最后一轮纯文本追问历史。会话码只在首轮提供，后续检查记忆及上一条实际 Action 中的 step。不会执行生成的设备动作。

**DeltaNet 边界：** 当前 Qwen2.5-VL 是全注意力模型，没有 DeltaNet/递归状态。本轮通过不代表 Qwen3.5 通过。切换 Qwen3.5 时，需在对应运行时上验证完整 recurrent/conv state、位置和 KV 的交接；不能只拿 KV 字节一致替代这项验证，也不预先裁剪递归状态。

## 本轮结果

已完成两条四轮真机回放：CPU/HTP 切换与始终 CPU prefill。两者的完整生成 token、输出和 prefix 字节逐轮相同；历史状态从 0 增长到 2083 token（含三帧视觉 token），每张截图只编码一次。四轮 logits top-10 集合均一致。结果与边界见 docs/validation/results.md；离线汇总命令为 `conda run -n visiontrim-qwen25vl python scripts/analyze/multiturn.py`。
