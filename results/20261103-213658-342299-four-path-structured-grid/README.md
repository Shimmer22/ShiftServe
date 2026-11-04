# 四路径首轮校准

当前主线为Cost-aware Hysteretic Scheduling＋Slack-aware State Preparation。KV优化暂缓。本次只接入四路径与成本测量，未实现在线迟滞、D/Gap预测或异步准备。

96次串行测试，全部返回相同完整wait JSON与相同生成ID（每次6个前向评估，含EOG）。60次数值通过，36次不通过；completion.json为passed=false。逐步阈值仍为cosine≥0.999且NRMSE≤0.05，最差cosine=0.997824、NRMSE=0.0669311。未降低阈值补齐选路表。

配置：H1024/4096、P16/64、初始驻留CPU/HTP、四路径，各3次。GPU ViT公共阶段未包含；文本历史为结构化合成观察，非视觉KV。CPU/HTP双上下文常驻、F16 KV、ctx5120、batch128、CPU extra repack关闭、HTP queue/opbatch=1。初始化、历史构造、fixture恢复与核验不计入路径成本。

## 通过校验的CC/HC对照

H4096、初始CPU、G=0，完整短动作状态就绪成本的中位数：

| P | CC | HC |
|---:|---:|---:|
| 16 | 1431.6ms | 2702.6ms |
| 64 | 3929.3ms | 1924.7ms |

两种P下CC/HC各3次均通过；这说明小P留在CPU值得保留为候选，但没有隔离出“迁移成本独自改变阶段最优”的因果证据。HH在这些条件数值未通过，不能宣布四路径全局最优。

Gap=0/25/100/500ms仅离线模拟初始状态准备的理想重叠；尚未建立四路径全部通过后的最优路径翻转证据。D=1/2/4只是相同输出轨迹的计算前缀，不是三种完整响应长度。完整成本含EOG提交，不等同于流式动作可用时间。

## 文件

- H*.jsonl：全部原始测量和数值指标；checked-rows.jsonl：host响应检查。
- path-analysis.json：失败行、各路径成本及校验状态；path-costs.csv只含合格试验。
- figures/paths-cpu.png、paths-htp.png及PDF：完整动作成本；只画通过校验且样本数充足的点。qa.json无布局重叠/越界。
- ../four-path-memory-comparison.json：独立真实截图四轮与全CPU对照，完整token及prefix一致。

连续三截图回放曾再次FastRPC停顿，仍未根治。首帧截图＋后三轮历史追问的四路径成功不代表连续新截图问题已经解决。真实Tool并发、GPU ViT期间的准备基线和资源争用留待下一轮；不能将离线隐藏几十毫秒当作Tool Gap改变最优执行路径的证明。
