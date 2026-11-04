# CPU / NPU 阶段分工：文献与当前工程对照

调研日期：2026-11-03。只读论文、作者仓库和已有本地结果，本轮未运行手机实验。

## 结论

当前“所有有效点都选择HTP prefill”符合已有工作的常见方向。真正需要解释的是绝对吞吐低、相对CPU优势只有约1.6–2.3倍，以及8K重复文本fixture的空响应。B固定CPU decode，没有实测HTP decode，不能将它作为“CPU decode更优”的新证据。此前据没有路由翻转推断复杂调度价值不足，证据范围过窄，应修正为：当前执行配置和已通过的工作负载范围内没有翻转。

## 直接相关的一手来源

| 来源 | 读到的实证与实现 | 对我们的意义 |
|---|---|---|
| [llm.npu，ASPLOS 2025](https://arxiv.org/html/2407.05858v2) | QNN/MLLM，固定chunk、NPU友好量化、离群值处理和异序子图；默认chunk256，CPU decode被表述为易于实现的选择 | 不是把原GGUF直接放上HTP；选择CPU decode也不等于完整测量证明CPU总是最快 |
| [SplitLM，作者仓库](https://github.com/dastin359/SplitLM) | X Elite 32GB笔记本；Qwen3.5-2B表中CPU/NPU prefill 423/1614 token/s，CPU/NPU decode 55.4/29.0；NPU测量batch1024；有融合kernel、Flash Attention、共享权重 | 模型、硬件、batch和内核都不同。仓库同时列出按吞吐计算的请求时间与实测请求时间，应区分引用；main持续更新 |
| [Is Your NPU Ready for LLMs?，2026预印本](https://arxiv.org/html/2607.05475v1) | 同一OnePlus15/Qwen2.5-1.5B：GENIE NPU prefill1463.7、llama.cpp NPU115.1 token/s；分析整图/分段执行及大矩阵kernel扩展性 | 同名NPU后端的性能可差一个数量级，不能用当前ggml结果代表硬件极限 |
| [When NPUs Are Not Always Faster，2026预印本](https://arxiv.org/html/2605.27435v1) | Snapdragon8Gen3、llama.cpp b7588、batch128；反而测得CPU prefill最多快1.6倍，研究调用开销与跨后端回退 | 阶段分工并非跨所有后端版本成立；这类反例首先是运行时现象，不能直接上升为硬件规律 |
| [HeteroLLM，作者论文](https://arxiv.org/html/2501.14794v1) | QNN+GPU，研究shape/order敏感、张量切分与快速同步；CPU主要作为控制面 | “异构推理”不只指CPU decode；量化格式、图形状和执行粒度比后端标签更具体 |
| [OpenWeights，作者测量](https://experimentalmachines.org/openweights/prefill-decode/) | 同一llama.cpp构建比较多种阶段配对；区分短回复和持续长回复，观察到热状态下最佳decode后端变化 | 不能只用一次冷态吞吐或默认线程数宣称CPU decode普遍最优。网页不同回复长度表格必须分开读 |

## 当前工程中最值得检查的差异

### 1. 排障配置尚未校准为性能配置

`configs/prefill-grid.json`固定batch128、HTP opbatch=1、queue=1。`Engine`固定LLM Flash Attention关闭。相邻共享源码`ggml-hexagon.cpp`中的默认opbatch=1024、queue=16；本次原始日志确认实际n-ops=1。源码默认值仅用于说明差异，不假定共享源码和已部署二进制完全对应。

这些参数来自此前FastRPC/数值问题的排障。P1024实际被分成8个128-token调用，增加P不会变成一次1024-token GEMM。HTP日志还报告graph splits=291（bs128）；拆图可能带来频繁边界开销，但不能由split数量直接算出耗时或判定所有matmul的设备归属。需逐算子profile确认。

实际H141/P1024，CPU约40.1、HTP约62.5 token/s（纯prefill）；H4096/P1024为18.6/31.4。不能直接与无长prefix、不同模型的论文吞吐相除当成可实现提升。

### 2. 内存压力是已有证据，不是泛泛担忧

本轮CPU映射权重1656.22MiB；HTP repack权重1488.38MiB，另有CPU侧166.92MiB；两套ctx10240 F16 KV共720MiB，尚未计入计算buffer和临时snapshot。CPU和HTP权重重复驻留。

H4096/P1024第一重复结束时，进程VmSwap约683MiB，累计major faults为1,039,399；H8192/P1024对应约1055MiB和2,705,183。major faults是累计计数，不能据此直接认定发生同等数量磁盘读取，也不能归因到CPU或HTP某一阶段；但足以否定“这是无内存压力的稳态算力比较”。mmap CPU权重和HTP重排权重的驻留差异可能造成不对称影响，方向与幅度尚需隔离。

### 3. CPU基线被关掉一类优化

`mp.use_extra_bufts=false`。本地llama-model.cpp明确该开关控制CPU extra buffer类型；本轮CPU日志为CPU_Mapped，HTP仍正常使用HTP0-REPACK。因此不能误读为HTP repack也被关闭。CPU可能失去适用的ARM重排GEMM优化，但实际收益必须测量，不能假定开回就必然变快：额外权重内存也可能恶化当前压力。CPU固定4线程，也没有经过最优线程数校准。

### 4. 当前矩阵没有覆盖最可能的切换区域

P从64开始，未测1/4/8/16/32-token增量。通过校验的历史只有H≤4096，8K全部排除。较短增量与长prefix可能更接近带宽/固定开销受限区；这是待验证假设，不是现有结论。

更直接地，对固定H/P，如果G=0时HTP总成本已经低于CPU，那么增大G只会降低或保持HTP成本，按现有公式不可能反转成CPU。因此四张Gap图同选HTP不是独立的四次调度发现。

当前通过点准备中位数4.14–101.57ms，对比秒至数十秒的prefill，A2O能隐藏的仅是前者。权重repack在初始化，当前F16序列接口没有外置KV tiling；不能将QNN工程的编译/布局准备成本移植到本表，声称本表也应有同量级准备窗口收益。

### 5. 8K空响应与异构收益是两个问题

原始CPU上下文在restore之前已生成EOG，restore后logits逐值一致。空响应不由本次snapshot恢复引入。重复note fixture应替换为有意义的历史，并继续维持输出和数值检查；这不排除HTP另外存在长上下文数值误差。

## 最小后续验证顺序

不重跑全矩阵，先用一个有意义短prompt做单模型CPU/单模型HTP对照，记录逐阶段资源差分与吞吐；CPU比较extra repack开/关，HTP从已通过的排障参数逐项恢复批量/队列，保持输出检查。再加回双模型双context，识别驻留造成的差异。最后只补短增量P=1/8/32与一个长prefix，检验真实路由边界。

先确认运行时没有拖慢两端，再讨论是否需要更强的QNN Prefill。更换QNN需要独立验证KV格式、位置和多轮交接，不能只拿单次吞吐替换当前路径。保留A的基线；C/D不必因本次调研扩大范围。

## 本地证据

- `results/20261103-201017-725874-prefill-grid/manifest.json`、`H*.log`、`H*.jsonl`、`analysis.json`。
- `results/20261103-205410-506448-long-context-restore-check/H8192.jsonl`。
- `src/a2o/native/agent_runtime.cpp`、`src/a2o/native/profiling/prefill.h`、`src/a2o/scheduling/cost_model.py`。
- `../shared/runtime/llama.cpp/src/llama-model.cpp`与`ggml/src/ggml-hexagon/ggml-hexagon.cpp`只读核查；未修改共享运行时。
