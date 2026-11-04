# 响应诊断与验证

2026-11-03，Asia/Shanghai。设备：Xiaomi 15 / SM8750 / Hexagon v79。模型：Qwen2.5-VL-3B Q4_0；四线程，ctx2048，F16 KV，LLM Flash Attention 关闭。

**本轮设备测试已完成。** 固定真机截图回放、CPU/HTP 后端定位、混合内存诊断和四条件历史截图消融均已完成。最终四条件全部通过，设备上没有遗留本次 runtime 或临时目录。当前地址为 `10.29.11.213:32927`。

## 原输出问题

原自写工具将分段文本直接 tokenize，没有构造 user/assistant 消息边界与 assistant generation prefix；在原上下文上直接预测到 `<|im_end|>`。之前的 mtmd CLI 使用 `common_chat_format_single` 格式化消息，入口不同，因而先前正常响应与后续裸文本立即结束并不矛盾。

此次关闭所有 KV 整理、预搬移与状态导出，以同一原始历史和 next prompt 做诊断：

| 输入路径 | 结果 |
|---|---|
| HTP0，原始 191 token 裸文本，原地推理 | 首 token 为 `<\|im_end\|>`，普通输出为空 |
| HTP0，同一上下文套模型 chat template，原地推理 | `Toggle the Wi-Fi switch to enable it.` |
| CPU，正确模板，要求 READY | `READY` |
| HTP0，正确模板，要求 READY | `READY` |
| CPU，同后端 seq 导出/恢复，再追加观察 | `WIFI` |
| HTP0，同后端 seq 导出/恢复，再追加观察 | `WIFI` |

这组结果将原问题定位到提示协议：不需要删除 KV、position 空洞、序列化或跨后端迁移，就能重现立即结束；修正模板后响应恢复。不是由坏的 KV 选择才能触发的故障。

运行器现在使用模型内置模板和 assistant 起始标记，EOG 正常停止但不作为响应文本；空响应使 host 校验失败。prefill/decode 阶段在一个进程内执行，不再每阶段重新加载模型。

## 八条件文本闭环

统一历史含 140 token，测试目标为识别任务中的无线连接名称。`recent` 保留 system、完整上一动作与最近历史，删除 95 token，留下 45 token。所有条件都在手机上实际运行，无后端回退。

| LLM 路径 | KV | 预搬移 | 回答 | 状态检查 |
|---|---|---|---|---|
| CPU | none | 关 | `WI-FI` | 正常响应 |
| CPU | recent | 关 | `WIFI` | 正常响应 |
| HTP0 | none | 关 | `WI-FI` | 正常响应 |
| HTP0 | recent | 关 | `WIFI` | 正常响应 |
| hybrid | none | 关 | `WI-FI` | CPU→HTP→CPU 字节一致 |
| hybrid | none | 开 | `WI-FI` | CPU→HTP→CPU 字节一致 |
| hybrid | recent | 关 | `WIFI` | CPU→HTP→CPU 字节一致 |
| hybrid | recent | 开 | `WIFI` | CPU→HTP→CPU 字节一致 |

四个 hybrid 条件真正执行了：CPU 历史 KV → HTP 新观察 prefill → 更新状态回到 CPU → CPU 继续 decode。首 token 从 HTP prefill logits 选取；序列导入不恢复 logits，CPU 从评估该首 token 开始继续生成。

未压缩的上行/下行序列状态分别为 5,164,648 / 6,308,052 字节；recent 为 1,660,668 / 2,804,072 字节。每次迁移重新导出目标状态，与源进行完整字节比较，覆盖 KV 和位置元数据。

原场景要求严格四字母 `WIFI`，所以 `WI-FI` 在原始结果里记录为格式失败：严格格式通过 4/8，连接名称语义正确 8/8。原始检查和输出均保留。当前场景显式接受 `WIFI`、`WI-FI` 两个拼写，并单列 strict_format_passed；没有归一化或改写模型输出。

总表：`results/validation-summary.json`，由 `scripts/analyze_results.py` 离线生成。旧 result.json 的严格检查没有覆盖；总表并列初始检查与当前连接名检查。

## 视觉路径的独立问题

固定输入为 `inputs/phone_screenshots/shot0.png`，画面标题是「开发者选项」。关闭 KV 整理和预搬移，独立交换视觉与 LLM 后端：

| 视觉后端 | LLM | 视觉 FA | 回答的 screen | 页面识别 |
|---|---|---|---|---|
| CPU | CPU | auto | Developer Options | 正确 |
| CPU | HTP0 | auto | Developer Options | 正确 |
| OpenCL | CPU | auto | current | 错误 |
| OpenCL | HTP0 | auto | 19:49 | 错误 |
| OpenCL | HTP0 | off | Developer Options | 正确 |

CPU 视觉搭配两种 LLM 的生成 token 完全相同，OpenCL 视觉关闭 FA 后也恢复相同输出。这将本例的识别问题缩小到 OpenCL 视觉 auto FA 路径；当前配置显式关闭它。没有修改共享 kernel，底层数值偏差仍需专门定位，不能把单图结果推广到所有输入。

这些基线输出带一个完整 json 代码围栏。响应适配器现支持裸 JSON 和单个完整 json 围栏，保留原始输出并标记 strict_format_passed=false；同时校验 screen 内容及 action=wait。原失败记录仍保留，离线汇总并列初始检查与当前检查。

## 混合路径的内存与调度

初始双模型截图基线在严重换页中超过两分钟，现场 VmSwap 约2.2GiB，主动终止并保留 `memory-pressure.txt` 和日志。

采用 CPU 原 GGUF mmap、关闭额外 CPU repack、batch128 后，文本混合闭环 decode 为228.6ms，decode 开始 swap 约37MiB，仅新增2次 major fault。与先前两三个 token decode 2.95–10.11s 相比，内存配置明显改善。

随后发现视觉编码器留在内存直至 decode 结束，会继续挤出 CPU 权重。在队列深度1的保留编码器对照中，17个输出 token 的 decode 为27.797s，期间新增169,378次 major fault。单观察运行器改为同步完成图像 prefill 后释放编码器；最终四条件 decode 为1.75–1.91s，major fault 增量471–942。该调整不修改 KV 或量化身份。

另一次 batch128、2秒模拟 A2O 的复测在最后图像 batch 后卡住，主线程 wchan 为 `fastrpc_dspsignal_wait`，主动终止并保留 `stall-sample.txt`。非 root 环境不支持 debuggerd backtrace；未绕过权限。混合配置随后显式设 HTP queue depth=1：一条保留编码器对照与最终四条件均返回。该配置是当前绕行方案，FastRPC 底层根因与长期稳定性仍待定位。

运行器会实时写日志，180秒超时终止经 PID 与命令行核对的本次 runtime，并保存失败状态；不自动重试或改用其他后端。

## 最终四条件历史截图闭环

统一使用 CPU mmap、batch128、OpenCL 视觉 FA off、HTP queue depth1、2000ms 模拟 A2O。文本历史141 token，新观察600 token（含560个图像 token）。`recent` 删除95个历史 token，留下46个，保留 system 与完整上一动作，原 position 不变。

| KV | 预搬移 | 历史 KV | 上传 checked ms | 回传 checked ms | CPU decode s | 观察→响应 s | 校验 |
|---|---|---|---:|---:|---:|---:|---|
| none | 关 | 141 | 7.655 | 46.556 | 1.806 | 22.104 | 通过 |
| none | 开 | 141 | 3.818 | 46.004 | 1.808 | 21.005 | 通过 |
| recent | 关 | 46 | 2.792 | 41.519 | 1.907 | 20.828 | 通过 |
| recent | 开 | 46 | 2.457 | 34.945 | 1.753 | 20.985 | 通过 |

四组均生成同一17-token裸 JSON，screen=Developer Options，action=wait；JSON、页面识别、跨后端 KV/位置完整字节比较均通过。上行 seq 为5,201,532→1,697,552字节；下行为27,331,932→23,827,952字节，下行包含本轮新观察 KV。

预搬移两组实际准备约3.818ms、2.484ms（后者含recent选择），均在2秒模拟预算内。上传/回传时间包括目标状态重新导出与字节验证，并非优化后的纯传输测量。

每条件仅一次，视觉初始化、系统负载与冷暖缓存会波动；表中约1秒的请求差异不能归因为预搬移加速。本场景的上行仅几毫秒，观察阶段仍主要花在视觉编码和图像 prefill。模型与视觉初始化单列，进程 wall 约40.7–42.0s，包含模型加载；没有把每步冷启动开销隐藏掉。

结果：`results/vision-summary.json`、`results/vision-history-latest.json`。通过 `scripts/analyze_vision.py` 离线重建，保留两次主动终止与所有失败输出。最终四条结果目录时间为14:52:29、14:53:15、14:53:58、14:54:40（Asia/Shanghai）。

## 验证范围

- 已完成：模板诊断、CPU/HTP 基础文本与状态恢复、八条件文本响应；固定真机截图的CPU/HTP视觉交叉对照；最终四条件历史截图 CPU→HTP→CPU 闭环。
- 本地：Android 交叉编译，6项响应契约测试，代码/配置/文档检查。每条运行保存 native source 与 binary hash。
- 后续：更多截图与长历史质量、匹配执行参数的重复性能对照、常驻多请求、真实动作/UI循环；FastRPC长期稳定性。

当前结果支持这条观察回放工程闭环能够正常响应，尚未用单图单次结果评价KV选择算法优劣。

## 2026-11-03：视觉 HTP 与 A2O 准备检查

本轮固定同一截图与 141 token 文本历史、600 token 新观察、batch128、CPU mmap、F16 mmproj，KV 整理关闭。所有条件均为 HTP LLM prefill＋CPU decode，视觉后端独立配置。CPU、OpenCL、HTP 按需、HTP A2O 四轮全部输出同样 17 个 token 的 Developer Options / wait JSON，CPU↔HTP 状态迁移检查通过。

| 视觉后端与准备方式 | 视觉初始化 | 视觉编码 | LLM prefill | 观察后响应 |
|---|---:|---:|---:|---:|
| CPU / 按需 | 1.81s | 37.22s | 7.32s | 48.37s |
| OpenCL / 按需 | 2.42s | 8.87s | 8.34s | 21.83s |
| HTP / 按需 | 1.58s | 45.54s | 8.43s | 57.58s |
| HTP / A2O 准备 | 1.86s | 49.99s | 8.29s | 60.14s |

以上为定位用单轮数据，非稳定性能比较。前三类执行过程中完善了准备超时计量和加入只读 capability 日志，源码 hash 并非全部相同，manifest 保留具体 hash。早两轮 `prep_overrun_ms` 包含约 2ms 睡眠抖动，后续已修正为准备工作本身超过预算的尾部。A2O 条件将视觉初始化及 KV 上传共 1864ms 放入 2000ms 模拟空档、超时为零；并未证明整体加速，更未验证真实动作并发。

诊断图记录在 `20261103-153912-535801-hybrid-htp-probe/vision-placement.json`：292 个 MUL_MAT 节点中，290 个输出缓冲区在 CPU，2 个末尾 projector 节点在 HTP；65 个 RMS_NORM、32 个 SOFT_MAX 节点在 HTP。节点计数不代表计算量或耗时占比。该图的 HTP 分配方向不符合大矩阵乘优先上 NPU 的目标。无追踪 HTP 编码仍为 45.54s，说明问题不能归因于追踪开销。

直接向手机当前二进制查询 F16 线性层能力：560/1024 token 支持，1025/2240 不支持；这张截图在视觉骨干内为 2240 patch token。因而单纯指定 HTP0 不能解决主体线性层的 CPU 分配。后续正确的工程路径是保持同一输入和精度，对线性层按 token 分块，并单独验证 HTP fused attention 的正确性与覆盖；不能通过删除后端的能力检查强行执行不受支持的大矩阵，也不能降低图像分辨率后冒充等价性能收益。

新增实现：`vision_backend=htp` 的设备预检与初始化后后端名检查；`vision_prepare=arrival/a2o`；视觉节点诊断与当前二进制 capability 查询；分离预处理、视觉编码、LLM prefill 计时。未修改共享 llama.cpp 或手机设置。七项 host 响应/配置契约测试通过。完整结果由 `scripts/analyze_npu_cpu.py` 汇总到 `results/npu-cpu-summary.json`。

HTP fused attention 补测：`results/20261103-154525-908810-hybrid-htp-fused-probe`。显式 `vision_flash=on` 时 32 层 FLASH_ATTN_EXT 输出缓冲区均在 HTP，图 split 从关闭 FA 的 578 降至 450，但观察 prefill 后检测到 `non-finite logits`，进程以失败退出；保留 `runtime.log` 和 `failure.json`，没有自动回退。此结果证明端到端数值检查不通过，尚未通过逐层嵌入对照将首个异常定位到具体 kernel，不把它当作可用优化。

## 2026-11-03：复用官方 QNN 视觉适配

使用已有 3B safetensors 视觉权重，在远端 CUDA 上复用 Qualcomm AI Hub 的 Conv3d/Linear→Conv2d、split QKV 适配。输入仍为 448×980、2240 个 patch token，输出 `[560,2048]`；FP32 对照 NRMSE=1.6749e-5、cosine=1.0。ONNX 导出使用 external data，通过格式检查。产物及编译留在远端共享输出目录，入口见 QNN.md。

手机接口回放结果：`results/20261103-161629-654584-qnn-adapted-embedding-replay`。将 CUDA 适配后的 embedding 交给原 HTP LLM prefill、CPU decode，输出与既有正确截图基线相同的 17 个 token（Developer Options / wait），跨后端 KV 检查通过。这仅验证 embedding 与 mtmd M-RoPE、LLM 的接口，不是 QNN 编码耗时或完整 NPU 路径结果。

原始全 PyTorch FP16 适配尝试未通过数值对照，保留失败，当前采用 FP32 ONNX 交给 QAIRT 以 float_bitwidth=16 转换。默认高优化编译在运行超过 10 分钟后终止，保留日志；功能验证使用显式 optimization_level=1，后续高优化待用户指定空闲时再做。

级别1的 context 编译运行约6分钟后仍未完成；按用户后续空闲时再编译的安排，已停止本轮编译并保留 DLC、环境与日志。**当前未生成可部署 HTP context，未运行手机 QNN ViT，未证明 NPU 视觉加速或同进程 QNN＋ggml 协同。** 已通过的是官方适配的 FP32 数值对照、ONNX 格式、DLC 转换，以及适配 embedding 接入现有 Agent 的实际响应。

## 2026-11-03：GPU ViT＋CPU/HTP Prefill 多轮基线

当前范围见 MULTITURN.md。模型 Qwen2.5-VL-3B，GPU ViT、CPU decode；无 KV 整理、预搬移、模拟 Tool 等待。一个进程内保留 CPU 和 HTP 上下文，ctx4096、batch128、F16 序列状态。

首轮尝试 `results/20261103-192758-308757-multiturn-baseline` 在第三轮 HTP 末尾文本 prefill 停顿，`wchan=fastrpc_dspsignal_wait`，已终止本次进程，保留完整设备工作目录、前两轮结果和原始日志。首轮 step 起点提示有歧义，已明确第一轮为 1。

最终对照（相同 native 源码和二进制 hash）：

- `results/20261103-193638-545355-multiturn-serial`：四轮 prefill CPU→HTP→CPU→HTP；同一 snapshot 的 CPU/HTP paired prefill 诊断。
- `results/20261103-194350-190510-multiturn-cpu`：四轮始终 CPU prefill，关闭 paired 诊断。仍保持相同双上下文驻留条件，不是优化到单模型内存的 pure-CPU 系统基准。

| 轮次 | 轮前缓存 token（含视觉） | 切换路径的 prefill | 新图编码数 | 输出 step |
|---|---:|---|---:|---:|
| 1 | 0 | CPU | 1 | 1 |
| 2 | 715 | HTP | 1 | 2 |
| 3 | 1355 | CPU | 1 | 3 |
| 4 | 1995 | HTP | 0 | 4 |

两条件四轮均通过页面、会话码记忆和 step 检查，逐轮完整生成 token ID、输出和已提交 prefix 字节完全一致。最终累计 2083 个实际评估 token，完整序列状态 76,830,260 字节，M-RoPE 下一位置为 508（不能用 2083 替代）。会话码只在第一轮输入中提供。三张已保存截图为同一开发者选项页面的不同帧；未执行实际设备动作。

四轮 paired prefill 各比较 151936 维末尾 logits：cosine 0.999669–0.999952，NRMSE 0.0120–0.0394，top-1 和 top-10 集合均一致；KL(CPU||HTP) 为 0.001804、0.0000667、0.0000127、0.0000092。原始 logits 保留在对应运行目录。CPU 对照没有执行数值比较，标记 numerical_compared_turns=0、numerical_passed=null。

实际输出带一个完整 JSON 代码围栏，语义/记忆检查通过，strict_format_passed=false。8 项 host 契约测试通过。跨路径对照见 `results/multiturn-path-comparison.json`，完整多轮汇总见 `results/multiturn-summary.json`。

通过条件使用 HTP queue=1、opbatch=1，并将 GPU encoder 在新 embedding 与 M-RoPE 位置取出后释放，再做 LLM prefill。两项改动同时应用，未单独归因；单次四轮成功不能证明原 FastRPC 问题根治。paired 诊断重复执行 prefill，整体时间不可用来宣称端到端加速。当前未验证 DeltaNet，因为该 2.5 模型无递归状态；Qwen3.5 必须独立验证。异步准备、增量/提前返回回退与真实 Tool 间隔均尚未实现。

## 2026-11-03：Prefill 优势区间首轮

`results/20261103-201017-725874-prefill-grid`：H=141/1024/4096/8192，P=64/256/560/1024，每点3次CPU/HTP配对，共48次。使用精确长度文本fixture，ctx10240、batch128、F16 KV、CPU raw mmap（extra repack关闭）、HTP queue/opbatch=1；双上下文同进程，视觉权重不驻留。构造历史、初始化、成对测量恢复fixture、核验均单列；不包含GPU ViT时间。独立tiling未测，记录null。

H≤4096的36次响应与数值检查通过，生成ID全部一致；logits cosine最低0.999446，NRMSE最高0.0341746。12个有效H/P点在G=0/100/500/2000ms下均选HTP；G=0计入双向交接的CPU/HTP中位成本比为1.56–2.31。各点准备中位数4.14–101.57ms，返回中位数8.34–288.51ms。查表使用每次试验总成本的中位数，不把各阶段中位数的和当成总成本中位数。

H=8192的12次两路径均立即结束、空输出，生成ID一致；其中仅P=560的3次数值阈值通过。状态导入后逐字节核验通过，但这不证明状态语义或长上下文响应正确；所有12次排除路由查表。`completion.json`保留passed=false，原始JSONL、完整历史/后缀与日志均保留。不是静默回退，也不降低数值/响应标准以填满矩阵。

首个fixture smoke要求READY，但模型两路径均返回wait JSON，保留在`20261103-200443-134736-prefill-smoke`，判为失败。随后将fixture改为一致的GUI wait任务，`20261103-200735-616323-prefill-contract-smoke`通过。正式矩阵使用后者同一任务契约；8K仍失败，不能只以退出码0判通过。

离线分析入口`analyze_prefill.py`，图表入口`plot_prefill.py`；不访问设备。9项host契约/成本模型测试通过，native运行器构建通过。Gap仅离线模拟理想重叠，未执行真实Tool并发，不能据此宣称异步准备或复杂调度收益。

固定CPU Decode只表示相同执行后端，并不保证两路径耗时严格相同。例如H4096/P1024，CPU路径decode中位数687ms，HTP返回后CPU decode为1433ms。首版路由仍不预测输出长度；这些缓存/驻留差异单列保留，后续用真实端到端延迟校正，不将其误报为状态导入时间。

8K定位补测：`results/20261103-205410-506448-long-context-restore-check`，H8192/P64、1次、相同重复note fixture。原始CPU上下文在任何restore前直接继续，与恢复后CPU末尾151936维logits逐值完全相同，NRMSE=0；二者和HTP都首先生成结束符151645、输出为空。由此可排除本次空响应由snapshot恢复引入，但不能区分历史构造、重复输入退化或更上游模型计算问题。HTP对CPU的NRMSE仍为0.0508801，超过0.05阈值，未放宽标准。诊断结果保留passed=false；所有手机进程已结束。下一步应以真实多轮历史重新建立8K正确性，而不是把失败点填入性能表。


## 2026-11-03：四路径主线首轮接入

主线改为GPU ViT固定、CC/HH/HC/CH四路径；不再强制Decode回CPU。`session_runtime.h`的plan增加显式Decode后端，首token仍从所选Prefill logits采样，跨后端恢复后不读取陈旧logits；真实生成token及EOG在所选Decode后端提交，再导出下一轮snapshot。旧plan默认CPU Decode，保留兼容。

首次真实三截图四轮测试`results/20261103-212207-593469-four-path-session`在第二轮完成前再次停在`fastrpc_dspsignal_wait`。已保存设备工作目录和wchan，终止本次进程；首轮CC完成，不能称为四路径多轮通过。没有自动回退。随后测试首轮一张截图、后续纯文本历史记忆的独立用例，以区分状态交接与连续视觉回放。

`20261103-212609-883550-four-path-smoke`：H141/P16、两种初始驻留位置，8次四路径均返回相同wait JSON，但首次诊断将提交EOG后不会再使用的logits也作为决策数值门槛，标记passed=false，原始结果未改。`20261103-212752-995772-four-path-decision-check`补查每一步实际采样的logits，全部8次通过原cosine≥0.999/NRMSE≤0.05标准；同时保留EOG后未使用的logits差异。

原重复note输入`20261103-212919-854405-four-path-grid`在H4096/P16的CPU参考中生成重复note、无EOG，因此停止该组，保留aborted.json、失败设备目录及部分原始记录。H1024的48条原始记录已保留；该组与后续结构化输入不混合。

`path_runtime.h`改用有变化的结构化合成观察历史。CPU参考先通过完整wait动作与EOG检查，再测两种驻留位置的四路径。完整JSON与逐步数值检查独立记录；Decode前缀D=1/2/4指实际token前向评估次数，不代表已经测了三种完整响应长度。完整路径成本包含EOG提交后的状态就绪时间；流式动作可用时间尚未单列。10项host契约/成本模型测试通过。


结构化组`results/20261103-213658-342299-four-path-structured-grid`完成96次，全部合法wait JSON且生成ID一致；60次同时通过逐步数值阈值，36次不通过，completion.json保留passed=false。最差cosine=0.997824、NRMSE=0.0669311。H4096、初始CPU、G0：P16的CC/HC中位数1431.6/2702.6ms，P64为3929.3/1924.7ms；这些CC/HC试验各3次通过。HH在对应4K条件未过数值阈值，因此不宣布四路径全局最优。Gap仍是离线理想重叠，未证明主线要求的最优路径翻转。

独立真实截图记忆测试`results/20261103-214617-564245-four-path-memory`：HH→HC→CH→CC，首轮截图GPU编码一次，后三轮无新截图。页面、PINE-47记忆及step1–4均通过；snapshot来自实际Decode后端。全CPU对照`results/20261103-214834-584625-four-path-memory-cpu-control`采用同一native二进制与场景，四轮完整生成ID、输出、提交prefix字节全部一致，见`results/four-path-memory-comparison.json`。轮前缓存0/715/800/885，最终970个token；该2.5模型无DeltaNet。两条件输出均有完整JSON代码围栏，strict_format_passed=false。此对照未逐步比较原始logits，不把任务一致性写成数值阈值通过。

首帧＋历史追问通过不替代连续多张新截图回放；之前FastRPC停顿仍未根治。本轮设备测试已结束，无残留agent进程；10项host检查通过。后续Tool Gap基线必须允许在GPU ViT期间准备状态，以免把现有公共窗口可以隐藏的准备成本归功于Tool Gap。

## 2026-11-04：最小异步准备机制

详见 [ASYNC.md](ASYNC.md) 与 `results/20261104-081812-212379-async-mechanism-validation/summary.json`。全 CPU 对照和异步条件各四轮通过；native 源码与二进制 hash 相同。后台状态准备、就绪选 HTP、未就绪选 CPU、旧版本拒绝及执行前取消通过；两条件输出、生成 ID、提交 prefix、计数和下一位置逐轮一致。CPU 权威快照、HTP 导入字节及 HTP→CPU 交接均检查。

本地17项测试和 Android 构建通过。没有真实 Tool、图像输入或逐步 logits 对照；事件中的动作完成仅为进程内部生成完成，完整结果文件仍在 drain 后写出。手机存在用户并行竞争，全部耗时不用于性能结论，本轮未跑长期实验。

## 2026-11-04：跨轮回收、成本 Router 与实际只读 Tool

新入口为 `scripts/run_tool_session.py`，详见 [TOOL_SESSION.md](TOOL_SESSION.md)。`results/20261104-084947-618120-tool-session-validation/summary.json` 与 `final-checks.json` 均 passed=true：CPU 对照和机制条件各六轮，native 源码/二进制相同，Tool 原文、完整生成 token、响应、prefix、计数和下一位置逐轮相同，最终521个已评估 token。

屏障持有旧任务期间，当前轮及下一轮 CPU 均已实际向消费者发布响应，之后才释放旧任务；单lane拒绝繁忙时的新准备。3个后台任务在关闭时全部回收。ready 但 CPU 成本更低，以及 ready 且 HTP 成本更低的分支均通过；分支成本是明确标记、样本数0的机制夹具，不能当作实测收益。实际 HTP prefill、回传和 CPU decode 正确；旧版本/资源所有权另有确定性 C++ 测试。

普通模式另跑三轮实际 `adb shell wm size`，禁用全部屏障与合成成本，JSON/记忆/step均通过，与 CPU 对照前三轮 token/prefix一致；实际 H/P 不在校准表内，明确记录 coverage_missing 并选 CPU。HTTP GET 适配器通过本地实际HTTP服务测试，未做外网API或Browser/GUI动作。

本地22项测试、Android构建通过。手机存在并行竞争，全部耗时不用于性能结论；未做图像回归、逐步logits阈值验证、能耗或长期压力实验。backend API仍不可抢占，关闭仍可能等待；不能据本次屏障验证声称所有共享后端锁都已排除。
