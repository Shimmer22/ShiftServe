# 优化后decode细粒度归因

当前H4096、单请求、Qwen3.5-9B Q4_0场景，扫描历史KV的attention核心不是主要瓶颈。完整FFN时它占诊断墙钟约5%–7%；FFN保留25%后约6%。CPU输出投影、DeltaNet相关CPU节点、CPU执行段的派发/等待与HTP权重矩阵乘更突出。尚未通过硬件计数器分离权重读取、解量化和算术计算，不能把矩阵乘时间直接称为纯访存。

## 同一重复内的分解

运行时使用CPU融合AR DeltaNet、HTP OPBATCH1024/OPQUEUE16、图复用开启、offload_kqv=false。CPU节点从线程0开始执行计时到已有barrier完成，不增加同步；融合节点整体计时。DSP使用安装的原版v75库profile=1，非此前改写PMU槽的路由探针。每组预热至少40秒纯decode，然后3次×8 token。本轮全部条件为CPU/HTP混合执行，CPU部分4线程。

下表取末次H4096完整组与25%组，各自中位数对应的同一次正式重复。CPU与DSP算子内部时间加外层调用差额，与本次墙钟闭合；不是跨重复独立取各项中位数。

| 部分 | 完整FFN ms/token | 保留25% FFN ms/token |
|---|---:|---:|
| HTP FFN | 60.20 | 14.89 |
| HTP DeltaNet/attention权重投影 | 26.95 | 24.86 |
| CPU输出投影 | 29.60 | 28.48 |
| CPU DeltaNet核心及相关节点 | 21.66 | 32.97 |
| CPU attention/KV核心 | 11.22 | 10.64 |
| CPU执行段减节点时间 | 42.10 | 36.24 |
| HTP调用减DSP算子时间 | 16.91 | 15.36 |
| 其他节点、拷贝、同步与剩余主机工作 | 25.30 | 8.41 |
| 总墙钟 | 233.94 | 171.85 |

CPU DeltaNet相关节点含状态选择、拼接、归一化和状态写回。25%组其中L2_NORM约13.17 ms、CONCAT约9.50 ms、GET_ROWS约2.87 ms、CPY约1.77 ms；这些时间包含既有线程同步等待，不能理解为小张量本身需要如此多的算术时间。CPU融合GDN核心另为5.65 ms。节点与外层执行段仍提示CPU小算子/线程组织有较大优化空间。

CPU执行段差额包含线程池派发、线程准备、收尾与插桩，不等于纯OS调度；HTP差额包含提交、缓存维护、队列等待、profiling/打印，不等于纯RPC。异步批次的响应时长可能重叠，不将其总和作为墙钟。每token仍有137次HTP批次提交。

## 历史长度对照

| 条件 | attention/KV核心三次ms/token | KV逻辑输入量 |
|---|---|---:|
| H512，完整FFN | 2.55 / 2.62 / 2.67 | 24 MiB/token |
| H4096，完整FFN末次 | 11.22 / 11.74 / 11.03 | 136 MiB/token |
| H4096，保留25% FFN | 10.64 / 10.61 / 10.80 | 136 MiB/token |

attention核心明确包含8层各自的QK矩阵乘、softmax和V加权，共24个CPU节点/token，另单列16次KV写入。它是处理KV的总算子时间，不是从中单独剥离出的DRAM等待时间。

H512对真实前缀前512个token重新prefill、重新计算递归状态，然后用相同P40和固定输出token；未截断H4096递归状态。H4096复用严格身份、token和字节核验的完整历史快照。实际attention KV维度按内存对齐为768/4352，8层、K/V各4头、head_dim256、F16，逻辑输入量分别24/136 MiB。该量按实际图节点形状推导，不包含GQA重复访问、cache命中和额外拷贝，不是DDR实测流量。原metadata字段logical_bytes来自ggml_nbytes，非连续view可能为地址跨度；分析时按ne元素数重新计算KV逻辑量。

当前模型只有8层完整attention，另24层DeltaNet采用固定尺寸递归状态。实际GDN输入状态每层128×128×32个F32元素，即2 MiB/层，合计48 MiB，另有状态更新等开销，不能与随H增长的attention KV混称。

## 为什么FFN裁剪后还有明显固定成本

实际权重张量统计：FFN由2592 MiB降至648 MiB；输出投影始终795.70 MiB、Q6_K、留CPU，词表248320。裁剪FFN不会减少输出头、DeltaNet状态处理或CPU/HTP切换。权重存储量仅说明潜在数据规模，不能据此推算实际DDR带宽或访问次数。

## 波动、验收与清理

| 条件，按执行顺序 | 三次正式ms/token |
|---|---|
| 无插桩完整模型前置 | 296.49 / 286.51 / 299.02 |
| H4096细分 | 318.33 / 217.27 / 180.90 |
| H4096保留25%细分 | 171.85 / 166.70 / 174.21 |
| H512细分 | 216.79 / 216.36 / 214.26 |
| H4096细分末次 | 233.94 / 252.09 / 228.14 |
| 无插桩完整模型后置 | 225.14 / 231.45 / 230.31 |

存在明显时段漂移，尤其首个H4096组不稳定。不能由两组墙钟相减精确量化H效应或插桩比例，也不把本轮诊断墙钟替换上一轮无插桩速度结论。比较的可靠部分是KV核心随H增大的方向、其量级、逐节点位置及不同固定成本的存在。计时本身可能影响线程同步，不能宣称所有分类都是无扰动的生产占比。

全部条件正常OK响应、历史恢复字节核验、有限logits与预热检查通过；正式窗口无major page fault。实际maps验证私有CPU/base/融合策略库确实加载，默认安装库哈希未变。未测试本轮稀疏自由生成、JSON、记忆或投机接受率。

首次wake lock超时，未启动模型，失败状态保留initial-failure.json；重试成功后完成六组。结束时SSH连接曾导致自动释放模型锁失败，重连确认无模型进程、只清理本次owner后重试释放wake lock成功，cleanup.txt与wake-unlock-retry.json保留。未回退ADB，未修改频率、设置或默认运行时。

## 入口

- 配置：`configs/npu-decode-fine.json`。
- 构建：`python3 scripts/build/decode_fine.py`，原始共享源码只读，CPU补丁和库仅存私有探针目录。
- 采集：`python3 scripts/run/decode_attribution.py --config configs/npu-decode-fine.json`。
- 分析：`python3 scripts/analyze/npu_decode_fine.py`。
- 本地原始结果：`results/20261113-npu-decode-fine/`；构建：`results/20261113-decode-attribution/build-fine/`。CPU节点元数据、每轮DSP逐算子记录和分解在summary.json；全部原始日志与失败输出保留，不纳入Git。
