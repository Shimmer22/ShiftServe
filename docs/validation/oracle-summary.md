# 真实 Tool Gap 下的选路、联合 Oracle 与 KV 预整理

**计时口径更新（2026-11-04晚间）：只读设备探针中 steady_clock 经过5.00秒，同机 CLOCK_BOOTTIME/实时时钟经过14.34秒。旧记录的 steady_clock 排除系统挂起，不能保证覆盖完整用户等待；旧数值保留为 active-clock 诊断。新主图统一使用包含挂起的 CLOCK_BOOTTIME，并保留双时钟对照，不改手机设置。探针见 results/20261104-observation-main/clock-check。**

本任务要回答：CPU/HTP 的最优执行路径由什么决定；提前利用 Tool 等待准备历史状态能增加多少收益；把历史 KV 提前整理成 HMX tiles 后，能否进一步缩短完整响应。现有证据支持 H/P 和 kernel 路径影响选路、准备可真实重叠；尚未证明稳定的 Gap 选路翻转或 tiles 的独立端到端增益。

## 实验与候选

固定手机已有 Qwen2.5-VL-3B Q4_0、F16 KV、ctx5120、batch128，关闭压缩、通用预搬移与 CPU repack。CC 为 CPU prefill＋CPU decode；HC 为 HTP prefill＋CPU decode。同一条件在一个进程内使用双上下文，Tool 正文真实参与 prefill，完整自由生成 wait JSON 至 EOG，D=6。不执行输出动作，无图像、能耗或真实外网 Tool 结论。

准备从已有 CPU 历史快照开始，包含 HTP clear/import 与完整状态字节核验；初始化、历史构建、准备、观察处理、prefill、交接、decode 和响应后回收分别记录。手机通过本轮独占 adb reverse 访问只读 HTTP Tool，用设备同一 steady clock 测量实际 Gap，不跨设备时钟相减。

| 比较对象 | 定义 |
|---|---|
| 三种准备策略 | 返回后准备、始终提前准备、按独立校准代价决定是否提前准备 |
| 在线 Router | Tool 返回后按实际 P、完整 D 的独立代价与 ready 状态选路，不等 pending HC |
| 选路 Oracle | 固定准备策略，比较状态匹配的完整路径重放；HC 必须在返回时 ready，保留每次候选的 trial ID |
| 联合 Oracle | 提前知道 G/P/D，从不提前准备与提前准备的完整重放中取三次中位数的经验最小值；严格 HC 要求三次均 ready |
| 允许等待的扩展 Oracle | 单列 pending/返回后准备的候选，不能与严格 ready 语义混用 |

HC 未准入时排除整个 H/P 条件，收益记 null；反事实状态未匹配也记 null。选路 Oracle 包含实际 Router 轨迹；联合 Oracle 属于串行反事实估计，可能因波动出现 Router 更快的观测，不能拼接各阶段最小值构造收益。

## 后端交界与 Oracle

正式粗扫及 H4096/2048 的 P31/32/33 细扫共660条记录。使用用户认可的探索门槛 cosine≥0.99，NRMSE 只记录；原严格门槛另列，不改历史失败。完整 JSON、生成 ID、EOG 与状态迁移独立检查，每条件三次。

下表取 G目标=200ms，单位 ms，比较实际返回后到完整响应的中位数，均为已准入候选：

| H | P | CC（返回后策略） | 提前准备 HC | 更快路径 |
|---:|---:|---:|---:|---|
| 4096 | 31 | 2822.2 | 3819.5 | CC |
| 4096 | 32 | 2289.4 | 1718.2 | HC |
| 2048 | 31 | 2505.3 | 2308.6 | HC |
| 2048 | 32 | 1951.3 | 1502.0 | HC |

同一个 P31，改变历史长度 H 就会改变最优路径。直接 DSP 探针中，P31 attention 的216条记录均为 HVX（M<32），P32/33各216条均为 HMX 且成功；18条完整校准通过。直接标签来自保持原 dispatch 的私有重建库，与原发布库 profiling/HMX关闭对照相互印证，诊断耗时不进入正式结果。

选路与联合 Oracle 已按上述资格分别输出。以 H4096/P32、G目标200ms 为例，成本感知 Router 返回后三次中位数1654.2ms；同准备策略的严格选路 Oracle 为1654.2ms；联合 Oracle 从 Tool 发起计时为2149.2ms，选择 always/HC。三个数的计时起点不同，不能直接相减。H4096/P32、G目标0的反事实状态未匹配，严格选路差距保留 null。

历史状态准备的中位数约为 H4096 43ms、H2048 27ms。准备可以藏入 HTTP 等待，但本批未观察到稳定的“增加 Gap 使最优路径由 CC 翻为 HC”。同策略选路差距小或为零也不能证明未知任务上的 Router 已最优。详细候选、实际 Gap 和重复样本见[交界扫描](../../results/20261104-oracle-boundary-comparison/README.md)、[4K细扫](../../results/20261104-110010-017117-http-oracle-grid/README.md)与[2K细扫](../../results/20261104-111238-772327-http-oracle-grid/README.md)。

## 提前整理历史 KV 的增量价值

固定 HC 比较“普通 KV／额外历史 tiles”×“返回后准备／提前准备”。CPU decode 始终使用普通 KV；历史 tiles 在 DSP DDR 中保存，返回后仍需 DMA 到 VTCM，新输入及填充尾部仍需处理。H4096 的72份历史 K/V tiles 占144MiB，实际 attention 分别读取 K/V 各72MiB；新增表示不替代普通 KV，也不等于全模型 KV 常驻 VTCM。

机制验收通过：预整理与普通 HC 的完整 prefill logits 数值完全相同，最终 CPU sequence 状态逐字节相同，JSON/完整 ID/EOG 一致。初始矩阵132条、独立算子诊断6条、后续三轮各30条记录均保留；首轮 DMA 异常和所有失败日志同样保留。独立 basic profile 中，普通 HC 的 F16 attention 合计约66.1ms，预整理两次为55.7/56.4ms，表明缓存被复用并减少了该阶段计算，但不是稳定端到端收益估计。

最终生命周期对照为 H4096/P32，释放校准副本，HTP prefill 后释放历史 tiles，释放开销计入完整响应。以下均为3次测量的返回后响应中位数，单位 ms；本轮带资源审计，属于诊断：

| G目标 | 仅状态提前准备 | 状态＋tiles提前准备 | tiles增量收益 | 配对获益 |
|---:|---:|---:|---:|---:|
| 0 | 1793.2 | 2174.2 | -381.1 | 0/3 |
| 1500 | 1861.8 | 1854.0 | +7.8 | 1/3 |

G目标1500ms时两侧均3/3 ready，完整响应差异仅约8ms，且配对差值中位数为-78.5ms；不能据此宣称稳定增益。该条件中 tiles 准备中位数约504ms，完整准备约570ms；成功隐藏准备并不等于把这些时间全部省到响应里。[最终分组与原始记录](../../results/20261104-155714-318982-http-oracle-grid/README.md)保留实际 Gap、阶段计时、ready 和资源计数。

早期大矩阵存在严重 CPU decode 换页长尾，不能用其损失概括方法本身。生命周期规范应共同适用于 tile 与非 tile：校准副本在测量前释放，临时 tiles 在最后使用后释放，普通 KV 保留到会话不再需要该历史为止。实现审计、失败路径与诊断回归作为[补充证据](../../results/20261104-kv-prepack-audit/README.md)，不构成新增的 Router/联合 Oracle 收益。

后续 [DMA/HVX 优化对照](../../results/20261104-kv-dma-hvx/README.md)将72次准备RPC合并，使用临时VTCM中的DMA收集、HVX整理和DMA写回，并恢复多N块时的提前加载。H4096/P32、64，G目标0/200/1500ms，每条件5次，共132条校准/测量记录；整理中位数127.4ms，旧标量轮528.2ms仅作跨轮参考。长Gap下，P32完整响应增量收益-8.6ms、P64为+16.2ms，各组配对bootstrap区间均跨零，仍未观察到稳定端到端增益。另6条独立profile中，attention由65.2ms降到52.0–52.7ms；实测8MiB VTCM，K/V各只有一个N块，因此缺失双缓冲不是本场景此前变慢的原因。

## 复现与后续路线

- [Oracle 指南](../guides/oracle.md)：真实 HTTP、粗扫/细扫、Oracle 重分析与候选定义。
- [KV tiles 指南](../guides/kv-prepack.md)：私有远端构建、显式开关、机制验证与生命周期配置。
- [DSP 路由证据](../../results/20261104-133528-960312-http-oracle-grid/README.md)：P31/32/33 实际分支与二进制来源。

下一步保持相同内存管理与预算，分别评价后端选择、提前迁移、提前表示转换的增量价值；扩大完整输出 D、真实多轮 Tool 历史及受控内存条件。当前预整理仍默认关闭；不能把 tile 的增量实验当成完整多模态调度、四路径全部准入、压缩或能耗结果。
