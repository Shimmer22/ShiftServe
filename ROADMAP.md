# 主线：Tool Slack 下的投机状态准备与延迟绑定

| 阶段 | 具体改动 | 完成标准 | 当前状态 |
|---|---|---|---|
| A 多轮基线 | CPU/HTP 双 Context，真实历史状态复用 | 连续多轮正确，不重复 prefill 旧截图 | 首帧截图＋历史追问的HH/HC/CH/CC及全CPU对照通过；连续新截图仍有FastRPC问题；DeltaNet未覆盖 |
| B 优势区间 | CPU/HTP prefill、双向交接与布局成本测量 | 得到给定 H/P 下的实际路径查表 | 四路径结构化组96次完成、60次数值通过；尚无合格的四路径全局最优/Gap翻转证据 |
| C 两阶段调度 | Tool 执行与 State Preparation 异步重叠 | 开始时决定是否准备；返回后按真实 P 和准备进度绑定后端；保留 CPU 权威状态 | 跨轮回收、固定D成本Router、只读文本Tool完成最小验证；真实轨迹覆盖、Tool/Gap策略与性能未完成，见 TOOL_SESSION.md |
| D KV-aware 优化 | 压缩、Tile-aware 选择、部分准备 | 相比全量异步准备获得额外收益 | 未实现 |

手机多模态回归固定 GPU ViT；BFCL 文本轨迹不加载视觉编码器。已有成本校准包含 CC（CPU/CPU）、HH（HTP/HTP）、HC（HTP/CPU）、CH（CPU/HTP）。四路径是代价模型的支撑组件，不将四路径迟滞调度本身列为核心创新。此前固定 CPU Decode 的测量保留为历史证据。

## 开发状态与下一步顺序（2026-11-04）

按本次提供的模块表整理如下。状态来自仓库已有记录；BFCL 行表示已安排采集，不表示本次查询过远端实时状态。

| 模块 | 状态 | 当前判断 |
|---|---|---|
| 多轮 KV Snapshot/Restore | 已通过既定用例 | 复用已有实现，不重做；连续新截图稳定性与 DeltaNet 仍不在通过范围 |
| CPU/HTP 四路径 | 部分通过 | 96 次文本校准中仅 60 次同时数值合格，不将完整四路径视为全部可用 |
| 异步状态准备 | 基础通过 | 22 项本地测试；新常驻入口六轮机制用例及六轮 CPU 对照通过，保留旧四轮验证 |
| 版本校验、取消、CPU 策略回退 | 基础通过 | 新常驻入口通过跨轮持有旧任务、结果发布和下一轮 CPU 推进；后端 API 仍不可抢占，关闭允许等待 |
| BFCL Tool Trace | 已安排远端采集 | 仓库文档尚无正式数据结果；不能作为真实手机 Tool 等待的替代证据 |
| 真实 Tool 驱动的调度 | 只读文本基础接入通过 | 实际 ADB 系统查询、固定D成本分支和普通模式通过；Browser/GUI/外网API与完整收益未验证 |
| KV 压缩和独立 Tiling | 未实现 | 暂缓，待基础 Runtime 与调度成立 |

旧夹具仍有两个明确限制：`async_preparation.h::bind()` 只按就绪/版本/取消状态绑定，未比较成本；`session_runtime.h` 轮末仍 `join()`。其旧验证只说明当前轮 CPU 计算可不提前等待。新 `tool_session_runtime.h` 使用会话级 lane 和独立成本 Router，已用屏障证明结果发布及下一轮 CPU 推进可先于旧任务退出；保留旧夹具作为回归，不把新结论套到旧入口。

执行顺序调整为 **C0 跨轮非阻塞资源回收 → C1 成本 Router → C2 真实 Tool 集成 → C3 可信环境中的性能验证 → D 可选优化**。四路径数值问题保留为候选准入门槛；先用已验证条件推进机制，不必重做已通过的 Snapshot/Restore，也不先增加 D/Gap 预测特征。C0/C1 与 C2 只读文本入口已通过本地及最小设备验收；代码、证据与限制见 [TOOL_SESSION.md](TOOL_SESSION.md)。以下保留完整目标与验收要求，不将基础接口等同于性能证据。

### C0 / P0：跨轮非阻塞 Runtime（基础设备验收通过）

- 将候选任务、不可变 snapshot 和目的 HTP context 的生命周期交给会话级管理器。当前轮放弃候选时只做失效标记和取消请求，将仍在执行的任务转入待回收集合，不在动作交付或下一轮 CPU 入口调用阻塞 `join()`。
- 任务退出前，仍持有其 snapshot 与目的 context；该 HTP lane 不得清理、导入新版本或执行另一轮 prefill。先保留一个有界 HTP lane：繁忙时跳过新投机准备，由显式 CPU 策略继续执行，记录原因，不无限创建线程、context 或排队候选。
- 只在确认任务已退出后回收并 `join()`；会话关闭/失败清理允许显式等待，必须与逐轮关键路径分开记录。不能仅删除显式 `join()` 后让析构函数等待，也不能用无所有权的 detached thread 代替资源管理。
- 取消不承诺抢占正在执行的 backend API。过期候选永不提交；后台异常在逐轮轮询及最终关闭时必须被收集并报错，不能在回收时丢失。设备不可用仍直接失败，不能与策略选择 CPU 混为一谈。
- 动作完成、结果实际发布、CPU 状态提交、下一轮开始、后台任务退出、资源回收分别记录事件。结果发布不得隐含等待待回收任务。

验收采用同步屏障，故意将旧任务卡在不可抢占调用的模拟边界：在释放屏障之前，证明当前结果已经发布、下一轮 CPU 已开始并能够完成；旧 HTP lane 没有被复用。释放后检查旧版本拒绝、错误传播、资源安全回收及后续新版本正常导入。另测连续放弃候选时任务数/内存有界、会话关闭等待、异常路径无 use-after-free。先做本地确定性测试，再做最小真机机制回放，不依赖毫秒阈值。

先在 `src/a2o/` 实现生命周期与 lane 所有权。如果真实 CPU 推进仍被共享后端全局锁或同步阻塞，应插桩定位后再修改 shared llama.cpp 的相应 CPU/Hexagon 路径；本地模拟通过不能替代这一设备检查。

### C1 / P1：Tool 返回后的成本 Router（基础分支验收通过，实测覆盖有限）

- Runtime 只报告候选可用性、版本、准备阶段与 lane 占用；Router 根据成本作最终选择。`HTP ready` 只是候选可用条件，即使 ready，短输入仍可能选择 CPU。
- 输入包括真实新增 P、历史 H、权威状态驻留位置、候选版本/准备阶段、已知或显式配置的 D，以及通过响应和数值检查的实测代价表。准备进度先按安全阶段报告，不能将不可抢占 import 的耗时比例冒充可观测完成百分比。
- 比较返回时点之后的剩余准备、prefill、阶段交接和 decode 成本；ready 候选的剩余准备为零，但不意味着整条 HTP 路径为零成本。已花费的准备是沉没开销，不重复加入未来路径成本；仍单列累计浪费、资源占用与对前台的干扰。
- pending 候选不得直接交给 HTP prefill；若支持等待，等待必须是 Router 的显式成本决策，不能由 Runtime 隐式强制。旧版本、取消、繁忙 lane 不能被误当作 ready。
- 缺少合格代价、未覆盖的 H/P/D 或后端数值不合格时，记录覆盖缺口并使用显式保守策略；不能填造代价或把这种选择记为已测最优。新增实测只在资源不受抢占时用于性能校准。

验收至少覆盖：ready 但 CPU 更便宜、ready 且 HTP 更便宜、pending 且 CPU 立即执行、过期/取消/繁忙不可提交、代价缺样本、后端错误不得静默回退。每个决策记录真实输入、代价表来源、候选成本、选择及原因。首版固定/已知 D 即可，暂不增加预测器。

### C2 / P1 与 C3：真实 Tool 接入与性能证据（只读文本接口已实现，性能未验证）

C0/C1 已接入只读文本 Tool 生命周期：Tool 发起时决定是否准备，返回时读取真实 P 和准备状态并选路，保持 CPU 权威状态与明确的结果发布接口。当前真机工具为只读系统查询，HTTP适配器有本地实际服务器测试。BFCL 先用于轨迹回放与分布分析；需要拓展实际 API/Browser/GUI 等待，才能支持这些场景的结论，回放不自动执行模型生成动作。

比较基线包括固定 CPU、固定 HTP（仅合格条件）、返回后准备、总是投机准备和两阶段策略。多模态返回后准备基线允许与 GPU ViT 重叠。分别记录动作交付、下一轮可启动、状态就绪及资源回收，不把内部 `action_complete` 当成外部交付。

本次会话继续只做基础开发与机制验证，不长期跑；用户并行抢占手机资源时，所有耗时均不用于性能结论。待资源条件可信后，再评估端到端延迟、尾延迟、浪费、干扰、内存和能耗；不预设一定存在 Gap 翻转或调度收益。

## Observation 与贡献范围

Obs1：阶段最优不等于路径最优。历史长度H、新增输入P、预测剩余Decode长度D、权威状态驻留位置及切换成本共同影响路径选择。

Obs2：Tool 执行提供可利用但不确定的准备窗口。返回时间未知，返回内容决定下一轮 P；投机准备可能被隐藏，也可能无效或干扰 Tool。当前只有理想重叠测量，尚未观察到 Gap 引发最优后端翻转。

Obs3：状态准备需要提前启动，而最终选路依赖返回后的实际输入。核心决策是先按 H、工具类型和预计 G 判断是否准备，再按真实 P、预计 D 与准备进度绑定后端。提前准备不强制使用加速器。

RQ1 测量真实 Tool 窗口能隐藏多少准备开销；RQ2 验证是否改变最优后端；RQ3 比较两阶段策略与固定后端、固定预取和返回后准备；RQ4 验证移动共享内存和前台竞争下的延迟、能耗、内存及正确性。

本文不主张首次提出 Prefill/Decode 异构执行、KV 预取、状态转换或利用工具等待隐藏计算。创新性取决于未知 G/P 下投机准备与延迟绑定的联合机制能否产生独立预取和调度无法获得的真实收益；目前仍是待验证的系统设计。

| 模块 | 核心机制 | 当前状态 |
|---|---|---|
| Inno1 Workload-aware Cost Modeling | 联合 H/P/D、恢复/布局/同步成本、G 与后端性能；计入未隐藏准备与切换 | 四路径组件校准中，独立 Tiling 未测 |
| Inno2 Speculative Preparation with Late Binding | Tool 开始决定是否准备；返回后按真实 P、预计 D 和准备进度绑定后端 | 固定D、合格CC/HC表与真实P绑定基础通过；Tool/Gap策略与联合收益未验证 |
| Inno3 Asynchronous State Preparation Runtime | 异步恢复、表示转换和缓冲准备；版本一致性、取消与 CPU 安全回退 | 会话级单lane跨轮回收基础通过；后端API不可抢占，表示转换、多模态与性能验证未完成 |
| 可选 Hardware-aware KV Compression | 基础系统成立后联合压缩比例、状态表示和 Slack | 暂缓 |

远端排队与 BFCL 采集见 [BFCL.md](BFCL.md)。该数据用于工作负载与受控工具时间趋势，不替代手机异构准备、真实 API/Browser/GUI 等待和能耗测量。

## 四路径首轮入口

`configs/four-path-session.json`配合`scenarios/four-path-memory.json`使用首轮真实截图和后三轮历史追问，依次验证HH、HC、CH、CC，包含HTP Decode产生状态传给下一轮的检查。每轮新截图只编码一次，不执行生成动作。

`configs/path-grid.json`使用相同CPU历史快照，以CPU或HTP为初始驻留位置，串行校准四路径。首轮聚焦小P与短动作：H1024/4096，P16/64，各3次。历史为有变化的结构化合成屏幕/工具记录，非真实视觉KV轨迹；初始两种驻留位置载入同一CPU生成的canonical快照，以隔离执行路径差异。真实HTP Decode历史交接由独立截图多轮用例验证。旧重复note输入在小P/4K下失效，保留失败组，不与结构化组混合。完整自由生成必须通过JSON、EOG和生成ID对照；D=1/2/4在本次校准中指token前向评估次数，仅取同一完整输出轨迹的实际累计Decode时间，是部分计算成本，不是完整动作响应，不外推到未测长D。完整路径时间包含EOG提交、对应状态就绪时间；不是流式首token延迟或剔除末次状态提交的动作可用时间，后续必须分开测量。每一步用于采样的logits要求cosine≥0.999、NRMSE≤0.05；提交EOG后的未使用logits单独保留，不用于响应决策验证。

`C(path,G,D) = max(0, entry_prepare-G) + prefill + phase_switch + decode_prefix(D)`

只有输入到达前的历史状态准备可以被Gap隐藏；Prefill产生的新状态交接全部计入phase_switch。完成路径保留其最终驻留位置，不强制回CPU，后续多轮调度还需计入下一轮成本。当前没有独立KV格式转换/Tiling步骤，仍记录null；不声称实现了这些优化。fixture恢复、数值/字节核验不计入校准成本，不能将该表冒充生产端到端延迟。

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_agent.py --config configs/four-path-session.json --scenario scenarios/four-path-memory.json --tag four-path-memory
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/profile_prefill.py --config configs/path-grid.json --tag four-path-grid
conda run -n visiontrim-qwen25vl python scripts/analyze_paths.py results/NAME-four-path-grid
```

B先补齐HTP Decode、双向交接和小P/短D的测量，再寻找Gap引发的路径翻转；C才实现预测D/Gap、切换裕量和真实异步准备。此前固定CPU Decode的验收不等于四路径基线通过。

## 历史：固定CPU Decode的Prefill测量

配置在 `configs/prefill-grid.json`：H=141/1024/4096/8192，P=64/256/560/1024，G=0/100/500/2000ms，CPU/HTP 两条路径，每个 H/P 重复3次。模型设备执行串行，候选顺序交替。固定 ctx10240、batch128、F16 KV、CPU mmap、不启用 CPU extra repack、HTP opbatch=1/queue=1。

先采真实阶段耗时，再离线模拟 G；不为每个 Gap 重跑模型或插入 sleep：

`T_CPU = CPU_prefill`

`T_HTP = max(0, State_Export + State_Clear + State_Import - G) + HTP_prefill + State_Return`

State_Return 分别记录 HTP export、CPU clear、CPU import。首次初始化、构造历史、为了成对测量而恢复 CPU fixture 的成本、重新导出核验的成本均单列，不混入公式。decode 在两条路径都固定 CPU，首版不将其加入 Router 决策，但单列记录以观察交接后的缓存/内存差异。

**Tiling 的测量边界：** 当前 CPU/HTP 使用同一序列状态接口和 F16 布局，没有外置 KV tiling 函数。`tiling_ms=null`，不能宣称测得 tiling=0。backend 内部 import 工作计入 import，图内激活转换/分块计入 prefill，模型权重 repack 计入初始化。独立硬件布局转换需后续增加插桩才可分离。另记录每个 P 的 batch 数和尾 batch 长度，避免将 token 分批等同于独立 KV 转换。

**工作负载边界：** 此矩阵先采用精确 H/P 的文本后缀微基准，CPU 生成完整历史 fixture，最新请求要求输出 wait JSON。P=560 在本表中也是文本 token，不能冒充已经测过560个视觉 embedding。GPU ViT 作为公共阶段排除、视觉权重不驻留，符合 A 中编码后释放的生命周期；多模态真实轨迹和前台资源争用留到后续校正。A 的实际多轮截图对照继续作为独立正确性依据。

## 入口

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/profile_prefill.py
conda run -n visiontrim-qwen25vl python scripts/analyze_prefill.py results/NAME-prefill-grid
```

每个 H 使用独立进程，进程内持有两上下文并复用一次构造的历史，串行扫描 P/重复次数。结果保留原始 JSONL、CPU/HTP 输出、完整模板 fixture、耗时、数值指标、进程资源、源码与二进制 hash。分析导出 `analysis.json` 和 `router-table.csv`；错误输出及数值未通过的 trial 不进入路径选择，缺样本不填造数据。

### 第一轮实测

结果目录：`results/20261103-201017-725874-prefill-grid`。48次成对测试中，H=141/1024/4096的36次通过响应与数值检查；对应12个 H/P 点在全部4种 Gap 下均选 HTP。G=0时含交接的路径成本比值 CPU/HTP 为1.56–2.31；没有观察到 Gap 改变最优后端。有效点的准备成本中位数为4.14–101.57ms，返回成本中位数为8.34–288.51ms，不能把整个2秒 Gap 当作实际节省。

H=8192的12次测试中两路径均输出为空、生成ID一致；只有3次数值阈值通过。原始记录保留，但全部排除选路，图上标为“未通过”。补测`results/20261103-205410-506448-long-context-restore-check`：原始CPU上下文在任何恢复前就输出结束符151645；恢复后CPU末尾logits逐值完全一致，NRMSE=0。因此本次空响应不是Snapshot/Restore引入，问题已存在于上游历史构造/模型计算。重复 `note` 填充不是有效8K响应基准；下一步换成真实多轮历史复查，不应据此开展8K调度优化。该诊断不排除另有HTP长上下文数值差异。

本结果仅说明固定参数、双上下文驻留、CPU未启用extra repack条件下的路径收益，不代表CPU最优实现或GUI端到端收益。H按升序分进程扫描，未控制长期热状态；同一 H/P 内交替候选顺序。现有数据支持保留简单查表，尚不支持以复杂 Scheduler 或 KV压缩作为已证实收益。

## C / D 的边界

C 的后台准备只操作带版本号的 CPU snapshot 副本，不修改仍可继续 decode 的 CPU 权威状态。新观察提前返回时，根据就绪状态和成本选择 CPU；未准备完成的 HTP 状态不提交。真实 Tool 长度和前台减速必须实测，B 的理想重叠公式不替代这项验证。Tool Gap的比较基线必须允许观察到达后在GPU ViT期间并行准备状态；不能用“等ViT完成才开始准备”的串行基线夸大Tool Gap的独有收益。

GPU ViT 的公共时间虽然从 B 的路径公式中排除，但真实流水线中也可能提供历史状态准备窗口。C 需要记录从 Tool 开始、观察到达、GPU embedding 就绪到 prefill 开始的时间线，测量准备与 GPU 的资源争用；不能把 Tool Gap 与 ViT 时间相加后直接当成无争用的可用窗口。

D 在 C 通过后比较：无整理、全量异步准备、压缩后全量准备、部分增量准备。压缩不能仅优化搬移字节而破坏历史记忆；递归/DeltaNet 状态在对应模型单独验证，不按普通 KV token 删除处理。


## 当前四路径结果

详见`results/20261103-213658-342299-four-path-structured-grid/README.md`和`results/four-path-memory-comparison.json`。真实截图首帧＋后三轮追问的四路径与全CPU输出、token、prefix完全一致；不等于连续新截图故障已解决。96次文本校准响应全通过、60次数值通过。有效CC/HC对照在H4K、G0、初始CPU时，P16为1.43/2.70秒，P64为3.93/1.92秒；支持保留小P不迁移的候选，未证明Tool Gap使最优路径翻转。HH数值仍待校准，当前不能宣布四路径全部可用。固定D在线成本分支与只读Tool已通过基础验证，见TOOL_SESSION.md；预测D/Gap、多模态Tool和收益验证仍未完成。
