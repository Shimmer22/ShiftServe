# 主线：Tool Slack 下的投机状态准备与延迟绑定

| 阶段 | 具体改动 | 完成标准 | 当前状态 |
|---|---|---|---|
| A 多轮基线 | CPU/HTP 双 Context，真实历史状态复用 | 连续多轮正确，不重复 prefill 旧截图 | 首帧截图＋历史追问的HH/HC/CH/CC及全CPU对照通过；连续新截图仍有FastRPC问题；DeltaNet未覆盖 |
| B 优势区间 | CPU/HTP prefill、双向交接与布局成本测量 | 得到给定 H/P 下的实际路径查表 | 四路径结构化组96次完成、60次数值通过；尚无合格的四路径全局最优/Gap翻转证据 |
| C 两阶段调度 | Tool 执行与 State Preparation 异步重叠 | 开始时决定是否准备；返回后按真实 P 和准备进度绑定后端；保留 CPU 权威状态 | 未实现 |
| D KV-aware 优化 | 压缩、Tile-aware 选择、部分准备 | 相比全量异步准备获得额外收益 | 未实现 |

手机多模态回归固定 GPU ViT；BFCL 文本轨迹不加载视觉编码器。已有成本校准包含 CC（CPU/CPU）、HH（HTP/HTP）、HC（HTP/CPU）、CH（CPU/HTP）。四路径是代价模型的支撑组件，不将四路径迟滞调度本身列为核心创新。此前固定 CPU Decode 的测量保留为历史证据。

## Observation 与贡献范围

Obs1：阶段最优不等于路径最优。历史长度H、新增输入P、预测剩余Decode长度D、权威状态驻留位置及切换成本共同影响路径选择。

Obs2：Tool 执行提供可利用但不确定的准备窗口。返回时间未知，返回内容决定下一轮 P；投机准备可能被隐藏，也可能无效或干扰 Tool。当前只有理想重叠测量，尚未观察到 Gap 引发最优后端翻转。

Obs3：状态准备需要提前启动，而最终选路依赖返回后的实际输入。核心决策是先按 H、工具类型和预计 G 判断是否准备，再按真实 P、预计 D 与准备进度绑定后端。提前准备不强制使用加速器。

RQ1 测量真实 Tool 窗口能隐藏多少准备开销；RQ2 验证是否改变最优后端；RQ3 比较两阶段策略与固定后端、固定预取和返回后准备；RQ4 验证移动共享内存和前台竞争下的延迟、能耗、内存及正确性。

本文不主张首次提出 Prefill/Decode 异构执行、KV 预取、状态转换或利用工具等待隐藏计算。创新性取决于未知 G/P 下投机准备与延迟绑定的联合机制能否产生独立预取和调度无法获得的真实收益；目前仍是待验证的系统设计。

| 模块 | 核心机制 | 当前状态 |
|---|---|---|
| Inno1 Workload-aware Cost Modeling | 联合 H/P/D、恢复/布局/同步成本、G 与后端性能；计入未隐藏准备与切换 | 四路径组件校准中，独立 Tiling 未测 |
| Inno2 Speculative Preparation with Late Binding | Tool 开始决定是否准备；返回后按真实 P、预计 D 和准备进度绑定后端 | 未实现 |
| Inno3 Asynchronous State Preparation Runtime | 异步恢复、表示转换和缓冲准备；版本一致性、取消与 CPU 安全回退 | 未实现 |
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

详见`results/20261103-213658-342299-four-path-structured-grid/README.md`和`results/four-path-memory-comparison.json`。真实截图首帧＋后三轮追问的四路径与全CPU输出、token、prefix完全一致；不等于连续新截图故障已解决。96次文本校准响应全通过、60次数值通过。有效CC/HC对照在H4K、G0、初始CPU时，P16为1.43/2.70秒，P64为3.93/1.92秒；支持保留小P不迁移的候选，未证明Tool Gap使最优路径翻转。HH数值仍待校准，当前不能宣布四路径全部可用。在线迟滞、预测D/Gap和真实异步准备尚未实现。
