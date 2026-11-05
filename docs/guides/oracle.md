# 真实 HTTP Tool Gap 与 Oracle

**计时口径更新（2026-11-04晚间）：只读设备探针中 steady_clock 经过5.00秒，同机 CLOCK_BOOTTIME/实时时钟经过14.34秒。旧记录的 steady_clock 排除系统挂起，不能保证覆盖完整用户等待；旧数值保留为 active-clock 诊断。新主图统一使用包含挂起的 CLOCK_BOOTTIME，并保留双时钟对照，不改手机设置。探针见 results/20261104-observation-main/clock-check。**

入口：`scripts/profile/oracle.py`，配置：`configs/oracle-grid.json`。使用现有 shared 原生运行时和手机已有 Qwen2.5-VL-3B Q4_0，不加载视觉投影，不执行模型输出动作。H=4096、P=16/64，CPU/HTP 双上下文在同一进程，所有 decode 固定 CPU、自由生成到 EOG；全部条件要求同一完整输出及生成 ID。

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/profile/oracle.py \
  --device 10.29.131.50:42369

# 已有完整运行重新汇总
python scripts/analyze/oracle.py results/<具名运行目录>
```

## 时间与并发

手机通过本轮独占的 `adb reverse` 端口访问 host 上只读 HTTP Tool。服务器按 G=0/50/200/500ms 延迟后，返回预先验证 token 长度的观察正文；正文实际用于 prefill。手机主线程等待 HTTP，准备线程真实执行 HTP clear/import/完整状态字节核验。观察只在 HTTP 完整返回后 tokenize。服务器事件保存请求 ID、body hash 和独立单调时间；手机的发起、返回、准备完成、响应完成使用同一 steady clock，禁止跨设备时钟相减。

准备从已有 CPU snapshot 开始，不包括共同历史建立与快照首次导出。每个 trial 开始前恢复同一 CPU 权威历史并清空 HTP；这部分在 Tool 发起之前。提前准备是本实验的显式策略变量，KV 压缩和通用 pre_move 始终关闭。

返回后准备 `after_return`：发起时不准备，强制 HC 在返回后执行状态准备。始终提前准备 `always`：请求前启动真实工作线程。成本感知 `cost_aware`：发起时按相同 H/D 的所有已校准 P 是否有潜在 HC 收益决定准备；不提前读取当前 HTTP 正文、不使用当前 G。返回后按实际 P、完整 D 的独立校准代价及 ready 选择路径，保持现有 Router 不等 pending 的规则。因此 `after_return/router` 为 CC；强制 HC 测量给出按需准备的成本。

每个条件重放强制 CC、强制 HC、Router 三个分支，共 2×4×3×3×3=216 次测量。另有 12 次独立完整响应校准。每次响应完成时间先记录，再回收未完成的准备线程，回收耗时另列；CC 与后台准备的资源竞争自然进入响应耗时。全部条件串行执行，不与其他模型实验争用手机。

## 两类 Oracle

选路 Oracle：在同一准备策略下，以实际 P/D 的完整响应重放成本比较路径。严格候选是 CC 与返回时已 ready 的 HC；pending HC 不能当作 ready。按重复编号匹配 Router 与反事实重放，既要求 Router 返回时 ready，也要求对应 HC 重放返回时 ready。报告每次候选及所选原始 trial ID。

联合 Oracle：提前知道 G/P/D 后，在“不提前准备”和“提前准备”的完整路径重放之间选择。以三次完整端到端耗时的中位数比较，不拼接阶段最小值、不使用离线重叠公式。严格 HC 要求三次都在返回时 ready；另外单列允许等待 pending 或返回后准备的扩展 Oracle，展示其不同候选语义。

HC 数值未准入时，Oracle 只有 CC，选路差距与联合收益标记为不可辨识（`null`），不能将单路径退化解解释为 Router 已达最优。

二者都是**串行反事实重放的经验估计**：无法观察同一轮未选路径，取最小值有选择偏差；三次重复不足以证明小幅收益，差值可能为负。`summary.json` 同时保留返回后延迟、Tool 发起到完整响应、候选资格和每次原始数据。

## 验收与证据

校准检查完整 JSON、同一生成 ID（含 EOG）、prefill/final logits cosine≥0.999、NRMSE≤0.05，以及迁移后的完整 KV/位置字节一致性。测量不采集逐 token logits，以免审计改变计时；检查真实 HTTP 的 P、完整生成 ID、JSON、EOG 与状态迁移。该固定历史只验证相同状态恢复，不能替代独立历史记忆任务验收；既有 Tool 会话的 PINE-47 记忆验收仍独立保留。本入口不含图像，图像验收不适用。

初始化、历史构造、准备、观察处理、prefill、交接、decode、响应后回收分别记录。响应、完整 ID、EOG、状态或 HTTP 失败会保留日志并终止整组。数值失败保留全部三次校准并按整个 H/P 条件排除 HC，不从成功子集算代价；强制 HC 测量仍保存为诊断，不能进入 Oracle。汇总中的 passed 表示响应/状态完整通过，HC 数值准入另列。输出包括 `manifest.json`、`checked-rows.jsonl`、`http-server.jsonl`、`native-events.jsonl`、设备原始 JSONL、协议与 stderr，汇总为 `README.md`、`summary.json`、`summary.csv`。

## 首批结果

[2026-11-04 完整记录](../../results/20261104-095858-881195-http-oracle-grid/README.md)：12 条独立校准＋216 条测量完成，响应/完整 ID/EOG/迁移检查均通过，D=6；72 次提前准备均与 HTTP 时间区间真实重叠。两个 P 的 HC 均未过数值门槛，因此本批没有有效的两路径 Oracle 收益结论。服务端 G 目标 0/50/200/500ms 对应手机实际 Gap 中位数约 44.8/107.1/304.0/626.0ms，不能把无线 adb reverse 开销忽略为零。原始失败校准和全部异常耗时保留。

`manifest.json` 记录启动时源码，`summary.json` 另外记录实际重分析脚本 hash；`passed` 表示测量响应/状态完整，`HC_numerically_qualified` 与 `oracle_gain_identifiable` 分别表示候选及收益结论是否成立。独立复核见运行目录中的 `final-checks.json`。

## 交界扫描与用户认可的探索门槛

新增 `configs/oracle-boundary-4k.json`：H4096、P24/32/40/48/56、G目标0/200ms、三种策略、CC/HC及Router重放，每条件3次，共30次独立校准＋270次测量。`oracle-boundary-4k-fine.json` 与 `oracle-boundary-2k.json` 对P31/32/33在H4096/2048比较；全部固定ctx5120，避免改变H时同时改变分配容量。

本轮用户明确认可探索精度99，`numeric_acceptance.min_cosine=0.99`，`max_nrmse=null`表示NRMSE只记录。原0.999/0.05判断另留在`legacy_numerical_passed`，不改历史结果。响应、完整生成ID、EOG和迁移核验仍须通过。校准阶段的失败HC按整个P排除，保留诊断原始记录；CPU或已准入路径响应失败则终止。

夹具版本固定`neutral_padding_v1`：完整的中性`note`词重复，并以换行隔开完整`Wait`指令，手机tokenizer验证精确P。旧按字符截断的填充可能拼成`cWait`，其失败目录保留，不作为交界性能证据。

严格选路Oracle总包含实际已选轨迹；反事实重放若准备/ready状态与决策时不匹配，该条件差距标为null，不用相交可用集制造负的Oracle差距。联合Oracle仍是完整路径重放的经验最小值；允许等待的版本另报，不拼接阶段最小值模拟重叠。

```bash
python scripts/profile/oracle.py --config configs/oracle-boundary-4k.json --device 10.29.131.50:42369
python scripts/profile/oracle.py --config configs/oracle-boundary-4k-fine.json --device 10.29.131.50:42369
python scripts/profile/oracle.py --config configs/oracle-boundary-2k.json --device 10.29.131.50:42369
# 所有模型条件必须串行；比较脚本独立复核原始HTTP、状态与发布事件
python scripts/analyze/oracle_boundary.py results/<粗扫> results/<4K细扫> results/<2K细扫>
```

Kernel诊断配置`oracle-kernel-default.json`/`oracle-kernel-hvx.json`仅做H2048/P31,32,33独立校准并启用profile；后者显式禁用HMX。输出标记`diagnostic_only`，不进入正式时延汇总。直接路径日志的可行性另见[HTP门槛调研](../reference/htp-kernel-threshold.md)。

后续 [KV tiles 预整理原型](kv-prepack.md) 在历史状态提前迁移之外，将历史 K/V 的 HMX tile 整理放入真实 HTTP Gap，固定 HC 做开/关与准备时机的对照，分别验证缓存实际复用、完整 logits 和最终状态一致。它与原先只搬运普通 KV 的 Oracle 实验分开记录。
