# Self-Calibrating Runtime（无手机开发）

整体机制是**离线 Cost Model 提供候选路径的初始成本，在线测量随正常请求持续修正**。第 1 轮开始记录实际执行的 H/P/D 和阶段成本，第 2～N 轮继续积累，数据充分的成本桶开始修正对应路径，后续持续更新。不额外运行模型，不等待整张成本表收齐。在线学习无法补出从未执行过的另一条路径的反事实成本，因此不取代离线模型。

已有独立 host 模块、合成演示与 native 常驻会话接线；不改写原有精确成本表。native 接线仅经过本地合成测试和 Android 交叉编译，未做实际模型会话验证。所有合成演示耗时不代表设备性能。

## 三个独立模块

| 模块 | 入口 | 工作方式 |
|---|---|---|
| 自预热 | `a2o.warmup.RequestWarmup` | 根据正常样本数给出 cold / collecting / ready 状态；桶不足 3 个有效样本时要求密集采样，无推理循环；ready 表示可提供在线修正，不是 Router 才能开始工作 |
| Profiling | `a2o.profiling.SamplingProfiler` | 冷桶前 3 次正常执行连续采集；成熟后每 8 次采一次；请求边界批量处理，缓冲区有界 |
| 在线成本更新 | `a2o.scheduling.online_cost.OnlineCostModel` | 对阶段成本率做 EMA；默认 alpha=0.2、3 样本可用、300 秒过期、最多 256 桶 |

模块可分别关闭。没有新增响应解析、状态比较、张量抓取、设备探针或逐 token 日志。执行失败与可选初始化推理不进入正常样本统计。异常向上传播，不重试或更换设备。

## 自预热与抽样如何配合

1. Router 使用离线 CC/HC 成本和候选就绪状态选择路径，未覆盖的在线桶每次记录正常阶段耗时。
2. 请求结束后 `flush()` 批量更新；同一桶的第 3 个有效样本到达后，开始对该路径提供在线成本修正。
3. 成熟桶降频采集；新桶仍密集采集，不受成熟桶降频影响。
4. 桶过期或被容量限制淘汰后，组合器恢复密集采样并重新积累。

因此，同一桶默认第 1、2、3 次执行就建立预测，不再等到第 33 次。单独使用 Profiler 也具有前三次连续采样能力；组合器还根据成本是否过期、是否被淘汰和实际 `min_samples` 要求强制补采。这里的“充分”仅指样本数量门槛，不声称统计置信度已经得到证明。

已有阶段计时应直接 `record()`，不重复读时钟。`span()` 只在命中时读取两次计时时钟；未命中仍有 Python 上下文管理器、分桶和计数器开销，组合器还读取成本的过期时间。没有新增 synchronize、线程或热路径 JSON 输出。实际设备开销待测。

## 变化形状如何共享数据

原始 `Sample.key` 仍保留实际 H/P/D。在线模型按阶段去除无关维度，再用小桶共享**单位工作量的成本率**：

| 阶段 | 分组使用的维度 | 成本率单位 |
|---|---|---|
| entry | H | 每个历史 token |
| prefill | H、P；忽略 D | 每个新增 token |
| switch | H+P | 每个迁移 token |
| decode | H+P、D | 每次 decode evaluation |
| observation | H、P；忽略 D | 整段耗时 |
| model_init / vision_init | 配置身份、模式、路径、阶段 | 整段耗时 |

默认 H 桶宽 128、P 桶宽 8、D 桶宽 4，采用向下取整分桶；小于桶宽的数值保留原值，零单独处理。例如 P32～39 可以共享，P31 与 P32 不合并。不同 signature、显式执行模式、路径与阶段保持隔离。调用方 signature 必须涵盖模型、量化、上下文、batch、设备、视觉输入类别/分辨率、内核执行配置等会改变成本的因素。

例如 H256/P32/D4 与 H272/P34/D5 的 prefill 可进入同一桶，分别用耗时除以 P 更新成本率；查询 P36 时乘以 36 得到近似耗时。不需要精确 H/P/D 重复三次。

`prediction()` 与 `path_prediction()` 返回 `source=exact` 或 `shape_bucket`；桶中混合过形状，或查询形状不同于采样形状时，明确标记为近似。桶内成本率近似是轻量启发式，**尚无真机误差评估**；不会自动识别全部 kernel 路径阶跃。已知执行区域不同应使用不同 signature 或收紧桶宽；不将近似值导出为原有 native 的精确实测表。设置 `shape_reuse=false` 可恢复完整 H/P/D 精确分组和耗时 EMA。

每次只更新实际执行过的路径。没有某个候选路径的在线数据时，继续使用该路径的离线成本，不额外执行候选推理。只执行单一路径不能自然学出其他路径的成本。分桶解决邻近形状复用，不解决未执行路径的覆盖。

## 离线基线与 Router 适配

`OfflineCostTable.load()` 读取现有 `a2o-router-v1` TSV，并由调用方绑定配置 signature 和执行模式。离线表保持原有精确 H/P/D 的覆盖范围；在线数据不自动扩张其候选覆盖。组合器通过 `offline=table` 启用基线模式：

- 没有在线数据、样本不足、过期或关闭在线模块：使用原离线成本。
- 在线 entry 成熟：替换该路径的 entry 估计。
- 在线 prefill、switch、decode 均成熟：用其合计修正该路径的 future。
- 离线表只有 future 合计，不拆分猜测各阶段；三个阶段未齐时继续使用完整离线 future，不将部分在线阶段再次加到原合计上。

例如离线 CC=48ms、HC=38ms，候选 ready 时先选 HC。前三轮只执行 HC，在线测得 HC future=70ms 后，比较变成 **CC 离线 48ms 对 HC 在线 70ms**，建议改选 CC。CC 的成本没有被 HC 样本改写；不声称已经获得 CC 的最新实测值。

`choose(key, availability)` 是 host 侧的 Router 适配：继续限定 CPU 权威状态、CC/HC、固定 D；候选未 ready 时仍使用 CPU 权威路径，ready 时比较两条路径的 future。entry 已在准备阶段完成，不能再次计入。返回结果包含两条路径的成本及来源。`source` 标记 offline / online_corrected，entry_source 与 future_source 进一步区分 offline / exact / shape_bucket。

host 适配与下述 native 接线均保留设备、上下文、候选版本和可用性约束。离线基线模式是接入现有 Router 的主路径；不传 offline 时保留原纯在线预测 API，供模块独立使用和测试。

## Native 接线（无手机验证）

`native/scheduling/online_cost.h` 提供有界、会话内的 EMA 修正，`CostRouter` 查询离线行时覆盖其 future；`should_prepare()` 也使用相同修正。默认关闭，开发配置为 `configs/tool-session-self-calibrating-dev.json`，host 入口传递 `self_calibrating` 与 `online_cost` 参数并记录 manifest。此配置仍依赖本地离线校准结果，不是无手机运行命令。

- 普通路径仅对离线表已有的精确 H/P/D/路径积累；不足 3 样本仍用离线值，冷桶每次采，成熟后每 8 次采；TTL 默认 300 秒、最多 256 项。
- 复用已有 prefill，抽样轮另取迁移 export/clear/import 工作耗时，排除目的状态重新导出比对。原 handoff_checked_ms 继续保留。
- decode 只在第 D 次 evaluation 完成时额外读一次时钟，不逐 token 输出。EOG 若被 evaluation 也计数；不足 D 时不更新。完整 decode_ms 仍单独记录。
- 响应发布后更新 `prefill + handoff_work + decode_prefix(D)`，下一轮生效。准备已完成的候选只比较 future，不重复收取 entry。
- 原始离线行保持不变。无在线值、数据过期或被淘汰时直接继续用离线行；未观测 CC/HC 不被另一条路径的样本改写。缺少离线覆盖时不凭在线数据扩张候选域。
- key 隔离表来源、执行条件、路径及精确 H/P/D；模型、库和设备配置在一个进程内固定，跨进程不共享缓存。native 暂不启用 host 的桶内近似复用，真实输入变化可能使在线覆盖稀疏。
- 开启执行计划准备或机制夹具时普通路径修正 inactive；控制模式不采样。计划命中/失配成本不会进入普通 EMA，联合机制的成本建模留待后续。

输出 `cpu_cost_source` / `htp_cost_source` 标明 offline 或 online_corrected；抽样轮记录 decode_prefix_ms、handoff_work_ms，发布后的 online_cost_updated 事件带更新原因和样本数。短于 D 的输出不占冷桶有效样本名额。

本轮新增单元测试共 5 项，入口 `python3 -m unittest discover -s tests -p 'test_native_self_calibrating.py'`。覆盖候选成本修正、D 前缀与抽样、TTL/容量/身份隔离、API 内取消与跨轮/关闭、预算错误传播与释放。未增加新的线上响应或状态校验链路。

## 接入方式

```python
from a2o.profiling import CostKey, Sample
from a2o.runtime.self_calibrating import SelfCalibratingRuntime
from a2o.scheduling.offline_cost import OfflineCostTable

table = OfflineCostTable.load('router.tsv', signature='model-runtime-config-device-id')
runtime = SelfCalibratingRuntime(offline=table)
key = CostKey('model-runtime-config-device-id', 'hybrid', 'HC', 256, 32, 4, 'prefill')

# 正常请求中的已有同步阶段，不额外执行推理：
# with runtime.span(key):
#     existing_synchronous_prefill()

# 或直接复用已有阶段计时，无须再测一次：
runtime.profiler.record(Sample(key, duration_ms=20.0))
runtime.flush()
state = runtime.state(key)  # collecting；充分后为 ready
estimate = runtime.costs.prediction(key)  # 当前不足 3 样本，返回 None
decision = runtime.choose(key, availability='ready')  # 在线不足时已有离线成本可选路
```

key 可由调用方在阶段外构造。异步 API 应在已有完成点提交耗时，提交耗时不能替代完成耗时。该实现面向单会话串行调用，不保证跨线程共享安全。

`path_prediction()` 仍计算 `max(0, entry-gap) + prefill + switch + decode`，所有必需阶段有成本后才返回；Gap 仅隐藏 entry，不能隐藏 switch/return 或 decode。这里是固定 D 的计算成本，不是完整响应延迟。初始化、视觉初始化、整体 observation 分开记录，不重复加到已含 prefill 的合计中。组合器提供预测与来源，由后续 native 接入将其作为辅助选路信息。

## 可选初始化工具

原 `WarmupController` 保留为**初始化收敛检测器**，默认关闭，不再是关键自预热机制，也不作为组合器的默认对象。仅在调用方显式启用时执行额外步骤，按稳定性、预算和次数停止；耗时应标记 `warmup=True`，不占用正常样本名额。

该工具不创建上下文、不清空/整理/搬移用户 KV；调用方负责初始化 scratch sequence 的隔离。不可抢占 API 可能超预算，`overrun_ms` 单列记录。

## 本地运行

```bash
python3 scripts/run/self_calibrating.py
python3 -m unittest discover -s tests -p 'test_self_calibrating.py'
```

配置在 `configs/self-calibrating-local.json`，轨迹在 `scenarios/self-calibrating-local.json`，离线夹具在 `scenarios/self-calibrating-router.tsv`。默认演示不同形状的正常请求：首轮已有离线预测，不执行初始化推理，第 3 轮开始在线修正，成熟后降频采样，成本上升后继续更新。演示固定回放 HC，并记录 CC/HC 选路建议，建议不驱动额外候选执行。合成轨迹显式假设桶内单位成本相同，仅用于演示机制。

`results/<时间>-self-calibrating-local/` 保留配置、原始形状/阶段耗时、采样更新数、预热状态、预测来源和成本快照；失败保留 failure.json。输出标记 `execution=synthetic`、`timing_valid_for_performance=false`。KV 整理、预搬移默认关闭。
