# Self-Calibrating Runtime（无手机开发）

关键机制是随正常请求自预热：第 1 轮开始记录实际执行的 H/P/D 和阶段成本，第 2～N 轮继续积累，数据充分的成本桶开始提供预测，后续持续更新。不额外运行模型，不等待整张成本表收齐。

本阶段为独立 host 模块和合成演示，未接入手机 native 常驻会话，不改写原有精确成本表。所有演示耗时来自虚拟时钟，不代表设备性能。

## 三个独立模块

| 模块 | 入口 | 工作方式 |
|---|---|---|
| 自预热 | `a2o.warmup.RequestWarmup` | 根据正常样本数给出 cold / collecting / ready 状态；桶不足 3 个有效样本时要求密集采样，无推理循环、无额外成本表 |
| Profiling | `a2o.profiling.SamplingProfiler` | 冷桶前 3 次正常执行连续采集；成熟后每 8 次采一次；请求边界批量处理，缓冲区有界 |
| 在线成本更新 | `a2o.scheduling.online_cost.OnlineCostModel` | 对阶段成本率做 EMA；默认 alpha=0.2、3 样本可用、300 秒过期、最多 256 桶 |

模块可分别关闭。没有新增响应解析、状态比较、张量抓取、设备探针或逐 token 日志。执行失败与可选初始化推理不进入正常样本统计。异常向上传播，不重试或更换设备。

## 自预热与抽样如何配合

1. 正常请求执行当前既定路径，未覆盖的桶每次记录阶段耗时。
2. 请求结束后 `flush()` 批量更新；同一桶的第 3 个有效样本到达后，开始提供该桶预测。
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

每次只更新实际执行过的路径。没有某个候选路径的数据时，不生成该路径预测，也不额外执行候选推理。冷启动选路沿用调用方已有策略；只执行单一路径不能自然学出其他路径的成本。分桶解决邻近形状复用，不解决未执行路径的覆盖。

## 接入方式

```python
from a2o.profiling import CostKey, Sample
from a2o.runtime.self_calibrating import SelfCalibratingRuntime

runtime = SelfCalibratingRuntime()
key = CostKey('model-runtime-config-device-id', 'hybrid', 'HC', 256, 32, 4, 'prefill')

# 正常请求中的已有同步阶段，不额外执行推理：
# with runtime.span(key):
#     existing_synchronous_prefill()

# 或直接复用已有阶段计时，无须再测一次：
runtime.profiler.record(Sample(key, duration_ms=20.0))
runtime.flush()
state = runtime.state(key)  # collecting；充分后为 ready
estimate = runtime.costs.prediction(key)  # 当前不足 3 样本，返回 None
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

配置在 `configs/self-calibrating-local.json`，轨迹在 `scenarios/self-calibrating-local.json`。默认演示不同形状的正常请求：不执行初始化推理，第 3 轮形成首个可用路径预测，成熟后降频采样，成本上升后继续更新。合成轨迹显式假设桶内单位成本相同，仅用于演示机制。

`results/<时间>-self-calibrating-local/` 保留配置、原始形状/阶段耗时、采样更新数、预热状态、预测来源和成本快照；失败保留 failure.json。输出标记 `execution=synthetic`、`timing_valid_for_performance=false`。KV 整理、预搬移默认关闭。
