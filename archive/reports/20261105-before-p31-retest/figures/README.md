# Observation 与系统收益主图

**数据与图形验收通过：762条完整记录，26/26个H/P的HC准入通过；五图PNG/PDF、几何检查及视觉检查完成。**

- CPU/HTP性能交界：[PNG](01_backend_boundary.png) · [PDF](01_backend_boundary.pdf)
- Cost Breakdown：[PNG](02_cost_breakdown.png) · [PDF](02_cost_breakdown.pdf)
- 实际Tool Gap重叠收益：[PNG](03_actual_gap_overlap.png) · [PDF](03_actual_gap_overlap.pdf)
- End-to-End策略比较：[PNG](04_end_to_end_policies.png) · [PDF](04_end_to_end_policies.pdf)
- Oracle Gap：[PNG](05_oracle_gap.png) · [PDF](05_oracle_gap.pdf)

原自动出图曾被两个时钟分开读取的微秒级抖动误拦；双时钟诊断容差现为0.1ms，主时钟恒等式仍为0.01ms。原始记录和计时值均保留未改，修复测试包含小抖动放行、大异常和恒等式错误继续拒绝。

来源 2 个串行批次，共 762 条响应记录。数值校准与正式测量分开；每条件 n=3 次独立重放。所有准入路径完整输出相同，D=6（含 EOG）。

统一 Qwen2.5-VL-3B Q4_0、F16 KV、ctx6144、batch128，CPU 权威历史、CPU decode；同一进程内 CPU/HTP 双上下文。无额外 tiles、预搬移或 KV 整理。各策略预算相同，初始化及共同历史建立不在 Tool 发起到响应的计时范围内，原始时间单列于 plot-data.json。

## 图与读法

1. `01_backend_boundary`：P16–1024 全扫描与 P31/32/33 细节；实测点及中位数，CPU 为实线，HTP prefill＋CPU decode 为虚线。比较固定服务器目标200ms下实际返回后的完整响应，不把核阶跃推广成连续曲面。
2. `02_cost_breakdown`：堆叠采用阶段墙钟均值，包含阶段内系统挂起，黑点是各次完整响应；准备只计返回后阻塞部分，后台总准备时间见 comparisons.json；交接包含状态检查；Other 为同次完整时间减已列阶段的余额。初始化/共同历史不被虚构成每轮成本。双时钟拆出的响应内总挂起单列 suspension-summary.json，不能把含挂起的阶段差异全部归因于算子或机制。
3. `03_actual_gap_overlap`：横轴是每次设备 CLOCK_BOOTTIME 实测 G（包含挂起）；线连接各目标组的实测中位数，不使用服务端目标作为横坐标。三个 G 条件只能判断已有跨度，无法据此宣称精准隐藏阈值。
4. `04_end_to_end_policies`：统一 Tool 发起到完整 JSON 发布；CC 不提前准备，HC 在返回后按需准备；Post-tool 按包含准备成本的独立校准选路；Always Prepare 和 Cost-aware 均为实际在线 Router。服务器目标200ms，实际 G 各次记录，不能逐次视为完全配对。
5. `05_oracle_gap`：左为同准备状态、从Tool返回计时的选路反事实，状态无法匹配时 × 表示不可识别；右为全部九路完整重放、允许等待的联合经验最小值，从Tool发起计时。联合候选包含实际策略自身，也包含固定提前HC等强制路径；候选和所选trial ID均保留。串行重放实际G不同，三次中位数取最小有选择偏差，只能用于筛查空间。Oracle不拼接阶段最小值。

## 结论边界

小样本用于首轮筛查；三次重放不足以支撑几十毫秒的稳定收益。没有图像或真实外部 Tool 工作负载，只有真实 HTTP 固定文本微基准。独立历史记忆与实际图像验收沿用独立入口，本次不宣称其通过。P16–1024 是13点稀疏扫描，不是每个整数。未准入 HC 整组排除，不以成功子集作图。

初步统计：

- H=2048，HC 准入 13/13 个 P 条件。
- H=4096，HC 准入 13/13 个 P 条件。
- H2048/P24：Cost-aware 2182.2ms；CC 2125.0ms；Post-tool 2135.7ms；Always Prepare 2159.0ms。
- H2048/P31：Cost-aware 3041.0ms；CC 3039.0ms；Post-tool 2596.8ms；Always Prepare 3016.2ms。
- H2048/P32：Cost-aware 2039.4ms；CC 2481.7ms；Post-tool 1968.9ms；Always Prepare 1919.4ms。
- H4096/P24：Cost-aware 2417.3ms；CC 2400.2ms；Post-tool 2421.3ms；Always Prepare 2403.6ms。
- H4096/P31：Cost-aware 3369.0ms；CC 3370.3ms；Post-tool 2996.4ms；Always Prepare 3393.0ms。
- H4096/P32：Cost-aware 2078.9ms；CC 2835.7ms；Post-tool 2202.3ms；Always Prepare 2102.3ms。

来源：
- /Users/hui/code/exp/2026-11-a2o-gap-profiling/results/20261105-165933-954653-http-oracle-grid
- /Users/hui/code/exp/2026-11-a2o-gap-profiling/results/20261105-173045-235354-http-oracle-grid

## CPU线程补充

[1–8线程扫描与图1的P≤64最佳CPU线程曲线](../../20261105-cpu-thread-study/figures/README.md)。该补充包含独立复测与新4线程控制；本目录原5图和原始数据保持原设定。
