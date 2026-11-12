# τ²-bench 主机轨迹与 always-prepare 手机 pilot

本轮暂停 pad，不暂停 Tool Gap。主线后端仍为 CPU/HTP；主机 CUDA 只用于并发生成工作负载，不是手机候选后端。

## 固定环境

官方 `sierra-research/tau2-bench` v0.1.0，commit `37199f36924c8896f5e048360691f8476cd89ba1`。源码复用于 shared/references 与 wins references，模型沿用 wins 已有 Qwen3.5-9B BF16，手机沿用 Qwen3.5-9B Q4_0。配置为 `configs/tau2-host-qwen35.json`。

主机使用既有 vLLM 容器和独立 Python 3.13 环境，最多4路并发。原容器启动时 NVML 不可用，确认容器只有 `sleep infinity` 后重启恢复；未改变主机或手机设置。安装包不改动 vLLM 的 Python 环境。

Agent 和用户模拟器都使用本地9B。保留官方政策、工具、领域状态、编排和评分器，只替换模型传输并记录精确输入/输出 token。该配置不等于官方参考用户模拟器；不能把结果标作参考榜单分数。自然语言裁判如被任务调用，也被记录为本地模型调用。

主机入口：`scripts/collect/tau2_host.py`；服务管理入口：`scripts/run/tau2_host_server.py`。主机的请求、响应、工具事件、任务状态与评分写入 `results/20261112-tau2-always-prepare/host-traces/`。

## 全量轨迹与状态锚点

每域预先随机选择10题，共30题。所有失败保留；输出到达上限时记录不完整响应，不能按正常完成轨迹统计。官方状态评分与轨迹完整率分开。

分别统计逻辑公共前缀和现有手机服务的物理锚点。服务在上一输入结束时保存完整 recurrent/conv/KV 状态；只有该输入整体是下一 prompt 的前缀，才可直接恢复。不裁剪 recurrent 状态来构造命中。

本轮170个工具返回边界完整匹配，165个用户返回边界全部失配。该事实来自本轮主机模板和消息渲染，正式手机闭环需另核对，不能推广为所有 τ² 实现必然失配。用户边界保留在全量统计中，不能用它们较长的等待假设可直接隐藏有效准备。

分析入口：`scripts/analyze/tau2_traces.py`。全量统计在 `analysis/`；首轮手机选样在 `pilot-six/`：每域2个不同任务，各选P的25%/75%附近一个工具边界，不按reward筛选。它只覆盖可恢复的工具边界，不是整任务手机结果，也不代表全量分布加速比。

## 成本补测

已有 H4096/CPU6线程的47个P点、每后端10次，保留为原始证据。新轨迹P明显超出旧表128上限，不能直接沿用旧线性阈值或外推旧表。

`scripts/collect/tau2_state_curves.py` 串行测量 context8192 下 H=512、1024、2048、4096、6144、7168 的快照导出、准备恢复、HTP→CPU返回迁移和CPU恢复。准备与返回迁移不对称；所有成本包含实际生产助手中的状态字节核验。另检查 H1024/4096/6144 的 prefill 和CPU6线程decode，结果在 `results/20261112-tau2-state-curves/`。

正式pilot的历史超过8K，因此使用context16384，并重新测M(H)、代表P点与decode。P采用精确查表以保留阶跃，H插值的敏感性由独立P128点检查。成本校准使用独立BFCL token流构造长度夹具，绝不使用正式回放的实测后端耗时做事后选路。若H敏感性模型验证不通过，明确禁用HTP选路，原始结果保留。

## 手机回放协议

入口 `scripts/collect/tau2_phone.py`，实现位于 `src/a2o/profiling/tau2_phone.py`。等待上一手机校准完整结束并释放锁后启动，不强行接管失败现场。通过Termux SSH持锁，CPU6线程、HTP主机4线程，无KV整理和预搬移。

先检查正常响应、HF/GGUF逐序列token一致、短输入的pending→CPU与ready→HTP迁移。正式6个完整边界，每个CPU/HTP/Route各测3次，轮换顺序，共54次条件测量。不截历史、不截输出，不自动执行生成动作。

每个Route都真实启动HTP状态恢复，按轨迹实测G等待，返回时仅在准备ready且预测更快时选HTP，否则显式选择CPU。准备工作可能与CPU路径竞争；后台结束尾部单列。纯阶段、返回后响应、含G的窗口与串行清理时间分别保留。模型初始化、历史建立、缓存和夹具恢复不混入阶段耗时。

工具本身用定时等待回放，因此这是**真实异步准备＋冻结token＋实测G时长的机制实验**，不是工具与推理在线共同运行的完整E2E，也不测手机任务质量。Full HTP 从CPU构造的匹配状态夹具开始，数值和质量限制与早期冻结回放相同。

结果在 `results/20261112-tau2-phone-pilot/`。`status.json`为真实阶段，`background.log`保留失败，`calibration/`、`validation/`、`measurements/`分开。完成后收回本次日志、停止服务并释放锁。

## 贪心继续/放弃接续

用户明确接受当前预测可靠性，新的接续实验使用 `scripts/collect/tau2_greedy.py`，结果在 `results/20261112-tau2-greedy-return/`。旧轮原始数据、二进制和原生源码快照保持不变。新轮复用独立成本测量，不使用旧轮正式响应耗时做事后选路。

仍为always-prepare。返回时比较 `预计剩余准备时间 + HTP prefill + 回迁` 与 `CPU恢复 + CPU prefill`。剩余准备时间取独立准备测量中位数减去从启动准备请求到返回已流逝的时间，下界为0；这不是读取本轮后台线程未来的完成时间。ready时剩余时间直接为0。

继续预计更快就等待；否则取消准备并走CPU。等待预算取CPU路径成本减HTP路径成本，预算耗尽仍未ready就转CPU。等待期间阻塞当前请求，全部计入返回后响应时间；回迁、CPU恢复、prefill、decode和后台清理尾部继续分别计时。该预算限制单次试等，不保证在预测失误后仍优于立即CPU。

保留 `validated=false` 和原校验误差，同时显式记录 `allow_unvalidated_prediction=true`；不再通过极大HTP成本强制CPU。非有限数值或真实后端错误仍直接失败，不伪装成策略选择。

新轮仍为相同6边界、全H/P/D、三条件各三次，轮换条件位置。正式计时前覆盖贪心继续、主动放弃、已ready和等待超时四条分支，短机制请求不混入正式结果。此修改作用于τ²冻结回放入口，历史在线推理入口的ready-only规则不因本轮实验而被隐式改写。

本轮已完成54个条件测量，18次Route均等待成功并实际走HC。返回后累计CPU1433.573s、Full HTP1832.875s、Route1139.543s；相对CPU/Full HTP为1.258×/1.608×。等待与回迁计入返回后耗时，结果范围及同批阶段oracle见[贪心实验验收](../validation/tau2-greedy-return.md)。
