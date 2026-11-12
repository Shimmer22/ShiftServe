# Qwen3.5 手机三条件端到端比较

原五题全部保留，新增五题在运行前固定，见 `configs/toolsandbox-qwen35-ten.json`。正式比较要求 CPU、HTP offload、route + always state prepare 使用同一私有服务、4096 context、batch128、4线程、F16状态、thinking关闭；KV整理和预搬移关闭。HTP显式 `offload_kqv=false`，因原16K及4K HTP KV分配均发生FastRPC映射失败。该配置含CPU算子，不应称为全算子纯NPU。

服务源码在 `src/a2o/native/session/phone_service.cpp`，构建入口 `scripts/build/qwen35_preflight.py --service`；构建在wins缓存SDK/NDK镜像中进行，shared源码只读。启停入口 `scripts/run/route_phone.py`，通过Termux SSH持手机模型锁。

当前适配保存每次prompt结束时的完整sequence状态（包含KV、recurrent与conv）。下一轮prompt以该锚点为严格token前缀时，恢复锚点并重评估其余token；前缀不一致则明确从头计算。CPU和HTP对照采用相同规则。锚点不等同工具发出时完整H；同时记录H、H_anchor、P_append与P_evaluated，不能混用。

ToolSandbox模型发出工具调用后、官方环境执行前，调用 `/a2o/tool-start`；route在单lane中异步恢复真实HTP完整状态并做字节核验，不预读工具结果。返回后只在候选ready且版本、前缀匹配、独立成本表有覆盖时比较CC/HC，否则显式选CPU并记录原因。后台失败向前台传播。

独立校准入口 `scripts/collect/route_calibration.py` 从CPU采集轨迹提取实际锚点H/P，每形状CC/HC各三次，验证完整状态导入/导出一致、prefill logits cosine≥0.99、D=4 greedy token一致。只导出精确H/P行，不外推；D=4仅计算成本预测范围，非完整回答时长预测。校准不计入正式E2E。

计时包括首请求模板/计量准备到最终HTTP响应，覆盖真实工具、网络、host处理、快照及迁移。模型/上下文初始化、观察处理、状态恢复、prefill、handoff、decode、snapshot分别记录。官方评分、轨迹完整性、JSON、记忆和状态迁移分别判断，失败保留。

本入口是单用户轮ToolSandbox bootstrap，不是完整多用户模拟成绩。正式比较之外的预检、调试和独立校准各存具名目录，不并入性能统计。

## Full 重复噪声测量

`scripts/collect/toolsandbox_full_background.py` 支持 `--rounds`（默认3）、`--start-round` 和 `--adopt-first-cpu`。2026-11-07 因原首轮长 SSH 连接断开而退出，改用 `--rounds 3 --adopt-first-cpu` 接管仍在 wins 运行的首轮 CPU，不重启服务、不覆盖结果。旧失败日志保留；当前状态以 `repeats/progress.json` 为准。

后续采集通过 wins 上独立 session 的 `docker exec` 启动，日志保存在远端 `<run>.detached.log`。本地后台调度每30秒用短 SSH 连接检查远端进程和73题 completion；SSH 断线/超时自动重连，已有采集不重复启动。每条件保存阶段文件，完成并回收服务后才切下一条件，重复执行器有独占锁。采集进程消失但输出不完整时明确失败并保留现场，不把单题部分输出当成功，也不盲目重启。

本地调度需要 Mac 开机联网；无需聊天窗口或持续 SSH 会话。服务启停使用 `--retry-network`，SSH/SCP 连接故障与超时每30秒无限重试；启动重试核对同名模型锁及服务PID路径，停止重试不会误杀复用PID。成本表上传与结果同步同样重试网络故障。设备计算、权限和协议错误仍明确失败。Mac关机或本地进程被终止不属于断网；阶段检查点支持正常阶段间重启，尚未保证任意崩溃时刻都可自动恢复。

三轮条件顺序分别为 CPU/HTP/route、HTP/route/CPU、route/CPU/HTP；使用相同配置、seed、原服务binary和成本表，不改正在运行的采集器。首轮保留原目录，后两轮分别保存在 `results/20261107-toolsandbox-full/round-02` 与 `round-03`，远端输出也分轮命名，旧结果不覆盖。

每轮结束调用 `scripts/analyze/toolsandbox_repeats.py`，在 `repeats/noise-summary.json` 保存逐题耗时均值、样本标准差、变异系数、评分波动及完整率/满分率计数，另附逐轮结果。失败保留在attempt分母中；完整轨迹耗时单列，不能据此宣称无损加速。三轮仅给初步离散度估计。当前工具时间和UUID未固定，测得的是端到端总波动，不能称为纯硬件执行噪声。
