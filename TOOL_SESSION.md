# 跨轮回收、成本 Router 与实际 Tool 会话

本入口在一个 Android 进程中持有 CPU/HTP 上下文，复用已有 sequence snapshot 接口；没有修改 shared llama.cpp/Hexagon。只做基础机制验证，所有运行标记 `timing_valid_for_performance=false`，不运行长期实验。

## Runtime 与生命周期

`preparation_lane.h` 将至多一个候选保存在会话层。退役只请求安全点取消，snapshot/context 在工作线程退出前仍有效。`poll()` 非阻塞检查错误和线程退出，仅在退役且线程真正退出后回收；`set_value_at_thread_exit` 用于区分“工作完成”与“线程已退出”。忙碌时不创建新线程、不排队新候选，CPU 可继续推进。HTP prefill 只能 claim 当前版本且已退出工作线程的候选。

`tool_session_runtime.h` 用 `START → OBS → RUN` 驱动轮次，CPU decode 提交新状态并失效旧候选后，立即向 stdout 协议与磁盘 JSONL flush `response_published`。消费者收到的是实际响应，而非内部生成完成标记。此后才能收到下一轮 START，逐轮无阻塞 drain。正常 `CLOSE`、异常析构允许等待，保证工作线程先于 Engine/context 销毁；后台错误由 poll/close 传播，不能当作 CPU 策略成功隐藏。

旧 `session_runtime.h` 的四轮 readiness 夹具保留作历史回归，它仍有轮末 join；新的跨轮行为只由本入口提供。

## Router 与数据边界

`router_table.py` 从原始四路径组导出 CPU 权威状态、固定 CPU decode 的 CC/HC 表。每个条件要求三次均通过响应、EOG、生成 ID 及数值检查，不挑成功子集。记录源文件 hash、配置和校准二进制 hash；仅精确匹配 H/P/D，不插值、不外推。当前导出六行，H1K 的 HC 不合格而排除。

Tool 发起时，首版按 H/D 是否存在合格且潜在受益的 P 判断是否准备，不预测 G 或 D。Tool 返回后使用实际模板 suffix 的 P、候选版本/可用性和实测成本比较：ready 仍可能选择 CPU；未就绪、退役、过期或覆盖缺失时显式选择 CPU，并记录原因。准备阶段和 lane 占用另记事件。API 内部无可观测完成百分比，首版不等待 pending 候选，也不估计剩余时间。

ready 候选只比较剩余 prefill、阶段交接、固定 D 次 token 前向评估的累计成本；已完成历史准备不重复计费。D=4 是配置的计算预测范围，**不是完整响应长度或完整动作延迟预测**。当前表只覆盖既有文本微基准，不能据此证明真实 Tool 工作负载的最优后端。运行配置严格匹配 ctx5120/batch128、HTP queue/opbatch=1、F16 KV、关闭 LLM FA、CPU raw mmap。

## Tool 适配器与协议

`tool_session.py` 执行显式白名单工具：`device_display` 实际运行只读 `adb shell wm size`，`http_get` 实际发起 HTTP(S) GET（结果超过64KiB报错，不截断）。不会执行模型生成动作或任意 shell。Tool 返回原文、hash、host 发起/返回/响应接收时间线及设备事件分别保存；两端单调时钟独立，不能直接相减。

native `tool_started` 是 runtime 接收到 Tool dispatch 的准备阶段确认；实际方法开始/返回由 host 的 `tool_issued` / `tool_result_available` 记录。观察文件在真实工具返回后才创建、上传，native 随后计算实际 P，不预读下一观察。HTTP 适配器有本地实际服务器测试；真机入口默认只读系统查询，不等于已经验证 Browser/GUI 交互或外网 API 等待分布。

普通自动模式：

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_tool_session.py \
  --config configs/tool-session-smoke.json --scenario scenarios/tool-session-real.json
```

## 确定性验收

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python -m unittest discover -s tests
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_tool_session.py
```

套件只串行跑同驻留条件的 CPU 对照和机制条件，各六轮实际 Tool：建立历史 → 持有旧任务 → 下一轮 CPU 继续并发布结果 → ready 但 CPU 更便宜 → ready 且 HTP 更便宜 → 自动覆盖缺失策略。检查 JSON、历史记忆、step、EOG、迁移字节、完整 token/输出/prefix 一致性与事件顺序，不做逐步 logits 对照。

屏障在**真实 HTP 导入/字节核验之后**模拟不可抢占调用的完成边界，直到下一轮 CPU 结果已发布才释放；它不表示真实 backend API 被抢占，也不等于测过所有共享后端锁。释放后显式 `DRAIN` 只用于隔离该测试，再测 ready 分支；生产协议禁用 WAIT/RELEASE/DRAIN 和成本夹具。

两条 ready 分支使用源标记 `mechanism-fixture`、样本数0的合成成本，证明 Router 分支与真实 HTP prefill/CPU 回传可运行，不算性能数据。实测表导入和 H4K/P16 选 CPU、P64 选 HTP 的固定 D 决策另由本地 C++ 测试验证。真实 Tool 的 H/P 不在表内时应记录缺口，不能制造匹配样本。

尚需：扩展真实轨迹代价覆盖、Tool 类型/Gap 决策、真实多模态流水线、前台争用与能耗验证。外部响应虽然能够跨轮非阻塞发布，后端 API 本身仍不可抢占，关闭仍可能等待。

## 2026-11-04 最小验证结果

汇总：`results/20261104-084947-618120-tool-session-validation/summary.json`，`passed=true`。CPU 对照与机制条件各六轮，native 源码及二进制 hash 相同；JSON、PINE-47 记忆、step1–6、EOG 和状态检查通过。逐轮 Tool 原文、生成 ID、输出、提交 prefix、计数和下一位置均相同，最终521个已评估 token。

- 旧任务在真实导入核验后被屏障持有，当前轮与下一轮 CPU 均发布响应后才释放；`cross_round_nonblocking=true`。忙碌时未启动新任务。
- ready 选 CPU 与 ready 选 HTP 两条合成成本分支通过；后者真实执行 HTP prefill、完整状态回传与 CPU decode。共启动3个任务，关闭时3个均回收，单 lane 上界1。
- `results/20261104-085317-179323-tool-session-auto/` 另跑三轮普通模式，屏障和合成成本全部禁用；实际 H/P 均无实测表覆盖，明确记录 `coverage_missing`、`cost_covered=false` 并选 CPU。响应、token、prefix 与六轮 CPU 对照的前三轮一致。
- 本地22项测试、Android 构建通过。`final-checks.json` 保存普通模式对照及最终验收。核心套件使用提交 `0a59717`，随后 `573a541` 增加覆盖缺口的明确标记和分阶段诊断计时，并由普通模式验证；旧原始日志未改写。

全部运行均为机制验证，耗时无效；未做逐步 logits 数值阈值对照，没有得到新的性能、能耗或 Gap 翻转证据。
