# 异步准备最小机制验证

本轮只验证机制。用户会并行抢占手机资源，所有耗时保留为原始诊断数据，`timing_valid_for_performance=false`，不得用于延迟、吞吐或收益结论。不运行矩阵、长期压力测试或真实设备动作。

## 实现范围

- `src/a2o/async_preparation.h`：独立工作线程、只在安全点取消、非阻塞就绪检查、快照版本校验、错误传播和带单调时间戳的事件记录。
- `src/a2o/session_runtime.h`：复制已提交的完整 CPU snapshot（含 prefix、位置元数据），后台只操作独立 HTP context；导入后重新导出并比较完整状态字节。CPU 权威状态另行核验。已就绪才允许 HTP prefill，decode 始终 CPU。
- CPU 路径不在 prefill/decode 前 join 未完成候选；响应及 CPU 新状态提交后才 drain，下一轮必须等候选安全退出才能复用 HTP。此实现尚不能跨下一轮持续遗留后台工作。
- 后端 API 调用本身不可抢占，取消只在同步/清理/导入/核验之间检查。后台异常必须报错，不按 CPU 策略成功掩盖。

屏障固定四轮：CPU 建立历史 → 准备完成后 HTP prefill/CPU decode → 提前返回选择 CPU、释放后台导入并在版本推进后拒绝旧候选 → 导入前取消并选择 CPU。它是合成 Tool 生命周期夹具，没有真实工具适配器，没有模拟 sleep，也不代表自然 Gap 分布。

## 运行与验收

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python -m unittest discover -s tests
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_async.py --device 10.29.11.213:35675
```

设备入口只串行执行两次四轮短文本回放：同双上下文驻留的全 CPU 对照，以及上述异步机制条件。使用手机已有 Qwen2.5-VL-3B Q4_0，CPU/HTP 显式配置，整理和传统预搬移关闭。无图像，不加载 mmproj/ViT；本次不验证图像理解或连续截图稳定性。

校验包括 JSON/action、PINE-47 历史记忆、step 连续性、EOG 提交、prefix/迁移字节、版本拒绝、事件顺序，以及对照条件逐轮完整生成 token、输出、已提交 prefix、token 计数和下一位置一致性。不进行逐步 logits 数值对照，不把 token 一致性称为数值门槛通过。

`results/具名目录/summary.json` 记录套件判定和两个原始运行目录；原始日志、失败输出保留，不自动重试。事件时间以候选创建为原点；Tool 开始是夹具事件，动作完整可用在 EOG 采样时记录，CPU 状态提交及 drain 单独记录。

`action_complete` 只表示进程内部动作生成完成；本轮完整结果文件在 drain 后写出，没有流式交付或外部动作消费者。因此这里只验证 CPU 计算路径不提前 join，不能宣称外部动作交付已经消除了等待。

## 2026-11-04 基础验证结果

套件：`results/20261104-081812-212379-async-mechanism-validation/summary.json`，`passed=true`。两个条件的 native 源码 hash 和二进制 hash 相同。全 CPU 对照与异步条件各四轮，JSON/action、PINE-47 记忆、step1–4、EOG、迁移检查均通过，完整生成 token、输出、提交 prefix、token 计数和下一位置逐轮相同。最终299个已评估 token，无图像输入。

| 条件 | 绑定结果 | 基础机制检查 |
|---|---|---|
| 已就绪 | HTP prefill / CPU decode | 后台导入/字节核验通过，CPU 权威状态未变 |
| 提前返回 | CPU prefill / CPU decode | 绑定时后台尚未开始；随后导入与 CPU 计算并行发起，版本推进后旧候选被拒绝 |
| 取消 | CPU prefill / CPU decode | 工作线程未进入导入，取消状态被识别 |

本地17项测试通过，包含同一 C++ 状态机的 pending、ready、过期、执行前/执行中/完成后取消，以及后台异常在 wait/bind/join 中传播；Android 交叉编译通过。手机只跑上述两个条件，没有扩展性能矩阵或长期测试。耗时不可信，全部标记不用于性能结论；未做逐步 logits 数值阈值验证。

## 尚未实现

按 H/Tool/G 决定是否启动、实际 P/D 的成本选路、真实 Tool/图像流水线、生产式跨轮取消回收、增量 KV 导入、独立 tiling、压缩与性能评估均未实现。当前只验证 readiness binding，不是完整两阶段调度器。共享 llama.cpp/Hexagon 后端没有修改。

下一步优先级与验收见 [ROADMAP.md](ROADMAP.md)：先做 C0 跨轮非阻塞资源回收，再做 C1 成本 Router，随后接真实 Tool。`bind()` 当前将有效 ready 候选直接选为 HTP，没有真实 P 或代价输入；CPU 新状态提交后的 `join()` 及候选析构仍可能阻塞下一轮。两点均为待开发项，不能用当前四轮通过代替验收。
