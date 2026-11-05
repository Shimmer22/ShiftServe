# 自校准与执行计划准备：小规模机制验收

2026-11-05，手机 `10.29.11.213:45179`，亮屏后一次进程完成 **14 个短条件**，全部通过。复用已有 Qwen2.5-VL-3B Q4_0，CPU/HTP hybrid，H256、P32/64、context1024、batch64、F16 KV。未加载视觉投影，不执行生成动作，不改手机设置，未跑正式采样矩阵。

## 已确认的机制

- 同一对 CPU/HTP 上下文内，以单次实际 CC/HC 耗时作为本轮机制参考；该参考不是可发布的性能校准表。
- 在线仅积累普通 HC 的真实 prefill、迁移工作与前 D=4 次 decode evaluation 成本。三个样本依次计为 1、2、3；前两个仍用离线 HC 值，第 3 个后来源为 online_corrected，EMA 与实际样本计算一致。
- 未执行在线更新的 CC 保持离线来源；pending 候选不被选中，P65 未覆盖形状不被外推。不要求或制造选路翻转。
- 执行计划 P32/P64 精确命中、页预触及、重复消费和后台准备条件中，实际复用图并消费全部 832 个描述符。形状失配、版本失配、绑定前丢弃和准备完成后取消均没有消费旧计划。
- 对应普通 HTP 参考下，准备条件的完整 logits、prefill 后状态、最终 CPU 状态与生成 ID 保持一致；全部 14 条完成 EOG 并返回预期 JSON。验收后没有残留模型进程，成功轮的临时设备目录已清理。

## 原始记录（仅本地）

成功轮：`results/20261105-154712-775863-execution-prepare-probe/`。包含 manifest、源码和库 hash、原始 runtime/protocol 日志、native-events、checked-cases、completion、mechanism-summary 与设备运行前后状态。这些原始结果不纳入 Git。

先前轮次 `results/20261105-154324-346159-execution-prepare-probe/` 启动时检测到 Dozing，随后终止。终止时已经留下 7 个条件的部分输出，原始日志和 precondition/failure 记录保留；不作为完整通过轮。之后由用户手动解锁，重新开始上述成功轮。

## 边界与下一步

本轮确认的是设备上的 native 在线成本模块与执行准备机制，不等于真实常驻 Tool 协议循环已完成自校准验收。取消发生在准备完成后的测试屏障；**没有验证真实 API 调用期间取消，也未触发真实后端预算失败**。这两类客户端路径已有本地假后端覆盖，实际硬件资源行为仍需后续少量定向条件验证。

固定顺序、小样本、单次参考不能证明性能收益、稳定选路翻转或测量开销。`timing_valid_for_performance=false`。执行计划准备与普通路径 EMA 分开测试，不混用成本。

复现当前 14 条件：

```bash
python3 scripts/validate/execution_preparation.py --device <当前IP:端口> \
  --runtime-dir build/execution-prepare/20261104-214303-241077 --self-calibration
```

入口只在设备 Awake 且无其他模型进程时继续；需用户手动亮屏，不自动唤醒或修改设置。

## 后续真实 Tool 三轮闭环尝试：在线修正未生效

2026-11-05，在同一设备亮屏、无其他模型进程时运行 `tool-session-self-calibrating-dev.json` 与 `tool-session-real.json`。真实 Tool 为只读 `wm size`，模型连续三轮记住 PINE-47 并正确递增 step，响应、前缀、权威状态和退出生命周期均通过。

但**自校准闭环未通过**：实际 H/P/D 为 0/113/4、141/48/4、217/48/4，全部 `coverage_missing`、`online_sampled=false`，没有 online_cost_updated，也没有 Router 使用在线修正。既有 result.json 的 passed 仅代表 Tool/响应/生命周期检查，不能代表自校准成功。

原因是 native 在采样前要求离线表已有精确 H/P/D/路径；当前表只覆盖 H1024/4096、P16/64 的部分路径。真实会话因此在开始积累前就被排除。即使放开采样，H 随每轮增长，精确键也难以重复达到成熟门槛；这正是 host 桶内复用尚未接入 native 的实际缺口。默认三样本门槛还意味着要在后续轮次才能验证修正被使用，三轮本身不是完整的反馈观察窗口。

下一步须分开“允许采集实际路径样本”和“允许按候选成本选路”：覆盖不足时仍记录已执行 CC，候选选路约束继续保留；再为 native 补跨形状成本复用与有明确适用域的离线基线查询。不能靠扩展少量精确表行覆盖不断增长的全部历史，也不以机制夹具伪造通过。

本地记录：`results/20261105-155618-168731-tool-session-auto/`，包括原始 Tool 返回、模型响应、协议事件、导出的离线表，以及独立的 closed-loop-check.json。成功的 Tool 生命周期与失败的自校准闭环分别记录；没有残留模型进程。未继续扩大实验量。
