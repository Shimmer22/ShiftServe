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
