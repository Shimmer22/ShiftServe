# 设备内补齐：完整模型P31+D16验证

同一Qwen3.5-9B Q4_0、H4096权威CPU快照、P31真实输入、D16固定后续token。原版/guarded使用相同主机可执行文件与运行库，唯一干预是私有DSP。HTP主机4线程、batch128、F16状态、无KV整理/预搬移。

| 阶段 | 原版均值ms | 补齐版均值ms | 时间减少 |
|---|---:|---:|---:|
| 状态恢复（排除核验） | 17.531 | 18.066 | -3.05% |
| 完整模型prefill | 3358.447 | 3374.713 | -0.48% |
| HTP decode 16 token | 4966.850 | 5277.983 | -6.26% |
| prefill＋decode | 8325.297 | 8652.697 | -3.93% |
| 状态恢复＋prefill＋decode | 8342.828 | 8670.762 | -3.93% |

| 进程对照块 | prefill时间减少 | 恢复＋prefill＋decode时间减少 |
|---|---:|---:|
| a | -13.69% | -11.24% |
| b | 9.06% | 2.22% |

负数表示变慢。两块对照方向不一致，不能据此确认稳定的整体收益，也不能仅凭合并均值断言机制必然变慢。

## 数值与状态

完整prefill logits：cosine=0.998263579，相对L2差异=5.9296%，最大绝对差异=0.749379；top1相同=True。
D16之后logits：cosine=0.997008991，相对L2差异=7.9636%，最大绝对差异=1.572925；top1相同=True。
原版/补齐版greedy延续token完全相同=False；首次分叉位置（从1开始）=6。该延续从受控长度截断的历史开始，不冒充原BFCL任务成功或JSON语义验收。
每个进程的完整prefill状态恢复后，继续相同D16，最终全部logits及完整状态checksum与未恢复路径完全一致；全部40次计时最终logits也与各自验证参考一致。
原版/补齐版prefill状态checksum相同=False；D16后状态checksum相同=False。状态可恢复不等于两个版本状态逐位相同。

## 方法与限制

ABBA四个独立进程，各验证、预热一次后测10次，全部保留，不去长尾。两个进程块的时间变化分别保存在block_time_reduction；只有两个进程块，不能把20次进程内重复当作20次独立设备条件。
restore_work使用原有restore_session工作计时，完整导出/字节比较开销单列并排除；prefill与decode都在同一HTP上下文完成。模型初始化、状态checksum与导出、logits读取/比较、JSON写入、自由greedy验收均不计入性能段。
这是固定历史之后的完整模型推理段，不是包含工具调用、网络和冷启动的完整Agent端到端。不同HVX/HMX路径存在数值差异，不要求严格allclose通过，原始结果全部保留。

自由greedy延续（原版）：

```text
a": {"type": 10, "description": "First number."
```

自由greedy延续（补齐版）：

```text
a": {"type": max_value, "description": "First number.", "
```
