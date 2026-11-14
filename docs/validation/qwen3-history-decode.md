# Qwen3 历史长度与 Decode 成本

本轮测试 Qwen3-8B Q4_0 在 128、256、512、1024、2048、4096、8192 个历史 token 下的逐 token decode 墙钟成本，比较 CPU 与 HTP offload。

配置入口：`configs/qwen3-history-decode.json`。原始日志、模型与库身份、每轮结果以及最终报告放在 `results/20261114-qwen3-history-decode/`。原始失败日志保留为 `driver-timeout.log` 和 `status-timeout.json`；首次驱动在8192 CPU的长prefill阶段超时，手机测试自行正常完成，恢复时取回原结果后继续最后一个条件。

历史 H 是第一个计时 decode 前实际 KV 中的 token 数，P=0。全部固定 context=8704、batch=128、F16 KV；CPU 6线程/ngl=0，HTP主机4线程/ngl=99；KV offload、flash attention、CPU repack、KV整理、预搬移和稀疏裁剪均关闭。HTP OPBATCH=1024、OPQUEUE=16。HTP 是混合算子的 offload 路径，不能将其耗时称为纯DSP计算时间。

每个条件独立进程，正常响应后在同一个上下文内重新建立完整历史；每轮恢复相同状态，先累计至少40秒decode预热，再正式3轮×16个固定token。表中ms/token是每轮总decode时间除16后取3轮中位数。初始化、历史prefill、状态恢复与功能验证不计入decode。输入为已有4096-token fixture重复至8192后取相同前缀，固定token回放用于成本测量。

验收要求：退出码0、完成标志、精确H/P、40秒预热、3轮计时与有限logits、重复logits cosine≥0.99、正常OK响应、两轮JSON记忆、检查点续跑最终状态逐字节一致。图中保留各轮数据，分别提供log₂及线性横轴，导出PNG/SVG/PDF。

```bash
python3 scripts/run/decode_attribution.py --config configs/qwen3-history-decode.json
# 如驱动中断且该实验仍持锁，仅恢复本轮：
python3 scripts/run/decode_attribution.py --config configs/qwen3-history-decode.json --resume
/Users/hui/miniforge3/envs/visiontrim-qwen25vl/bin/python scripts/analyze/qwen3_history_decode.py
```

描述性拟合比较 a+bH 与 a+b·log₂H 的RMSE和R²。拟合不证明渐近复杂度：标准全注意力decode的attention与KV读取工作量随H增长，总时间还有权重计算和运行时固定成本。本轮顺序按H递增，频率和热状态与长度混杂，不能把有限区间的曲线解释为对数复杂度。

全量验收完成：14个条件、42轮正式重复全部通过，正式decode无major page fault，手机锁已释放。128→8192 token时，CPU为129.31→277.32 ms/token，HTP offload为199.09→398.37 ms/token。CPU线性/对数拟合R²为0.946/0.927，HTP为0.902/0.959；仅描述本轮曲线，不据此判定复杂度。完整数表与双横轴图见结果目录report.md。
