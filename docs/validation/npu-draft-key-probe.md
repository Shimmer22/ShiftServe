# NPU草稿、CPU验证：关键成本检查

本轮只研究NPU稀疏草稿、CPU完整模型验证。prefill神经元重要性能否迁移到decode暂不检验，固定通道只用于性能探针。

Qwen3.5-9B Q4_0，H4096/P40、8192 context，CPU6线程、HTP主机4线程。完整prefill后原地压缩进程内FFN权重，保留50%/25%通道；CPU只测完整decode与k=1/4验证。模型文件未写入。所有设备计算通过Termux SSH持锁串行执行。

结果与边界在[实机报告](../../results/20261113-npu-draft-key-probe/report.md)，机器可读结果在[summary.json](../../results/20261113-npu-draft-key-probe/summary.json)。

当前CPU约138 ms/token；NPU保留25% FFN约219 ms/token，移除全部FFN矩阵的诊断点约199 ms/token。即使假设全部草稿接受并额外推进1 token，忽略打包、同步、状态追赶与回滚成本，k=1/4中最有利的合法宽度点仍约173 ms/token。它是成本估算，不是实际投机加速；当前实现仅靠FFN剪枝未显示收益空间。

完整NPU前后对照波动大，不能将全部速度变化归因于剪枝。CPU k=4批量与逐token的一个位置argmax不同，严格保持原输出尚未通过。recurrent尾部删除失败，完整检查点恢复通过，但需额外成本。这不是τ²任务E2E或通用硬件结论。

配置：`configs/npu-draft-key-probe.json`、`configs/cpu-verify-key-control.json`。构建：`scripts/build/sparse_feasibility.py`；采集：`scripts/run/sparse_feasibility.py --config <配置> --output <新的具名结果目录>`；分析：`scripts/analyze/npu_draft_key_probe.py`。采集默认拒绝覆盖已有结果。

先前两种额外HTP权重分配方式的FastRPC映射失败日志保留在`results/20261112-ffn-sparse-feasibility/`与`results/20261112-npu-draft-key-probe/`；最终改用私有进程已有映射，不增加映射或回退后端。
