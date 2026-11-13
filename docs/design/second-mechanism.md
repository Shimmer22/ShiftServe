# 第二机制：NPU草稿、CPU验证探索

本分支保存第二机制的探索快照，尚未合入主线或替换默认运行时。范围限定为CPU/HTP，不包含GPU/OpenCL候选。

## 动机与设想

当前τ²冻结边界的阶段oracle全部选择HTP prefill、CPU decode。返回后选路之外，希望进一步降低decode成本：复用完整prefill得到的FFN通道重要性，构造HTP上的稀疏草稿模型，再由CPU完整模型验证候选token。

“prefill贡献低的通道在紧接的decode中贡献也低”暂作为假设，不在本轮验证。投机方向固定为NPU draft→CPU verify。跨后端验证与拒绝恢复必须维护完整模型的正确状态，不能把稀疏状态直接当作完整模型状态。

## 已测与未测

| 项目 | 当前状态 |
|---|---|
| 旧运行时固定FFN通道稀疏 | 已测。保留25%时HTP约219 ms/token，CPU完整decode约138 ms/token；不是重要性选通道 |
| HTP decode归因 | 已测。融合DeltaNet关闭、CPU/HTP碎图、大量小批次与HVX单token路径是主要线索 |
| CPU融合AR DeltaNet＋HTP批量提交 | 已测私有候选。充分预热后约210–212 ms/token，两侧原配置约252–311；控制组存在时段波动 |
| 正常响应、JSON、两轮记忆、检查点恢复 | 组合候选的有限用例通过；内部检查点重放最终状态逐字节一致 |
| 跨版本严格输出一致 | 未通过。16个位置greedy一致15/16，不宣称严格无损 |
| 优化运行时＋稀疏 | 尚未联合复测，不能把旧稀疏数字拼接到新运行时 |
| prefill重要性提取、迁移与真实接受率 | 尚未验证 |
| τ²任务E2E与任务准确率 | 尚未验证 |

这些数字来自单一H4096/P40、Qwen3.5-9B Q4_0的短探针，不是通用硬件上限。原始结果与失败日志保留在本地`results/`，不纳入Git；仓库提交代码、配置和中文结论。

## 相关工作与MTP基线

- [POP](https://arxiv.org/abs/2602.06822)直接使用prefill通道重要性指导后续动态剪枝；该先验本身不能作为独立新颖性主张。
- [SIRIUS](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ae6b2bdf3a179e3e24129e2c54bd871-Paper-Conference.pdf)研究稀疏模型生成与完整模型纠错；它不保证严格保持原输出分布。
- [Qwen3.5-9B官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)说明模型具备MTP，并提供启用方式。当前手机实验未启用MTP；这份GGUF是否包含MTP权重、现有运行时能否执行，仍需核对。原生MTP应作为后续对照。

## 入口与接续

- [固定宽度与草稿成本](../validation/npu-draft-key-probe.md)：`configs/sparse-feasibility.json`、`configs/npu-draft-key-probe.json`、`configs/cpu-verify-key-control.json`。
- [decode归因及独立干预](../validation/decode-attribution.md)：`configs/decode-attribution.json`、`configs/decode-attribution-followup.json`。
- [组合优化及状态验收](../validation/npu-decode-combined.md)：`configs/npu-decode-combined.json`、`configs/npu-decode-combined-warm.json`。

执行脚本位于`scripts/build/`、`scripts/run/`、`scripts/analyze/`；原生探针位于`src/a2o/native/profiling/`，可重建的融合策略补丁位于`src/a2o/native/backend/cpu-fused-gdn.patch`。复用shared工具链、手机模型和已有完整历史快照；新检出需要按配置提供本地夹具/缓存，不能仅凭仓库文件直接复现实机数字。

接续顺序：先完善完整模型的HTP decode效率与数值验收，再联合复测稀疏，并核对MTP支持。默认运行时和主线路由成本表保持原状。
