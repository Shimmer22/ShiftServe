# 运行时优化叠加FFN稀疏

本轮已经在手机联合测量CPU融合AR DeltaNet、HTP批量提交、图复用与固定FFN通道裁剪。保留25% FFN达到166.40 ms/token，仍慢于同轮CPU完整模型的143.88–158.48 ms/token。当前实现尚未达到更快草稿所需的速度，未测投机接受率或端到端收益。

## 条件与结果

Qwen3.5-9B Q4_0，H4096/P40，context8192，batch/ubatch128，F16状态。CPU为6线程；HTP主机4线程，CPU融合AR DeltaNet、OPBATCH1024、OPQUEUE16，图复用开启，offload_kqv=false。Termux SSH持锁串行执行，每条件独立进程，完整prefill与后续decode在同一个上下文。每组至少40秒纯decode预热，再测3次×16个固定token。

| 按执行顺序 | 中位数ms/token | 三次正式测量 |
|---|---:|---|
| CPU前置 | 143.88 | 143.23 / 145.05 / 143.88 |
| 优化HTP完整FFN前置 | 208.02 | 205.32 / 208.02 / 209.06 |
| 优化HTP保留50% FFN | 176.22 | 176.22 / 175.73 / 178.07 |
| 优化HTP保留25% FFN | 166.40 | 163.93 / 166.40 / 168.04 |
| 优化HTP完整FFN后置 | 237.72 | 239.18 / 237.72 / 235.85 |
| CPU后置 | 158.48 | 153.36 / 158.48 / 158.88 |

相对两侧优化HTP完整模型，保留25%减少约20%–30%耗时；相对两侧CPU仍慢约5%–16%。CPU与完整HTP均出现前后漂移，因此报告对照范围，不认定一个精确稳定收益。正式窗口无major page fault，所有正式重复均记录16次图复用。

## 稀疏实现与验收边界

完整prefill后，将每层FFN从12288通道裁至6144或3072通道。沿用固定前缀通道，gate/up选对应行、down选对应列，保留Q4_0格式，原地使用进程私有HTP权重buffer，不改模型文件。HTP支持检查失败直接退出。替换后先用batch2重建图，再恢复完整prefill检查点，后续单token图正常复用。

保留50%/25%的权重打包分别为2626.50/2454.79 ms，单列且不计入decode；初始化、完整prefill、检查点恢复均不计入每token时间。当前打包实现读取GGUF，不能把此成本视为已解决。

所有条件在裁剪前正常返回OK、恢复历史状态逐字节核验通过；裁剪后logits有限，固定token在中间保存/恢复后继续运行，最终完整序列状态逐字节一致。本轮没有验证稀疏模型的自由生成、JSON、历史记忆或实际接受率，也没有提取prefill重要性。单组固定token计时不能推导端到端投机收益。

实验结束后无probe进程，模型锁已释放，脚本请求释放wake lock，安装的原版libllama哈希未变；未修改手机设置、频率、共享运行时或主线路由表。

## 复现入口

- 构建：`python3 scripts/build/decode_attribution.py --followup --tag build-stacked-sparse`。
- 执行：`python3 scripts/run/decode_attribution.py --config configs/npu-decode-stacked-sparse.json`。
- 分析：`python3 scripts/analyze/npu_decode_stacked_sparse.py`。
- 原始结果：`results/20261113-npu-decode-stacked-sparse/`；私有构建：`results/20261113-decode-attribution/build-stacked-sparse/`。这些本地资产不纳入Git，执行需已有夹具、历史快照与工具链。
