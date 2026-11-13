# Qwen3.5-9B的HTP decode归因

2026-11-13，H4096/P40，CPU6线程、HTP主机4线程，F16状态、context8192，固定token短探针。仅Termux SSH持锁串行计算，无CPU频率、手机设置或模型文件修改。

本轮原版计时：CPU约123 ms/token，HTP约533 ms/token，关闭图复用约546 ms/token。图复用不是主要差距。

私有计时库在原同步函数边界累计时间，不改变算子、设备分配或同步行为。归因测量约548 ms/token，其中CPU split约291 ms、HTP调用约247 ms、显式跨后端拷贝约8 ms，其余约2 ms。HTP调用包含DSP计算及提交/等待，不能当作纯DSP算子时间。

关键证据：

- CPU路径启用融合Gated Delta Net；HTP层上的融合节点落CPU后，运行时设备一致性检查关闭融合，解体为小算子。原路径每token有186段CPU计算、185段HTP调用，显式张量拷贝约195.5 MiB。
- 私有对照仅保留CPU上的融合AR GDN，其他HTP计算保留、prefill策略不变，约553→402 ms/token，减少27.3%；split从371降至275，拷贝约195.5→3.9 MiB/token。
- 原配置OPBATCH=1、OPQUEUE=1，DSP诊断每token有1036次单算子提交。独立放开至1024/16后约553→420 ms/token，减少24.0%。没有将两项收益相加或当作已测组合。
- 私有v75 DSP路由探针直接记录每token248次DSP矩阵乘全部走HVX，HMX为0，原因M_lt_32。独立诊断DSP算子累计约116 ms/token，其中FFN矩阵乘约55 ms；该profile运行不与原版墙钟混算。
- 输出投影output.weight为Q6_K，加载日志确认留在CPU；当前offload_kqv=false也使标准attention保留CPU部分。HTP配置不意味着所有计算均在DSP上执行。

各干预的4个位置argmax与原路径一致，融合GDN最低cosine约0.99933；仅短轨迹数值诊断，不是完整任务、长程状态或自由生成验收。主机1线程有单次突变，不据此宣称稳定收益。共享源码、安装库和默认生产配置未修改。

详细报告：[实机归因](../../results/20261113-decode-attribution/report.md)；机器结果：[summary.json](../../results/20261113-decode-attribution/summary.json)。原版、主机分解和DSP路由的日志分别保留；受控干预在`results/20261113-decode-attribution-followup/`。

构建：`scripts/build/decode_attribution.py`，受控融合对照加`--followup`。采集：`scripts/run/decode_attribution.py --config configs/decode-attribution.json`或`configs/decode-attribution-followup.json`。分析：`scripts/analyze/decode_attribution.py`。结果禁止覆盖，复测需新具名输出目录。

对稀疏草稿的判断应以修正后的运行时基线重新评估：FFN剪枝不会移除上述递归状态拆分、搬运、CPU输出投影和小批次成本；当前测量不能代表NPU硬件的稀疏能力上限。
