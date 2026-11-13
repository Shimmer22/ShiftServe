# NPU decode组合优化候选

2026-11-13，先优化完整模型运行时，暂不开展稀疏草稿。组合项为CPU融合AR DeltaNet和HTP批量提交1024/16；不是所有算子都放NPU。模型仍为手机已有Qwen3.5-9B Q4_0，F16状态、offload_kqv=false，CPU6线程、HTP主机4线程，无KV整理和预搬移。

## 已完成

- 实现可重复构建的私有运行时补丁：`src/a2o/native/backend/cpu-fused-gdn.patch`。只显式允许CPU承接融合AR GDN，CH prefill策略不改；默认不启用。
- 完成H4096/P40、16个decode token、每条件3次的完整模型组合对照。
- 因原配置出现同一重复内突变，追加原配置→组合配置→原配置的交错补测，每条件至少40秒纯decode预热。只读观察CPU频率/cgroup，不修改频率或手机设置。
- 正常OK、两轮有效JSON及同一上下文记忆均通过；在第8个token保存/恢复检查点后继续生成，16-token最终状态逐字节一致。

## 结果与边界

充分预热后的组合配置约210–212 ms/token；两侧原配置分别约252–259和260–311 ms/token。按两侧组中位数，组合减少约17%–32%；控制组有时段波动，不能宣称固定的总体收益。当前CPU对照约136 ms/token，组合仍慢于CPU，尚未打开稀疏草稿的净收益结论。

跨版本16个位置的greedy预测一致15/16，有数值漂移。虽然JSON、记忆和各自内部检查点继续生成通过，仍不宣称严格无损，不替换默认库、不更新主线路由成本表。仅保留显式候选。

完整报告：[组合优化与验收](../../results/20261113-npu-decode-combined/report.md)；机器结果：[summary.json](../../results/20261113-npu-decode-combined/summary.json)。预热补测原始结果在`results/20261113-npu-decode-combined-warm/`。

## 复测入口

构建复用`ssh wins`缓存镜像，不写shared：

```sh
python3 scripts/build/decode_attribution.py --followup --tag build-combined-warm
```

配置：`configs/npu-decode-combined.json`与`configs/npu-decode-combined-warm.json`；采集：`scripts/run/decode_attribution.py --config <配置> --output <新的具名结果目录>`；分析：`scripts/analyze/npu_decode_combined.py`。

私有开关为`A2O_DIAG_CPU_FUSED_GDN=1`、`GGML_HEXAGON_OPBATCH=1024`和`GGML_HEXAGON_OPQUEUE=16`，图复用开启。必须同时使用匹配的私有libllama.so；普通安装库不识别融合开关。采集器分别加载私有/原库并保存实际maps、库身份、原始响应和日志。正式时延不加载归因计时库，不把profiling开销算成优化收益。
