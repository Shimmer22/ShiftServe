# EXP-010：keep-all 与既有 KV 策略的 EOG 诊断对照

2026-11-03，Xiaomi 15，HTP0；同一份 EXP-009 历史 state、同一工具二进制、同一 next prompt（20 token）、相同 ctx2048。两组均 load 原 state → apply mask → save → 新进程 load → prefill/greedy。保留原无 chat template 的输入，以复现原问题。每组一次，keep-all 先运行。

| 条件 | 历史 KV token | next prefill ms | 首 token | prefill 进程 wall ms |
|---|---:|---:|---|---:|
| 全保留 | 171 | 421.186 | `<\|im_end\|>` | 15675.063 |
| 原策略 | 115 | 416.388 | `<\|im_end\|>` | 15080.100 |

结论：原 EOG 现象在无压缩、无删除造成的 position 空洞时仍出现。它不需要坏的选择策略即可发生，不能当作压缩损害质量的证据。尚不能据此认定 chat template 是唯一原因，也未排除序列化/恢复或其他共同路径问题。

约 4.8ms 的单轮 prefill 差值没有统计意义；历史仅 171 token，未做热身/重复或顺序随机化，不声称加速。进程 wall 包含 adb shell、后端初始化、模型加载、状态恢复、prefill、退出等；不能全归为 KV 迁移开销。CPU/NPU 往返、真实图像与正常生成质量均未测。

复现（手机资产需仍在，约 1 分钟）：

```bash
conda run -n visiontrim-qwen25vl python EXP-010-kv-selection-control/scripts/run_control.py
```

`DEVICE` 可覆盖 adb 地址。脚本仅使用独立手机临时目录 `/data/local/tmp/a2o_exp010_control`，若目录已存在则拒绝复用；结束删除该目录。未修改 EXP-009 数据、模型或系统设置，原结果保留。

结果：`results/summary.json`、两组 apply/prefill 日志、stats 与 generation；综合诊断及下一步分组实验见上级 `PIPELINE_REVIEW.md`。
