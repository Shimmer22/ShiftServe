# Tool Slack 感知的端侧异构推理

围绕端侧手机 CPU/HTP 异构协作，包含两条互补机制（总览与当前边界见[双机制协同架构](docs/design/joint-architecture.md)）：
- **M1（Gap 准备 + 返回后选路）**：在 Tool 等待期间异步准备 HTP 状态，返回后依据实际追加长度 `P` 与剩余准备时间在 `CC`（CPU）与 `HC`（HTP prefill → CPU decode）间动态选路。
- **M2（NPU Prefill 先验指导 KV 裁剪）**：当长历史/大 `P` 场景收敛到 `HC` 时，复用 Prefill 注意力先验（`PrefillPrior`）与 `compact_anchor` 压缩活跃 KV，降低回迁与 CPU Decode 开销（当前 1.7B 单驻留已获端到端验证，4B `M1+M2` 联合仍在修复与复测中）。

默认文本基线模型配置见 `configs/model-default.json`（Qwen3-8B Q4_0），另有 Qwen3-1.7B / Qwen3-4B / Qwen3.5-9B 专项实验配置；HTP 条件明确称为 **HTP offload**，不能由阶段名称推断全部算子在 DSP 上执行。Qwen2.5-VL-3B 的历史配置、日志和结论原样保留。

主设备为小米 SM8650，手机控制统一使用 `configs/termux-ssh.json`。历史Qwen2.5-VL的[平台迁移验证](docs/validation/platform-migration.md)保留其图像与文本限制；新默认模型使用独立准入与成本表，不继承旧模型的通过状态。

[手机单闭环](docs/validation/single-loop.md)已通过：两轮同一进程，模型真实点击测试页，基于新截图确认完成并回忆任务码；当前为 CPU 执行路径的功能基线，尚未接入投机调度。

当前主线已完成真实 HTTP Gap 下的 CC/HC 选路重放、选路 Oracle 与联合 Oracle，以及历史 KV 的 HMX tiles 提前准备对照。[任务结果与后续路线](docs/validation/oracle-summary.md)汇总候选定义、实验结论和复现入口：H/P 与 kernel 路径显著影响最优后端；准备可以与 Tool 等待重叠，但目前尚未证明稳定的 Gap 选路翻转或 tiles 的独立端到端增益。KV 压缩与 QNN ViT 暂缓。

| 目录 | 内容 |
|---|---|
| [src/a2o/](src/a2o/) | Python 按运行时、调度、工具、验收、工作负载、视觉分包；原生机制放在 native/ |
| [scripts/](scripts/) | 二级入口：run、validate、profile、collect、analyze、plot、vision |
| [docs/](docs/README.md) | 二级文档：design、guides、validation、reference |
| [configs/](configs/) / [scenarios/](scenarios/) | 显式后端配置与观察/响应场景 |
| [results/](results/) / [archive/](archive/) | 具名实验结果与历史记录；保留失败输出和原始日志 |

常用入口（均从仓库根目录执行）：

```bash
# Qwen3-8B Q4_0：模型安装校验→四路径准入→关键成本表
python3 scripts/run/model_migration_campaign.py

# 本地测试，不使用手机
conda run --no-capture-output -n visiontrim-qwen25vl python -m unittest discover -s tests

# Self-Calibrating Runtime 无手机合成演示
python3 scripts/run/self_calibrating.py

# 实际只读 Tool 的常驻会话
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run/tool_session.py --config configs/tool-session-smoke.json --scenario scenarios/tool-session-real.json

# 固定 H/P/完整输出，真实 HTTP Gap Oracle（设备实验串行）
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/profile/oracle.py --device 10.29.131.50:42369

# 串行最小机制与 CPU 对照
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate/tool_session.py
```

开发路线见 [roadmap](docs/design/roadmap.md)，目录与迁移规则见 [layout](docs/design/layout.md)，环境和旧回放入口见 [运行指南](docs/guides/run.md)。机制证据与边界见 [Tool 会话](docs/guides/tool-session.md) 和 [验证记录](docs/validation/results.md)。第三方代码许可见 [第三方声明](docs/reference/third-party-notices.md)。
