# Tool Slack 感知的端侧异构推理

围绕 Tool 等待期间的投机状态准备与返回后的成本选路，研究手机 CPU/HTP 协作。基础手机服务迁移为 Qwen3.5-9B Q4_0，wins 主机运行 ToolSandbox 并向手机发送请求，入口见 [手机服务与测评](docs/guides/toolsandbox-phone.md)。现有 CPU/HTP 机制实验仍保留 Qwen2.5-VL-3B Q4_0 与 F16 视觉投影的原始配置和证据；不执行模型生成的设备动作。

主设备已切为小米 SM8650（默认地址见 `configs/device.json`）。[迁移验证](docs/validation/platform-migration.md)：运行库/模型哈希、状态迁移和三路径图像 JSON 已通过，历史文本响应仍未通过，暂不进入新平台性能比较。

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
# Qwen3.5-9B Q4_0 手机 CPU HTTP 服务（Termux SSH 控制）
python3 scripts/run/phone_server.py start
python3 scripts/run/phone_server.py status

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
