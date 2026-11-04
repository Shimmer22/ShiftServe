# Tool Slack 感知的端侧异构推理

围绕 Tool 等待期间的投机状态准备与返回后的成本选路，研究手机 CPU/HTP 协作。默认复用 shared 工具链和手机已有 Qwen2.5-VL-3B Q4_0，视觉投影为 F16；不执行模型生成的设备动作。

跨轮资源回收、固定 D 成本 Router、实际只读文本 Tool 已通过基础验证。实测代价覆盖仍有限，尚无可信的端到端收益或 Gap 翻转结论；当前存在并行资源竞争，耗时不用于性能判断。KV 压缩、独立 Tiling 与 QNN ViT 暂缓。

| 目录 | 内容 |
|---|---|
| [src/a2o/](src/a2o/) | Python 按运行时、调度、工具、验收、工作负载、视觉分包；原生机制放在 native/ |
| [scripts/](scripts/) | 二级入口：run、validate、profile、collect、analyze、plot、vision |
| [docs/](docs/README.md) | 二级文档：design、guides、validation、reference |
| [configs/](configs/) / [scenarios/](scenarios/) | 显式后端配置与观察/响应场景 |
| [results/](results/) / [archive/](archive/) | 具名实验结果与历史记录；保留失败输出和原始日志 |

常用入口（均从仓库根目录执行）：

```bash
# 本地测试，不使用手机
conda run --no-capture-output -n visiontrim-qwen25vl python -m unittest discover -s tests

# 实际只读 Tool 的常驻会话
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run/tool_session.py --config configs/tool-session-smoke.json --scenario scenarios/tool-session-real.json

# 串行最小机制与 CPU 对照
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate/tool_session.py
```

开发路线见 [roadmap](docs/design/roadmap.md)，目录与迁移规则见 [layout](docs/design/layout.md)，环境和旧回放入口见 [运行指南](docs/guides/run.md)。机制证据与边界见 [Tool 会话](docs/guides/tool-session.md) 和 [验证记录](docs/validation/results.md)。第三方代码许可见 [第三方声明](docs/reference/third-party-notices.md)。
