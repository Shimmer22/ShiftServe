# 移动端多轮 LLM Agent 的异构推理

Agent 每次调用外部工具后，需要等待结果，再结合历史状态继续推理。本项目研究如何根据每轮的新增输入、生成需求和历史状态，联合安排手机 CPU/HTP 执行与 KV Cache 维护，缩短工具返回到响应的延迟。

[研究思路与动机](docs/design/joint-architecture.md) · [文档索引](docs/README.md)

大模型部署可通过实验开关 `--shared-weights` 使用单份 CPU/HTP 权重。这个工程选项借鉴 [SplitLM](https://github.com/dastin359/SplitLM/tree/0a5105bb03c643e56f8b585f1c930f665ac2e715)，CPU 直接读取当前 Hexagon 的量化布局，M1/M2 继续管理选路与 KV 状态。构建入口是 `scripts/build/htp_shared_weights.py`，部署入口是 `scripts/run/route_phone.py` 的 `--runtime-overlay` 和 `--shared-weights`。当前原型支持 HTP0 上稠密 Qwen3 的 Q4_0/Q8_0 权重矩阵，默认关闭。
