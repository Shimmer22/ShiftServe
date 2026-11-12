# Qwen3.5-9B Q4_0 手机服务与 ToolSandbox

手机运行 llama.cpp 推理进程，wins（10.29.243.183）运行 ToolSandbox 官方执行环境与评分器，通过局域网 HTTP 向手机 `10.29.131.50:18780` 发送请求。控制、部署、日志读取只用 Termux SSH，不使用 ADB。

## 资产与环境

- 手机模型：`~/.local/a2o/models/Qwen3.5-9B-Q4_0.gguf`，SHA-256 为配置中的固定值。
- 模型来源：复用 wins 已有 `cache/pvst-vllm/qwen35-9b-bf16-20261016`，用现有 llama.cpp 源码转换 F16，再执行 `llama-quantize ... Q4_0 8`。官方镜像仓库检索只列出原始权重，未找到 Qwen 官方 9B GGUF 发布，故使用用户授权的自行转换路线。
- 编译：只读复用 `/home/wins/ai-workspace/tools/llama.cpp-hexagon-build/src`，私有输出在 `tools/llama.cpp-hexagon-probes/qwen35-server`。Android server 使用缓存工具链镜像 `ghcr.io/snapdragon-toolchain/arm64-android:v0.3` 和 NDK r28b；本次为 CPU 编译，不编入 OpenCL、Hexagon 或 OpenMP。
- ToolSandbox：官方仓库固定提交 `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`，位于 wins `references/ToolSandbox`。原版依赖安装在独立 Python 3.11 环境，容器 `a2o-toolsandbox-py311`，环境路径 `/workspace/tools/a2o-toolsandbox-py311-env`。
- 本次为文本工具任务，未加载 mmproj，未验证视觉输入、HTP 或 CPU/HTP 状态迁移。手机原 Qwen2.5-VL 模型保留。

## 启停

从本项目根目录执行：

```sh
python3 scripts/run/phone_server.py start
python3 scripts/run/phone_server.py status
python3 scripts/run/phone_server.py stop
```

启动前校验完整模型哈希、检查独占模型锁、请求 Termux wake lock 并执行双时钟探针。固定 CPU、单槽、4 线程、16K context、batch/ubatch 256、thinking 关闭。prefill 与 decode 在同一个手机进程上下文内执行；不做 KV 整理、预搬移或上下文截断。保留 1 个原生 recurrent context checkpoint 供服务器前缀复用，复用量单独记录。服务日志与 PID 在手机 `~/.local/a2o/qwen35-server/`。

`start` 遇到已有锁、PID 文件或哈希不匹配会失败，不自动接管其他实验。`stop` 只释放本服务拥有的锁。

## wins 发起测评

将本工程新增模块和配置同步到 wins 对应工程后，执行：

```sh
ssh wins 'docker start a2o-toolsandbox-py311'
ssh wins 'docker exec a2o-toolsandbox-py311 /workspace/tools/a2o-toolsandbox-py311-env/bin/python /workspace/projects/a2o-gap-profiling/scripts/collect/toolsandbox.py --config /workspace/projects/a2o-gap-profiling/configs/toolsandbox-qwen35-phone.json --output /workspace/outputs/a2o-gap-profiling/qwen35-phone-toolsandbox-five'
```

输出目录必须是新目录；5 个场景按配置顺序串行执行，不根据模型表现筛选任务。需要调试时使用另一个具名目录。官方场景、工具白名单、数据库状态更新、milestone/minefield 评分逻辑保持原样；模型适配器复用官方 OpenAI 消息与工具格式转换，HTTP 目标只有手机。没有调用云模型。

服务器工具调用 ID 为任意字符串，官方环境会将其拼入 Python 变量名；适配器因此将 ID 映射为 `call_步骤_序号`，每次保存映射，服务端原始响应不修改。采集进程持 wins 本地文件锁，阻止同一手机同时运行两份采集器。

这 5 个基础场景都是单个用户请求。用户端在模型首个最终回答后确定性结束，避免另接外部用户模拟模型。因此它是基础闭环验收，不是完整 ToolSandbox 多用户轮基准成绩。即使模型提前给出错误回答，官方评分仍按实际轨迹计算，不能用正常退出代替质量通过。

## H / P / G / D 定义

| 字段 | 含义 |
|---|---|
| `H` | 发出工具调用时，上一次实际手机 prompt 加实际采样 token 的长度；包含工具描述等提示模板 |
| `P_body` | 每个工具返回正文的 token 数，由手机 tokenizer 计数 |
| `G_ms` | wins 发出工具到官方执行环境返回结果可用的单调时钟耗时；无人工 sleep |
| `D` | 手机服务报告的本次 completion token 数；同时保存完整采样 ID、`D_raw_ids` 和停止原因 |
| `H_common_prefix` | 上次 prompt + 原始采样 ID 与下一次序列化 prompt 的最长公共前缀；不假设模板前缀总是稳定 |
| `P_append` | 下一次实际 prompt 长度减 `H_common_prefix`，含工具包装、模板与可能的重序列化变化 |
| `execution_ms` | 单次沙盒工具执行时间，单独于官方执行环境管理开销 |
| `issue_to_next_request_ms` | wins 发出工具到发起下一次手机请求，包含采集器 tokenize/序列化开销 |
| `result_to_next_request_ms` | 工具结果可用后到下一次请求的主机端处理时间 |

`G_ms` 是主机工具可用窗口，不等同于手机端观测到的完整网络往返空闲窗口。手机 HTTP 耗时、服务端 prefill/decode 时间、计量准备耗时分别保存。时钟值均在 wins 侧取 CLOCK_MONOTONIC；不直接相减手机和 wins 的绝对时钟。

每次请求保存原始 request/response、实际模板、输入 token ID；对 `/apply-template` 与实际推理 prompt、prompt token 数做一致性检查。遇到上下文或输出截断、缺少生成 ID、工具事件无法关联时标记轨迹失败并保留现场。

离线汇总：

```sh
python3 scripts/analyze/toolsandbox.py results/具名目录
```

输出 `tool-metrics.csv` 与 `metrics-summary.json`，逐工具关联 H/P/G/下一次 D。正式数据和调试数据分目录保留。
