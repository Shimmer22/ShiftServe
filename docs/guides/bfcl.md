# BFCL 多轮轨迹：远端排队采集

论文主线为 Slack-Aware Speculative State Preparation for Heterogeneous On-Device Agents。远端采集用于估计 H/P/D、工具输出规模及受控执行时间；手机 CPU/HTP 状态准备、资源竞争、能耗和异构收益仍需手机实测。

## 已安排作业

- SSH：`wins`，4090；结果：`/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261103-bfcl-q4-queued`。
- 配置：[configs/bfcl-remote.json](../../configs/bfcl-remote.json)。独立持久 Python 队列每 60 秒检查 SRNO 队列/子进程、GPU compute PID 和利用率，连续 3 次空闲才启动；不会停止或修改他人作业。
- 预先传模型、同步数据和低优先级双线程构建；不提前加载模型到 GPU。准备结束后再次检查占用；CUDA 不可用直接失败，不回退 CPU。
- 复用远端共享 llama.cpp 源码，独立 CUDA 构建目录；8 slots、每槽 ctx32768、batch512/ubatch128、temperature0、seed20261103。关闭 context shift，禁止静默截断；KV 整理与投机准备关闭。
- 同一份 `Qwen2.5-VL-3B-Instruct-Q4_0.gguf`，SHA-256：`63901ed05aead37348909142ed985acac6fef7d7cda3be46765468db787a2910`。从本地 shared/models 同步至远端并核对，未重新量化。手机 ADB 当前断开，手机文件实时 hash 待复核。BFCL 是文本任务，不初始化 mmproj/ViT。
- 正式推理引擎为 llama.cpp CUDA，采集 Python 复用既有镜像 `opencua-vllm:0.12.0` 内依赖；镜像名称不代表此次使用 vLLM。

BFCL 官方源码：[Gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)，固定 revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`。当前文件名为 v4；V3 引入的 multi-turn 在 V4 中沿用，不把同一套数据重复计为两个独立测试集。采集 base/miss_param/miss_func/long_context 各200个，共800个任务，保留官方数据与函数文档 hash。

## 并发与验证

先从每类按固定 seed 随机选5例，同一组20例依次跑1/4/8并发；随后完整800例用8并发。校准阶段完全失败时停止，不无条件进入全量。这里固定8槽容量，只改变请求并发，避免槽容量改变导致上下文预算变化。每个 worker 独占一个 slot；同一任务的 prefill/decode 在同一个 server 进程和上下文里完成。新任务首请求关闭 cache reuse，后续开启。

不同并发会影响工具调度、进程通信和推理排队。因此同时保留 issued→result available 的 gap、工具方法执行时间、dispatch/delivery 时间及后台 prompt/decode 计时；不能把总 gap 当作真实网络等待。函数均来自 BFCL 的内存沙箱类，文件系统/车辆/API 名称不代表宿主文件、手机动作或外部网络。只允许直接白名单函数和 literal 参数，拒绝属性访问、嵌套调用及任意 Python；每任务独立进程，超时仅终止自己的沙箱。

使用 BFCL 默认 Python function-call prompting 和 Qwen ChatML。保留原始文本、解析后调用与每轮完整序列；官方 `multi_turn_checker` 校验任务状态及返回值，质量失败、格式失败、上下文超限和超时均保留，不以退出码或成功调用数代替任务质量。

## 数据字段与边界

`events.jsonl` 用事件 ID 关联 tool、下一轮 prompt 和 decode：

| 内容 | 记录 |
|---|---|
| 工具时间线 | 发起、方法执行开始、首次返回/完成、父进程结果可用；非流式首次返回=方法完成 |
| 类型与规模 | BFCL 类/方法、完整参数和结果、UTF-8 bytes、结果 token |
| 历史 | 工具前逻辑历史 token、轮次；图像数量0 |
| 下一轮输入 | 完整 prompt、精确 token ID、公共 prefix H、新增 P、prefix 稳定性 |
| Decode | 原始生成 ID、D、EOG/调用位置、停止原因；Thinking 不适用，长度为null |
| 结果 | 工具异常、错误字段、超时、重试0、下一步调用、官方任务质量 |
| 阶段 | server 初始化原始日志；后台 prefill/decode 时长；迁移/视觉/独立Tiling为null |

H/P 是完整序列的 tokenizer 公共前缀与新增后缀，不冒充设备真实 KV 计数；同时记录后台 cache/prompt 计数。多调用批次共享一次 dispatch，结果在全批次完成后返回；每个方法执行单独计时。当前 gap 不包含序列化下一轮 prompt 的 host 开销，后者通过推理请求时间线区分。模型输出不等于正确轨迹，统计按质量通过/失败分层。

## 查看与复现

```bash
ssh wins 'cat /home/wins/ai-workspace/outputs/a2o-gap-profiling/20261103-bfcl-q4-queued/status.json'
ssh wins 'tail -30 /home/wins/ai-workspace/outputs/a2o-gap-profiling/20261103-bfcl-q4-queued/collector.log'
```

队列入口（只在远端启动一次，具名目录有排他锁）：

```bash
python3 scripts/collect/queue_bfcl_remote.py --config configs/bfcl-remote.json \
  --output /home/wins/ai-workspace/outputs/a2o-gap-profiling/NAME --detach
```

离线汇总：`python3 scripts/analyze/bfcl.py results/NAME`，生成 `tool-transitions.csv` 和按并发/类别/工具/任务质量分层的 `analysis.json`，包含分位数与 H/P/D。正式队列自动执行汇总。原始日志与失败目录永久保留，不自动覆盖、删除或补跑。

准备/运行失败写 `failure.json` 和 `status.json`；无自动重试。进程退出或远端重启后查看状态再重新排队，不把失联当作完成。
