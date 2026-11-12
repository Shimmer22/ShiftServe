# BFCL 冻结边界缓存与直连重复采集

本次接手继续前次尚未完成的实验夹具优化：缓存完整 sequence 状态，减少计时区间外的历史 prefix 重建。正式重放保持原始 H、P、D、Qwen3.5-9B-Q4_0、8192 context、batch128、四线程、F16 状态及显式 CPU/HTP 后端；不缩短输出、不裁剪历史。冻结输出采用 teacher forcing，不能用它评估回答质量或真实工具异步重叠。

## 缓存机制

`src/a2o/native/session/replay_cache.h` 保留一份内存快照，并可由 `A2O_REPLAY_CACHE_DIR` 启用私有磁盘缓存。同一轨迹只有在 token 严格前缀匹配时才延续计算，否则重建。缓存包含完整状态，元数据核对模型路径、大小、修改时间、构建标识、上下文、batch 和精确 prefix tokens；磁盘读取另查字节数与校验和，再执行原有状态恢复及导出字节一致性检查。校验失败直接报错，原文件保留。

bootstrap、缓存读写、夹具恢复放在测量区间外；真实 prefill、HTP→CPU handoff 和 decode 仍计时。HTP 准备单独记录，当前重放在计时前同步完成，不等于实际 Tool Gap 隐藏。路由沿用原成本模型，包括超出其 H 校准范围的标识；本次不把该外推升级为已验证结论。

## 执行与接管

`scripts/run/route_phone.py` 增加 `--context-size 8192 --replay-cache`。默认上下文仍为 4096。`scripts/collect/bfcl_phone_cached.py` 从本机直连手机 HTTP，要求服务报告缓存协议版本 2；绑定构建、后端、上下文、endpoint、请求和成本模型，避免混用结果。按任务和 step 排序，三轮按边界轮换 CPU/HTP/route 执行顺序。响应校验失败保留 `.raw.json`；已有合法结果跳过。

`scripts/collect/bfcl_cached_handoff.py` 等待 wins 上原 24 边界复测完整结束，拉回原始结果并核对各条件 H/P/D，再核对手机模型锁、旧服务 PID、启动时间和路径后停止旧服务。随后持新模型锁启动缓存服务，先执行短 token 机制 smoke，再使用相同 24 个完整边界做三轮，共 216 次条件测量。短 smoke 只测缓存功能，不混入性能统计。三轮结束检查实际磁盘命中，汇总计时，保存日志并停止服务、释放锁；私有缓存保留在手机供后续实验复用。

默认在旧采集失败、输出不完整或进程身份不符时保留现场，不自动接管。用户明确要求提前切换时，先停止旧 collector 并写入 `stopped_for_cached_replay`，再用 `--accept-stopped-previous` 校验和保留已有完整边界，允许提前接管；本次旧轮保留 9/24，未完成边界不计成功。接管器有本地独占锁，等待和网络重试不依赖长 SSH 连接；Mac 必须保持运行联网。该脚本没有跨任意崩溃阶段的自动恢复保证，失败后先检查 `status.json`，不能直接另起相同任务。

## 本次记录

- 旧结果：`results/20261108-bfcl-text-pilot/phone-frozen-retest/`。
- 新结果及后台状态：`results/20261109-bfcl-cached-repeats/`，其中 `status.json` 是接管阶段，`measurements/status.json` 是采集进度。
- 构建：`results/20261107-toolsandbox-route-e2e/build-service-165732/`，wins 缓存 SDK 镜像编译成功，源码摘要逐文件核对一致。
- 验证：`python -m unittest discover -s tests -p 'test_*cache*.py'`，2 项通过。覆盖内存/磁盘命中、精确前缀延续、不匹配重建、构建隔离、同长度磁盘损坏拒绝，以及采集顺序轮换、断点跳过、旧服务拒绝和原始失败响应保留。模拟测试不能替代待执行的手机验证。

模型初始化、真实准备与迁移等测量口径保持可区分。新增轮次和旧单轮分开保存，后续比较要按 condition 名称匹配，不能用数组位置假定固定顺序。当前采集未固定手机温度或外部负载，三轮只提供初步重复性信息。

## wins 持久备份与下次复用

首轮缓存完整建立后，采集器在轮次间串行调用 `scripts/collect/bfcl_cache_archive.py backup`。完整 `.state`、元数据及 SHA-256 清单流式经过 Mac 传到 wins；先在 incoming 私有目录核对，再发布到 `/home/wins/ai-workspace/cache/a2o-frozen-replay/<服务二进制SHA256>/`，不在 Mac 落大文件。相同路径内容不一致时拒绝覆盖并保留 incoming。备份失败保留测量及手机缓存，暂停后续采集，不伪报归档成功。

`cache-archive.json` 记录正式首轮备份位置、字节数和状态文件数。后两轮仍使用手机本地快照，主机备份不会在测量区间内争用 I/O。

下次使用**同一个服务二进制、同一模型文件身份与相同上下文配置**，取得手机模型锁后，将归档恢复到服务期望的空缓存目录：

```sh
python scripts/collect/bfcl_cache_archive.py restore \
  --phone-directory /data/data/com.termux/files/home/.local/a2o/runs/<服务目录>/replay-cache \
  --phone-owner <当前模型锁owner> \
  --host-directory /home/wins/ai-workspace/cache/a2o-frozen-replay/<服务二进制SHA256> \
  --output results/<本次具名目录>/cache-restore.json
```

恢复前在 wins 校验 SHA-256，传输后在手机再次校验，通过才发布缓存目录。已有目标目录时拒绝覆盖。随后服务还会逐个核对精确 token/配置和状态字节；换构建或模型不会把旧快照当作命中。需要复用时保留本次二进制及其构建清单，不要无故重新编译。

## 连接重置恢复记录

2026-11-09 约 17:41，collector 在第 3 个边界（H=4444、P=32、D=44）遭遇 HTTP ConnectionReset，已完成 2 个边界结果。手机 native 服务保持存活，健康检查正常，磁盘快照已建立。约 17:46 使用 `--resume-collection` 恢复同一服务，保留原失败状态与日志，跳过已保存结果，不触碰旧轮 PID。

采集器新增网络事件落盘和重试：连接中断后通过同一服务互斥锁的健康检查等待旧请求完成，核对构建/后端/上下文不变，再重放同一冻结请求。HTTP 协议或计算错误仍报错。测试覆盖一次丢失响应后的重试及日志保留。中断请求没有完整响应，不能把其耗时算进已保存测量；恢复后的首轮可能直接命中中断前已建立的磁盘缓存，因此累计已保存 bootstrap 耗时不等于总墙钟成本。
