# 历史 KV 的 HMX tiles 提前准备

用于回答：在已能提前迁移状态之外，把历史 KV 的内部 HMX 整理也放进真实 Tool Gap，能否进一步缩短返回后到完整响应的时间。正文入口保留在 `docs/guides/`。

## 实现与范围

默认关闭；只在 Oracle 原型中用 `--htp-kv-prepack 1` 显式启用。私有 host bridge 通过 `a2o_kv_control_v1` 提供失效、准备和计数查询，使用现有 profiler RPC 的保留 mode `0x41324f50`，不改 wire ABI，不与 profile=2/PMU 诊断混用；profile=1 仅允许独立校准诊断，不并入正式时延。原安装库与 shared 源码不变。

当前只支持 Qwen2.5-VL-3B 的 F16、单序列、36 层、2 个 KV heads、head dimension 128，且 H 为 32 的倍数。维度或库不满足要求直接失败。每次 trial 在恢复共同 CPU 状态前失效旧缓存；HTP 导入历史状态并逐字节核验后，将历史 K/V 整理成 tile-major 表示，额外保存在 DSP DDR。H4096 占 144MiB。GQA batched F16 attention 读取这些历史 tiles，新增/填充尾部仍从普通 KV 在返回后整理，CPU decode 与状态导出继续使用普通 KV。

这不是让全部 KV 常驻 VTCM：后续计算仍需把 tiles DMA 到 VTCM。Q/K/V 新增投影、attention 激活、HTP→CPU 迁移与 decode 不能在未知 Tool 响应时完成。没有跨轮增量更新、滑动窗口或多序列支持；不能将缓存直接用于普通会话。

## 构建与运行

复用 `ssh wins` 缓存镜像，构建脚本在独立远端目录用共享源码的符号链接创建私有工程，只替换必要源文件。DSP 使用 SDK 6.4.0.2/Hexagon Clang 19.0.04；Android host bridge 复用原编译命令与未修改对象文件重新链接。构建 manifest 保存源码、补丁、辅助源码与二进制 hash；失败输出保留在具名 `results/` 目录。

```bash
python3 scripts/build/kv_prepack.py
# 将打印的 build/kv-prepack/<时间戳> 写入配置 htp_kv_prepack_dir，再串行运行。
python3 scripts/profile/oracle.py --config configs/oracle-kv-prepack-smoke.json --device <当前设备序列>
python3 scripts/profile/oracle.py --config configs/oracle-kv-prepack-4k.json --device <当前设备序列>
python3 scripts/analyze/kv_prepack.py results/<本轮目录>
```

私有 bridge 放在本轮可执行文件旁，私有 DSP 库放在本轮 `probe-lib/`，仅用进程加载路径选择。启动前拒绝已在运行的模型实验，验证设备库 hash；结束拉取原始记录并移除本次 reverse 和临时目录，失败则保留副本。不得覆盖共享安装库。

## 验证与计时

每个 P 的第一条 HC 为未预整理参考，后两条预整理 HC 要求完整 prefill logits 数值完全相同、最终 CPU sequence 状态逐字节相同；此外保留 CPU/HC 原数值门槛、同一 JSON、完整生成 IDs、EOG 与状态迁移检查。校准额外逐元素核验 tiles，校准时延不混入测量。

`prepack_comparison=true` 的测量矩阵固定 HC：预整理开/关 × 返回后准备/提前准备，每条件随机顺序重复。配置保留通用 Oracle 字段，但本分支不运行 Router 或成本感知选路，CC 只用于校准。比较“普通提前准备”与“tiles 提前准备”可隔离 tiles 的额外收益；比较两种准备时机则衡量真实并发重叠的收益。G 是服务端目标，报告还保存手机实际 Gap 与 ready 状态。

`prepare_ms` 包含 HTP 状态恢复、核验、缓存分配、RPC 与 tile 整理，`kv_prepack_ms` 单列后半段。共同历史生成、快照和每个 trial 的 fixture 重置/旧缓存回收在 HTTP 发起前。模型初始化和历史生成在 session_ready 中单列；这是文本实验，视觉初始化为零。缓存计数查询与额外校准状态比较在响应发布后，不计入完整响应。

每次预整理 HC 必须读取全部历史 K/V，H4096 对应 K/V 各 72MiB、各 72 次读取；不只验证缓存存在。原始记录、实际重叠区间、响应发布事件、分组耗时和完整性检查保留。3 次重复仅提供当前固定场景的探索结果，不能把单次最快记录或阶段最小值拼成理论收益上限。

## 本轮记录

- [H4096/P32、64 对照](../../results/20261104-151046-645490-http-oracle-grid/README.md)
- H2048/P32 验收：`results/20261104-145934-187427-http-oracle-grid/`，6 条记录通过，预整理 HC 与原 HC 的 logits 完全相同。
- `results/20261104-145315-989153-http-oracle-grid/` 保留首轮二维 DMA 参数顺序错误导致的 DSP 异常；已修正并通过后续实际计算验收，不把该失败轮用于性能汇总。

## 内存生命周期修复与回归

[错误排查记录](../../results/20261104-kv-prepack-audit/README.md)发现三处问题：缓存加载失败时漏释放 HMX 锁；历史 tiles 在 CPU decode 期间继续占用额外 144MiB；校准专用完整状态副本在测量阶段仍驻留，原 P32/P64 两条件约占 290MiB。布局与数值验收通过不能排除这些性能问题。

修复配置为 `configs/oracle-kv-prepack-memory-fix.json`，两个显式开关分别是：

- `htp_kv_release_before_decode=true`：HTP prefill 后读取计数并释放 tiles，释放时间计入完整响应；响应后再次核验缓存清空。普通 F16 KV 仍用于交接和 CPU decode。
- `release_calibration_state=true`：第一条测量发起前释放校准的状态/logits 副本，保留准入结果、输出与生成 ID。原始校准记录已落盘；进入测量后拒绝再次校准。

新增 CPU decode 进程级 CPU 用时、minor/major page faults 与调度切换计数，涵盖本进程所有工作线程。major page faults 支持存在换页/页面重新调入，不能单独确定被逐出的页属于模型权重还是其他映射。不同生命周期条件串行分轮，仍存在设备状态漂移，不能把跨轮差值直接归因于单个开关。

原矩阵和失败日志全部保留；新的诊断结果不覆盖原始记录，也不把它当成 Router 或联合 Oracle 的收益测量。
