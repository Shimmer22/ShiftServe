# KV tiles 原型错误排查与回归

本次确认了不必要的内存驻留和一个错误分支漏锁。未发现当前支持形状下的 KV tile 布局错读；原来的大延迟损失不能直接解释为预整理机制无效。修复后仍不能宣称已有稳定的端到端增益。

## 已确认并修复

1. 校准快照未释放：`htp_reference_state` 在数值/状态验收结束后仍随整个测量阶段驻留。原 P32/P64 两条件合计约 291.5MiB。新增显式开关，在第一条测量发起前释放校准状态与 logits 副本，保留响应签名、生成 ID 和准入结果。最终回归的每条测量均确认保留的校准状态字节数为零。
2. tiles 生命周期过长：HTP prefill 后，CPU 交接/decode 只使用普通 KV，但额外 144MiB tiles 直到下一条 trial 才释放。新增显式开关，prefill 后先取计数再释放；释放 RPC 时间计入完整响应，响应后再次核验缓存清空。
3. HMX 锁泄漏：缓存加载返回 false 时直接退出，未释放已持有的 HMX 锁。现已解锁再返回，并补齐分块对齐、容量与 V 的 K 范围检查。正常路径的旧记录没有触发这一错误分支，不能用它解释已完成测量的变慢。
4. 实验互斥检查漏检：Android `ps` 实际可能显示 `agent --oracle-profile ...`，旧检查只匹配绝对路径。现按 PID/可执行 basename 检查，避免漏掉已有模型进程；不匹配系统的 otrpagent 服务或检查用 shell。没有证据表明此前三轮发生模型实验并发。

## 算子与内存证据

独立 basic profile（H4096/P32，6 条完整校准）中，普通 HC 的 72 个 F16 attention MUL_MAT 合计 66.119ms；预整理两次为 55.699/56.427ms。缓存确实减少了约 10ms attention 时间，不能从总响应长尾推断 attention 变慢。该 profile 含诊断开销，普通 HC 只有一次，不作为稳定性能估计。

带资源计数的保留缓存回归中，trial 6 完整响应为 16.671s，其中 CPU decode 为 14.546s、major page faults 为 169935。这直接支持该慢样本存在严重页面调入；它不能单独确定被逐出的页属于模型权重、匿名页还是其他映射，也不能排除调度/热状态的影响。

三轮均为 H4096/P32、G目标 0/1500ms、预整理开/关 × 返回后/提前准备；每条件三次，6 条校准＋24 条测量。相同种子和条件顺序，设备模型实验串行执行。以下为诊断概况，单位 ms：

| 生命周期 | 普通 CPU decode 中位数 | 预整理 CPU decode 中位数 | 最大完整响应 | 最大 decode major faults |
|---|---:|---:|---:|---:|
| 保留旧驻留 | 667.9 | 985.7 | 16671.4 | 169935 |
| 仅提前释放 tiles | 582.0 | 586.3 | 2634.7 | 1491 |
| tiles＋校准副本均释放 | 587.9 | 622.4 | 3279.6 | 1640 |

两轮释放版本未再观察到十几秒长尾，预整理组 decode 的 major faults 中位数由 1326.5 降到 467/159。各轮普通路径本身也有变化，且第三轮同时释放校准副本；这些是串行探索对照，不能把跨轮差值全部归因于某一个开关。G=0 仍包含真实 adb reverse/HTTP 开销，实际 Gap 和 ready 均保留。

## 验证与复现

共新增 96 条真机记录：6 条独立 profile＋三轮各30条回归。完整 JSON、相同生成 ID/EOG、数值准入、迁移检查均通过。预整理 HC 与普通 HC 的 prefill logits 数值完全相同，最终 CPU sequence 状态逐字节相同；释放版本在响应后核验空缓存。

22 项本地测试通过，其中原生测试运行真实的预整理和缓存读取源码，以 DMA mock 做逐行地址检查，并开启 AddressSanitizer/UBSan；覆盖 H64/2048/4096、两个 KV heads、多种分块、新增及填充尾部、错误范围和释放后访问拒绝。mock 不验证真实 DMA 的异步时序，真机回归独立验证实际计算。

配置：`configs/oracle-kv-prepack-audit.json`、`oracle-kv-prepack-regression.json`、`oracle-kv-prepack-release.json`、`oracle-kv-prepack-memory-fix.json`。普通 4K/smoke 配置也已改用新的私有库和释放开关。默认的非预整理运行保持关闭。

当前缓存分支仍采用逐块同步加载，准备阶段仍是标量整理；未在本轮恢复原算子的下一块 DMA 预取，也未把标量整理优化为 HVX。这些是实现效率限制，尚未证明它们导致本次 CPU decode 长尾。

```bash
python3 scripts/analyze/kv_prepack_audit.py \
  results/20261104-154057-153040-http-oracle-grid \
  results/20261104-154652-284128-http-oracle-grid \
  results/20261104-151046-645490-http-oracle-grid \
  --release results/20261104-155217-896642-http-oracle-grid \
  --fixed results/20261104-155714-318982-http-oracle-grid \
  --output results/20261104-kv-prepack-audit
```

原始目录：
- profile：[记录](../20261104-154057-153040-http-oracle-grid/diagnostic-summary.json)
- regression：[记录](../20261104-154652-284128-http-oracle-grid/README.md)
- original：[记录](../20261104-151046-645490-http-oracle-grid/README.md)
- release_only：[记录](../20261104-155217-896642-http-oracle-grid/README.md)
- memory_fixed：[记录](../20261104-155714-318982-http-oracle-grid/README.md)

全部检查与清理状态见 `final-checks.json`；逐算子、分组原始 ID 和阶段计数见 `audit.json`。原安装 host bridge/v79 库 hash 与 shared 包一致，已确认无模型进程、无 reverse 和本轮临时目录。原始失败与历史实验记录保留，未改手机设置、频率或用户数据。
