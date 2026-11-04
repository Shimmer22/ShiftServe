# KV tiles 的 DMA/HVX 优化与真机对照

当前 H4096/P32、64 的结果：准备阶段明显缩短，完整响应尚无稳定额外收益。默认继续关闭 KV 预整理。

## 改动

- 72 份 K/V 描述符合并为一次准备 RPC。
- 独立申请 256KiB 临时 VTCM；二维 DMA 收集输入、复用共享 HVX 整理、DMA 写回 DDR。下一块输入与当前整理重叠。
- HVX scatter 后执行 release 和同地址依赖读取，保证 DMA 读取整理结果前完成同步。
- 消费阶段拆分 enqueue/finish，在已有三份 VTCM 区域中使用两份 tile buffer 和一份尾部 scratch，恢复多 N 块时的提前加载。
- 保留普通 KV、144MiB 历史 tiles 的额外 DDR 表示、prefill 后释放和校准副本释放；无压缩、无自动回退、无跨轮增量更新。

## 完整响应

私有 v79 + host bridge；Qwen2.5-VL-3B Q4_0，F16 KV；HTP prefill + CPU decode。H4096，P32/64，G目标0/200/1500ms；预整理开/关 × 返回后/提前准备，每条件随机顺序重复5次，12条校准 + 120条测量。所有数值、状态、JSON、生成ID/EOG和实际复用检查通过。

以下比较均为“普通 KV 提前准备”和“tiles 提前准备”，单位ms。增量收益为两组响应中位数之差，正数表示 tiles 更快；配对按相同 repetition。G为服务端目标，设备实际HTTP Gap保存在原始记录中。

| P | G目标 | 普通提前准备 | tiles提前准备 | 增量收益 | 配对获益 | 普通/tiles ready |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 1560.6 | 1621.8 | -61.2 | 1/5 | 4/4 |
| 32 | 200 | 1558.8 | 1610.5 | -51.7 | 1/5 | 5/5 |
| 32 | 1500 | 1546.9 | 1555.5 | -8.6 | 1/5 | 5/5 |
| 64 | 0 | 1675.3 | 1684.0 | -8.7 | 2/5 | 5/4 |
| 64 | 200 | 1677.2 | 1690.7 | -13.5 | 2/5 | 5/5 |
| 64 | 1500 | 1704.4 | 1688.2 | 16.2 | 3/5 | 5/5 |

每组仅5次，配对中位数 bootstrap 95%区间均跨过零；不能把P64/G1500的一组正差值称为稳定收益。全部条件及配对差值见 `summary.json`，不选择最快样本或拼接阶段最小值。

## 准备成本与计时边界

本轮60条 tiles 测量的 `kv_prepack_ms` 中位数127.4ms；准备总成本171.6ms，普通准备42.7ms。旧标量内存修复轮整理中位数528.2ms，跨轮约4.1倍差异仅作参考，设备状态与条件矩阵不同，不能分别归因于DMA、HVX或RPC合并。校准逐元素核验未计入正式测量。

本轮合并条件的 prefill 中位数：普通1060.3ms、tiles1028.6ms；交接核验76.8/97.0ms；CPU decode451.6/452.5ms。各列中位数不能相加，不能从这些跨条件分布直接推断交接变慢的机制。完整响应包含约1.4ms的tiles计数读取与释放。

额外144MiB缓存和返回后的DDR→VTCM DMA依然存在。优化准备能缩短等待和提高ready概率，但准备已完成时，最终收益取决于实际attention节省及后续交接、回收和decode成本。

## 数值与构建证据

- H2048/P32：6条真机数值校准，完整prefill logits与普通HTP完全相同，最终CPU sequence状态逐字节相同；K/V各72次实际读取。
- H4096/P32、64：12条校准、120条测量；每个P独立核对完整logits和最终状态。
- 45项本地测试通过；原生ASan/UBSan覆盖延迟DMA、块间缓冲复用、两个heads、非整大块历史、尾部、非法范围和批处理部分失败回收。mock不代替硬件时序验收。
- 私有远端构建：SDK6.4.0.2/Hexagon Clang19.0.04；失败构建输出保留。shared源码、安装库和大资产不复制、不覆盖。

## 原始记录

- 数值验收：[H2048](../20261104-164845-801765-http-oracle-grid/diagnostic-summary.json)
- 完整对照：[H4096](../20261104-165046-094246-http-oracle-grid/README.md)
- 测量构建：[manifest](../20261104-164744-178763-kv-prepack-build/manifest.json)
- sync消融仅已构建：[manifest](../20261104-165028-430600-kv-prepack-build/manifest.json)
- 本地测试：`local-tests.log`
- 汇总：`summary.json`

## 独立 attention profile 与真实分块

独立 profile=1、H4096/P32 的6条校准全部通过；不并入前述正式计时。三个 HC 各有72个 F16 attention MUL_MAT，普通路径合计65.234ms，tiles两次52.704/52.028ms，约节省12.5–13.2ms。普通参考只有一次，不能当成稳定性能估计。

本机实际 VTCM 为8388608字节（8MiB）；K 的 N chunk=4352、V 的 N chunk=128，二者均只有一个 N 块，新增/填充尾部已计入形状。因此本轮没有下一块 DMA 可与当前块计算重叠；恢复双缓冲是多块路径的实现修复，不能用来解释本轮时延。此前将缺失双缓冲列为本场景拖累的推测，实测不支持。

准备阶段虽然使用临时VTCM，持久 tiles 仍在DDR；计算时全部历史数据仍要进入VTCM。当前证据支持存在小幅attention算子收益，尚不支持完整响应的稳定额外收益。保持显式实验开关，不作为默认策略。

- profile：[记录](../20261104-170047-910898-http-oracle-grid/diagnostic-summary.json)
- profile构建：[manifest](../20261104-165922-790435-kv-prepack-build/manifest.json)
- profile原始日志包含 `a2o-kv-layout`；该诊断通过私有stats缓冲传回，未扩展wire ABI，前六个计数字段保持不变。
- 仅含FARF布局输出的中间构建 `20261104-165705-590191` 未用于正式测量或真机结论；两个编译失败目录保留原始错误。

复现汇总：

```bash
python3 scripts/analyze/kv_dma_hvx.py results/20261104-165046-094246-http-oracle-grid \
  --profile results/20261104-170047-910898-http-oracle-grid \
  --output results/20261104-kv-dma-hvx
```

最终设备清理和共享库hash核验见 `final-checks.json`。
