# CPU decode / HTP prefill 路线与当前实现核查

核查日期：2026-11-03（Asia/Shanghai）。依据本项目脚本、结果及本地 llama.cpp 源码；未修改运行时。

## 目标路线

CPU decode 产出动作 → 执行动作 → A2O 内选择/压缩历史 KV，并使必要 KV 在 HTP 可访问的 buffer 中就绪 → 新截图视觉编码 → HTP 增量 prefill → 将后续 decode 必需的 KV/状态同步至 CPU → CPU decode 下一动作。

这里的 CPU 是手机 CPU，当前实验里的 Mac host 不能直接代表最终部署。

现有文档仅部分覆盖：EXP-007 README 写了空档整理、搬成 NPU 可读布局及下张图到达后 prefill；EXP-008 写了策略原型；EXP-009 写了真实 KV 状态写回。此前没有完整写出回传 CPU decode 的循环，也没有实现这个循环。

“NPU 缓存就绪”应先定义为：HTP backend 可访问的 KV buffer、正确布局与 position/sequence 元数据、所需同步已完成。不能把它直接等同于 Hexagon 片上缓存常驻。本地 `ggml-hexagon.cpp` 使用 rpcmem 分配共享内存，并通过 fastrpc 映射；普通 F16 tensor set 路径是 memcpy。因此不应先假定必须自行实现专用 DMA；是否需要额外复制、缓存同步或布局转换，应由实际 buffer 与接口决定。此处是本地源码证据，尚未用运行时插桩确认包内全部行为。

## 当前实际链路

```
手机 HTP0 文本历史 prefill（171 token）
  → state/seq 导出到手机文件
  → adb 拉到 Mac，Python 解析 K 并构造特征
  → Mac 策略 → keep_mask → adb 推回手机
  → 新进程加载 HTP0 模型与原 state
  → seq_rm 删除 56 token，保存压缩 state
  → 再一个新进程加载 HTP0 模型与压缩 state
  → 20 token 纯文本 next prompt prefill
  → 同一 HTP0 context greedy（首 token 为 EOG）
```

入口：`EXP-009-real-kv-e2e/scripts/run_e2e.sh`；实际编译的是 `tools/kv_gui_e2e_min.cpp`，不是同目录另一份 `kv_gui_e2e.cpp`。

尚无 CPU→HTP→CPU 切换；尚无真实截图进入 EXP-009 推理。脚本虽 push 图片，但没有把图像传入工具，工具本身也没有 mmproj 路径。三阶段均重新加载模型与状态，不能代表常驻进程的时间。

## 工程与计时问题

1. **7.9ms 不是完整 prep 时间。** `policy_total` 只计 EXP-008 内部策略，不包含真实 seq 解析、外层 exemplar mask 的第二次聚类、导出、adb、状态保存/加载。`apply_ms≈0.027ms` 只包住 `seq_rm`，不包含物理压紧/状态搬移。源码中的 seq_rm 主要修改 cell/sequence 元数据，不能把这个数字当成 KV 传输时间。
2. **4.24MB 序列化结果不是固定 KV buffer 缩小到 4.24MB 的证明。** context 仍按指定容量分配；实际 attention 访问范围、空洞与 kernel shape 对耗时的影响未测。
3. **EOG 尚不能归因于压缩。** 原试验缺少 keep-all、同状态 roundtrip、有效 chat template 的对照。decode_ms≈0.28ms 在首 token 即 EOG 时没有执行普通后续 token 的 llama_decode，不是稳定 decode 吞吐。
4. **position 稀疏不等于已证实损坏。** 当前代码保留原 position 并从 pos_max+1 追加。删 KV 后维持原 RoPE 位置有其合理性；不能未经对照就认定必须 renumber。单纯修改 position 元数据也未必正确，因为 K 已包含位置编码。本地 get_can_shift 对 n_pos_per_embd>1 返回 false，M-RoPE 重定位需要特别验证。
5. **尚未测试 A2O 重叠。** 当前没有动作/UI/截图并发，也没有 CPU decode 回传；现有各阶段的局部数字不能证明循环收益。

## 策略实际做了什么

解析器把每 token 的各层 K 平均，再随机投影至 64 维并归一化；V 未用于评分。不同层/头的表示以及位置成分是否适合这样聚合，未验证。

真实 npz 没有 importance 字段，所以“task attention”退化为任务段 token 特征与该段均值的余弦，并非 QK attention，也没有利用下一截图的 query（截图尚未来到）。history 的聚类也没有按任务 query 打分。

写回实际是 exemplar-keep：每簇保留一个原始段，其余删除，不写入质心 KV。原策略逻辑输出 112 token，最终 mask 保留 115 token，是两套输出口径。

本次既有 mask 的实际结果：task 21/21、action_prev 32/32 全保留；history_0 22/22、history_2 20/20、history_4 20/20 保留，history_1、3、5 全删除。其中 history_4 明写 unrelated About phone。它被留下不能单独证明失败，但说明聚类代表性不等于任务重要性。

因此策略确实朴素，且还缺固定预算的 recent/random 对照；但没有质量指标支持“它是当前失败主因”。K/V 平均也不保证等价 attention；当前 exemplar 路径至少避免了直接假设质心 KV 可无损替代。

## 最小归因顺序

第一步已建立 EXP-010：保留 EXP-009 的原始状态、后端、无模板 prompt，分别 apply keep-all 和原策略 mask，均经 save/load 后继续生成。唯一有意改变的是保留的 KV。它回答原始 EOG 是否必须依赖压缩才能出现；不回答跨后端兼容性或 GUI 质量。

实测结果：keep-all 171 token 与 policy 115 token 均首 token 输出 `<|im_end|>`。next prefill 分别为 421.186ms、416.388ms；各一次，不能据此声称速度差异。prefill 进程 wall 分别为 15.675s、15.080s，包含模型/状态加载等。**删除造成的空洞或选择损失不是原 EOG 的必要条件**。仍需正确模板与原地/roundtrip 基线定位共同路径原因。

第二步适合用一个分组实验区分工程与选择：先让正确模板的未压缩基线有可评估输出，再固定模型、输入、位置、KV 类型与保留预算。

| 分组 | 压缩 | 执行路径 | 作用 |
|---|---|---|---|
| A | 无 | CPU 原地 prefill/decode | 正确性参考、CPU 时间基线 |
| B | 无 | CPU 历史 KV → HTP prefill → CPU decode | 不引入选择，单独验证迁移/同步/状态兼容 |
| C | 现有策略 | 同 B | B 通过后评估压缩影响 |
| D | recent / random | 同 B，同样保留预算与必保留段 | 检查复杂策略是否优于简单选择 |

先文本做接口诊断，继而同协议接入 mmproj、多图和 GUI 轨迹。后端量化 kernel 有数值差异，不能要求所有 logits 按位一致；同时比较有限值、top-k/分布差异、固定 continuation 的 NLL、短生成与任务正确率。若 B 异常，应增加 CPU→CPU / HTP→HTP state roundtrip 对照以定位序列化与跨后端问题。

- A 正常、B 异常：优先修工程，选择策略不是必要原因。
- B 正常、C 退化：压缩路径或选择造成损失；C/D 再区分策略收益，必要时与相同 mask 的同后端执行核对。
- B 正常但不快：需看迁移与同步是否抵消 prefill 收益，不能靠换 selector 解释全部问题。
- C 质量可接受且更快：才支持压缩辅助的异构路线，随后验证 A2O 重叠。

## 需要记录的时间边界

`prep = select + materialize + CPU→HTP ready（含同步）`，按实际串并行关系测 wall-clock。

`截图到动作延迟 = 未隐藏的 prep 尾部 + 视觉编码 + HTP prefill + HTP→CPU ready + CPU decode`。

常驻模型加载应单列；一次性加载可以排除，每步反复加载不能排除。回传若只传新增 KV，必须保证 CPU 已持有完全一致的压缩历史状态和 position/sequence 元数据。不能只同步新增 token，而让 CPU attention 继续看已在 HTP 删除的历史 KV。

先实测整份 KV 的正确迁移，再优化增量同步/共享 buffer；并与全 CPU 及全 HTP 的同条件结果比较，不能预设 CPU decode 必然更优。
