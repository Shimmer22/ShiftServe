# Tool Gap 中预测准备 NPU 执行计划

私有运行时已编译，48项本地测试、11项短历史真机机制验收通过。命中时实际复用同一图和全部832个HTP算子描述符；准备前后CPU/HTP状态不变，完整logits、prefill后KV、最终CPU KV与普通路径完全一致。默认关闭，示例配置已指向通过验收的私有库。此结果不代表真实Tool会话的端到端收益，记录见[验收报告](../../results/20261104-execution-preparation/README.md)。

## 准备的内容

已有 Tool 会话在 Gap 中将 CPU 权威历史导入独立 HTP 上下文。新步骤在这之后，按一个预测的首 microbatch 形状提前准备：

1. 使用与实际 decode 相同的 batch allocator 和 KV 槽位规划，计算追加后所需的 KV 视图宽度。KV 槽位规划原有的模拟更新会还原元数据；新 preview 不执行 `apply()`，不写 K/V 内容。
2. 在该 HTP 上下文中构建真实 decoder 图并调用 scheduler 分图、分配输入/激活/临时缓冲区，保留 `gf_res_prev`。不调用 `set_inputs` 或图执行；占位 token 仅供 batch 形状分配，不产生预测输出或新的 KV。
3. 按 scheduler 已重写跨 backend 输入的真实 split 构造 HTP opbatch、buffer/tensor/op 描述符。缓存包含地址、形状、stride、参数及输入指针的精确快照，不只按 P 命中。
4. 通过现有 `htp_iface_mmap` 提前完成所需 DSP 页映射。Host FastRPC 映射沿用现有缓冲区所有权，不复制权重或额外 KV 表示，也不让普通 KV 常驻 VTCM。
5. 可选、有预算的页预触及仅写临时 COMPUTE 缓冲区的页首字节；默认预算0。权重、普通 KV 和用户映射不在该路径中。页映射仍会准备。

准备入口不调用 `enqueue_op`、`flush_batch`、`graph_compute` 或 `dspqueue_write`。实际 Tool 返回、绑定真实输入并进入正常图执行后，才能提交描述符。

## 命中、失配和生命周期

只持有一个候选，复用现有 `PreparationLane`。预测 key 包含快照版本、下一位置、首 microbatch 的精确 token 数；上下文固定单序列、token 输入、每 microbatch 最后一个 token 输出。跨上下文并发准备、SWA、位置迁移、embedding输入、backend sampling 和 pipeline parallel 均拒绝。

Tool 返回后，先等待 lane 的 ready 判定并 claim，再检查实际首 microbatch 的形状和版本。失配会释放描述符缓存并 reset 预测图，继续显式 HTP 上下文的普通构图，记录失配原因。这里不切换计算设备；CPU/HTP 选路仍由原有 Router 决定。缺失私有 API、预算超限或映射错误会直接失败。

实际 `process_ubatch` 再检查位置、序列和 output mask，随后沿用运行时的 `can_reuse`。只有它真实复用了同一张图才增加 `graph_hits`；只有实际提交了缓存描述符才增加 `descriptor_hits`。准备完成和 bind 命中都不等价于实际复用。

每次重新导入历史前失效旧计划；私有 `state_seq_read_data` 也加入失效处理。低层直接调用 `llama_memory_clear/seq_add/seq_rm` 的其它新调用方必须先 discard，不能绕过会话生命周期。候选持有 preview batch/memory context，至少活到输入上下文被重新绑定或旧图销毁。退出时先 join lane，再释放执行计划，最后销毁 HTP 上下文。映射跟随原缓冲区释放，不在丢弃计划时拆掉正常执行仍需的映射。

## 范围和参数

首版限当前 Qwen2.5-VL 的普通 F16、无 SWA KV 和纯文本 Tool 会话。预测的是首 microbatch，不是完整未知响应。较长逻辑 P 的后续 batches 走正常 decode；图/地址仍兼容时，描述符模板可以继续复用，但不能宣称提前构建了整个未知 P 的所有图。

`predicted_input_tokens` 默认64，范围1..min(batch,64)。这是显式固定预测，尚无学习型长度预测器，也不对实际输入自动填充。选64是因为当前 host 对 F16 激活行数的1024上限，16个 Q heads 下 M128会使 attention 不被 HTP 接受；本分支不修改已有算子准入范围。

描述符预算8MiB，映射地址空间预算4GiB，页预触及预算默认0、最多64MiB。映射字节数是去重缓冲区大小之和，不是新增物理内存。描述符预算核算保留的 wire 数组及 stamp；容器、allocator 与 scheduler 的全部峰值内存仍需后续测量。缓冲区复用 scheduler 的现有保留空间，不额外保存一份模型或144MiB tiles。

## 文件和后续入口

- `src/a2o/native/backend/execution-preparation.h`：私有版本化 ABI。
- `src/a2o/native/backend/llama-execution-context.inc`：真实构图、buffer 分配、绑定与复用钩子。
- `src/a2o/native/state/kv-execution-preview.inc`：不提交 KV 的槽位/视图预览。
- `src/a2o/native/scheduling/ggml-execution-prepare.inc`：遍历真实 scheduler splits。
- `src/a2o/native/backend/htp-execution-prepare.inc`：描述符模板、页映射、可选预触及和实际提交。
- `src/a2o/native/scheduling/execution_preparation.h`：Tool lane 的客户端封装。
- `scripts/build/execution_prepare.py`：未来构建私有 `libllama.so`、`libggml-base.so`、`libggml-hexagon.so`，原 DSP 库继续复用。只读共享源码与原对象，修改源与产物在独立目录。
- `configs/tool-session-execution-prepare-dev.json`：显式启用的开发配置；既有配置默认关闭。

私有构建与短历史验收入口：

```bash
python3 scripts/build/execution_prepare.py
# 将输出目录写入 execution_prepare_dir。
python3 scripts/validate/execution_preparation.py --device <设备序列> \
  --runtime-dir build/execution-prepare/<本轮构建>
# 真实Tool会话的集成与收益另行验证：
python3 scripts/run/tool_session.py --config configs/tool-session-execution-prepare-dev.json \
  --scenario scenarios/<已有只读Tool场景>.json --device <届时设备序列>
```

事件 `execution_plan_prepared` 分别记录 graph/allocate/backend 耗时、HTP节点、描述符、映射和预触及字节；`execution_plan_bound` 记录形状/版本命中；`execution_plan_consumed` 记录真实复用和 miss。当前已验证私有链接、状态不变、精确命中、形状/版本失配、丢弃、1MiB页预触及、后台lane准备及准备完成后的异步取消。后续仍需验证更多模型/形状、预算失败、取消发生在API调用中时的回收、完整真实Tool会话，以及普通KV提前迁移基线上的稳定增量收益。

本次修复：scheduler分图分配会更新全图UID，预测计划必须在 `alloc_graph` 完成后记录UID。分配前记录会将真实图复用误记为失配，并丢弃已经准备好的描述符。失败运行和一次首次HTP处理超时均保留，未重置设备服务或修改设置。
