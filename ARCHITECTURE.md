# 执行路径与状态契约

当前论文主线为 Tool Slack 投机状态准备与返回后延迟绑定，见[ROADMAP.md](ROADMAP.md)。GPU视觉＋CC/HH/HC/CH为手机已有成本校准与正确性组件；`four-path-session.json`显式指定每轮Prefill/Decode后端，snapshot从当轮实际Decode后端导出。最小异步准备、版本校验、安全点取消与就绪绑定夹具见[ASYNC.md](ASYNC.md)，真实工具和成本选路尚未实现。固定CPU Decode为此前基线，见[MULTITURN.md](MULTITURN.md)；远端文本轨迹见[BFCL.md](BFCL.md)。QNN/NPU 视觉已搁置；下文保留既有执行契约。

## 单个进程内的回放

```
加载模型/上下文
  → 按模型 chat template 构造 system、历史 user、上一 assistant 动作
  → home backend 生成历史 KV
  → 可选 recent 整理
  → 可选在 A2O 内 CPU→HTP 同步历史 KV
  → 观察到达
  → 若未预搬移，此时 CPU→HTP 同步
  → 新 user 观察与 assistant generation prefix 增量 prefill
  → hybrid：HTP→CPU 同步更新后的完整序列状态
  → home backend decode
  → host 校验响应
```

home backend 在 cpu/hybrid 为手机 CPU，在 npu 为 HTP0。混合路径为两个独立模型/上下文；模型初始化在本次回放开始时完成，阶段间不重载。当前是一进程一回放请求，尚未作为常驻多请求服务接入设备动作循环。

## 模板与 logits

使用模型元数据中的 chat template，`add_ass=true` 生成 assistant 起始标记。历史 prefix 与完整 prompt 做前缀检查，只评估新增 suffix。历史末尾包含上一 assistant 动作的完整 framing。

序列状态接口不携带可直接供 CPU 采样的最新 logits。混合路径先在 HTP prefill 的 logits 上选取首 token，再在 CPU 上评估这个 token，后续由 CPU logits 继续 decode；不读取状态恢复后陈旧或未初始化的 CPU logits。

EOG 是正常停止 token，不作为普通输出文本。首 token EOG 会产生空响应并在 host 检查失败，不再被当成“正常 Agent 推理”。

## KV 迁移

CPU 与 HTP 固定相同 ctx2048、F16 KV、关闭 LLM Flash Attention，以对齐 V 布局。迁移使用内存中的 `llama_state_seq_get_data` / `llama_state_seq_set_data`，不走文件和 adb。包含 K/V、原 position 和 sequence 元数据。

每次迁移均 synchronize，并重新导出目的状态，与源字节比较。该检查能验证序列化数据完整性；实际 CPU decode 响应还要单独验证。`*_checked_ms` 包含目的端重新导出和比对的调试成本，不能当作优化后的迁移基准。

`recent` 在原 position 上删除中间历史，保留 system 与上一 assistant 动作（含角色标记），不进行位置重编号。迁移后源/目的都使用同一压缩状态。回传先传完整更新状态，后续再考虑只传新增 KV；不能让 CPU 保留已经在 HTP 删除的历史。

HTP backend 使用 rpcmem/FastRPC 共享 buffer。当前“预搬移”表示提前把历史状态装载到 HTP context 并完成同步，不声称 KV 已常驻片上缓存。

## 图像

图像通过 `mtmd` 真正 tokenize 成 multimodal chunks，按 chunk 分别调用 `mtmd_encode_chunk` 和 `mtmd_helper_decode_image_chunk` 执行视觉编码和图像 prefill，使用 helper 提供的 M-RoPE position。CPU 视觉使用 CPU；HTP/hybrid 视觉使用 OpenCL，LLM 使用 HTP0。所有剪枝环境变量需关闭，后续若加入视觉机制需单独配置。

`vision_backend` 可独立选择 cpu/opencl/htp，`vision_flash` 可选择 auto/off/on。截图对照发现 OpenCL 视觉的 auto Flash Attention 路径回答错误；明确 off 后恢复，故当前 HTP/hybrid 配置显式设 off。CPU 视觉仍可用 auto。host 按日志核对实际 CLIP backend，不接受静默回退。

混合默认 `cpu_repack=false`、`batch_size=128`：CPU 权重直接 mmap 原 GGUF，避免另一份 CPU_REPACK 匿名内存；两上下文均以同样 batch 大小创建。HTP 权重依然使用其必要的 repack，不改变模型量化身份。纯 CPU/HTP 基线与该执行配置的差异完整保存在 manifest 中。

本运行器每进程仅处理一个观察；prefill synchronize 并取得首 token 后立即释放视觉编码器及 scratch，而不是留到 CPU decode 后。这使约 1.3GiB 视觉权重及图计算缓存不再与 CPU decode 争抢内存。`vision_release_ms` 与 `vision_released` 资源快照单独记录。未来常驻多请求服务需要另行评估视觉模型驻留/重载策略，不能直接套用本次冷启动时间。

混合配置显式使用 `htp_queue=1`（`GGML_HEXAGON_OPQUEUE=1`）。默认队列深度下曾发生末尾文本 prefill 停在 FastRPC 等待的情况；队列深度1作为当前诊断配置保留，未修改共享后端源码或系统设置。响应超过180秒会终止经 PID 与命令行核对的本次 runtime，保留日志并报错，不自动重试或换后端。

响应允许普通 JSON 或一个完整的 json 代码围栏。原始生成文本始终保存，代码围栏的 strict_format_passed=false；不会从任意解释文字中猜测、抽取或修复动作。截图场景同时检查页面识别与动作字段，JSON 合法不等于截图理解正确。

## 时间

- `model_context_init_ms`：模型与上下文初始化。
- `history_prefill_ms`：构建回放的初始历史；部署后应由历史步骤自然产生。
- `kv_select_ms`：历史整理。
- `upload_checked_ms` / `download_checked_ms`：包含验证的历史上行/更新状态下行。
- `a2o_wall_ms` / `prep_overrun_ms`：模拟空档和未能被预算覆盖的准备成本。
- `vision_init_ms`：视觉模型初始化；目前每回放一次，单列记录。
- `observation_eval_ms`：文本增量，或视觉编码加图像/文本 prefill；视觉内部时间另见 mtmd 日志。
- `decode_ms`：首 token 确定之后的 token 评估与后续采样。
- `observation_to_response_ms`：观察到达后的整体处理，含延迟上传、视觉初始化/编码、prefill、回传和 decode。
- `process_wall_ms`：完整 adb shell 运行耗时，含本次模型加载与退出。

host 的 `budget_adjusted_response_ms = observation_to_response_ms + prep_overrun_ms` 将超出 A2O 预算的准备尾部计回延迟。默认 smoke 的 A2O 预算为 0，因此不能仅看预搬移条件的 observation_to_response 就声称隐藏了搬移成本。

一次 smoke 验证接口与响应。延迟结论需要模型/视觉常驻、重复与顺序控制；测预搬移收益时应使用有历史的同一场景。

运行结果的 `resources` 记录上下文加载后、历史就绪、观察到达、观察计算完成、decode 起止的进程 RSS、swap（KiB）与累计 minor/major faults。比较 `decode_end` 与 `decode_start` 的缺页增量，配合 decode 时间定位双模型驻留问题；全机 swap 总量不能替代该进程的分阶段证据。

## NPU–CPU 协同与提前准备

目标是 HTP ViT＋projector＋LLM prefill，CPU decode。`htp` 是显式请求视觉 HTP0 的配置，**不表示所有视觉算子均支持 HTP**；调度器仍可将算子分配 CPU。native 在视觉初始化后立即核对实际后端名，设备缺失或初始化回退直接失败。`trace_vision=true` 记录 `VISION_NODE` 的算子、shape 与输出缓冲区；回调会改变 split 同步行为，只用于分配诊断，不能混入正式耗时比较。`VISION_CAP` 向手机当前二进制查询 F16 线性层不同 token 数的支持情况，不分配实际矩阵数据。

`vision_prepare=arrival` 在观察到达后加载视觉模型；`a2o` 在空档内加载视觉模型和初始化视觉上下文，可以与 `pre_move` 独立配置。尚未提前构建依赖输入分辨率的真实图或执行虚拟图像 warmup；不读取下一截图来准备。模型/LLM 上下文初始化仍在本次回放的空档之前完成，不计作被 A2O 隐藏的工作。

`prep_work_ms` 记录选 KV、提前上传及提前视觉初始化的工作时间；`prep_overrun_ms` 按准备工作实际结束时间计算，排除睡眠唤醒抖动。超过 2000ms 模拟预算的尾部必须加回响应延迟。`image_preprocess_ms`、`vision_encode_ms`、`llm_prefill_ms` 将观察处理细分，保留整体 `observation_eval_ms`。观察后及时释放视觉资源的策略沿用既有基线。

运行 `npu-cpu` 串行对照时保持同一截图、文本历史、Q4_0 LLM、F16 mmproj、batch128、CPU mmap、KV 不整理。CPU/OpenCL/HTP 指视觉后端；LLM prefill 均 HTP、decode 均 CPU。HTP 按需和 A2O 条件只改变准备位置和历史预上传。所有条件仍是一进程单观察回放，不能声称验证了真实动作并发或多轮常驻。
