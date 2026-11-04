# 面向端侧 Agent 的工具等待感知异构推理状态准备与动态调度

论文主线：**Slack-Aware Speculative State Preparation for Heterogeneous On-Device Agents**。Tool 开始时根据 H、工具类型和预计 G 决定是否投机准备；结果返回后，根据真实 P、预计 D 和准备进度延迟绑定 Prefill 后端，同时保留 CPU 可执行状态。KV 压缩为可选扩展。已有四路径用于成本校准与正确性验证；最小异步准备夹具见 [ASYNC.md](ASYNC.md)，真实 Tool 成本策略与端到端收益尚未验证。远端 BFCL 轨迹采集见 [BFCL.md](BFCL.md)，手机状态链路见 [MULTITURN.md](MULTITURN.md)。默认仍关闭 KV 整理、预搬移和模拟等待，暂停 QNN/NPU ViT 工作。

当前工程用同一 Android 运行器提供 CPU、HTP、混合三条路径。默认关闭 KV 整理与预搬移，先验证模板、模型响应和状态迁移。观察回放输入支持文本历史和真实截图，输出由 host 校验；不会执行模型给出的设备动作。

真截图闭环已恢复正常响应：OpenCL 视觉默认配置关闭视觉侧 Flash Attention；混合路径使用 mmap 原始 CPU 权重、batch128，图像 prefill 后释放视觉编码器，并使用 HTP 队列深度1。具体对照与结果见 VALIDATION。

## 入口

开发状态与验收见 [ROADMAP.md](ROADMAP.md)。多轮状态与最小异步准备已有基础验证，四路径代价仍部分合格。下一步先解决跨轮非阻塞资源回收，再接成本 Router 和真实 Tool；当前 `bind()` 仅按就绪状态绑定，轮末仍等待后台任务。预测特征、KV 压缩和独立 Tiling 暂缓。

历史固定CPU Decode的B矩阵16个H/P点各测3次：12点通过，当前均偏向HTP；8K历史的4点空响应，已排除选路。因此还没有证据说明 Gap 会改变最优路径，也不把这轮文本微基准当作GUI端到端加速。

- [RUN.md](RUN.md)：环境、构建与复现命令。
- [ARCHITECTURE.md](ARCHITECTURE.md)：时序、KV 迁移、计时边界。
- [VALIDATION.md](VALIDATION.md)：输出问题的定位与设备验证结果。
- [BACKEND_REVIEW.md](BACKEND_REVIEW.md)：阶段分工相关工作与当前性能配置、内存压力的对照。
- [MULTITURN.md](MULTITURN.md)：当前主线、多轮基线、Snapshot/Prefix 管理与 DeltaNet 验证边界。
- [QNN.md](QNN.md)：已搁置的 NPU ViT 路线，保留已有代码与产物。
- [configs/](configs/)：后端、KV 策略与预搬移配置。
- [scenarios/](scenarios/)：响应、历史记忆和截图场景。
- [src/a2o/](src/a2o/)：统一 Android runtime 与 host runner。
- [results/](results/)：带配置、场景、源码 hash、日志和响应校验的运行结果。
- [archive/](archive/)：既有测量、脚本和原始结果，作为历史记录保留。

## 已有单轮消融配置（历史）

| LLM 路径 | KV 整理 | 预搬移 |
|---|---|---|
| `cpu`：CPU prefill + decode | `none` / `recent` | 关闭 |
| `npu`：HTP0 prefill + decode | `none` / `recent` | 关闭 |
| `hybrid`：CPU 历史、HTP0 新观察 prefill、CPU decode | `none` / `recent` | 开 / 关 |

整理策略先使用可解释的 `recent`：保留 system、完整上一动作和最近历史 KV，原 position 不变。后续选择器从这一对照起比较。预搬移只在混合路径有实际意义：开启时在模拟 A2O 内同步历史 KV，关闭时在观察到达后同步。

NPU 配置特指 LLM 的 HTP0；图像编码使用 OpenCL。CPU 配置的图像编码与 LLM 均使用 CPU。模型固定为手机现有 Qwen2.5-VL-3B Q4_0，视觉投影为 F16。

当前设备地址由 `configs/device.json` 统一保存；`DEVICE` 环境变量或 `--device` 可覆盖。混合机制消融统一使用 batch128、关闭 CPU 额外 repack、视觉 FA 关闭和 2000ms 模拟 A2O；CPU/HTP 的单后端基线保留 batch512 与 CPU repack，运行 manifest 会记录全部配置。跨配置速度比较需同时考虑这些执行参数。

## 此前固定CPU Decode的多轮基线

`configs/multiturn-baseline.json`＋`scenarios/multiturn-screenshots.json` 在同一进程内完成 CPU→HTP→CPU→HTP 四轮 prefill 选择，decode 始终在 CPU。每轮从上一轮真实 CPU decode snapshot 继续；三张新截图各 GPU 编码一次，最后一轮纯文本追问历史。同一 snapshot 的 CPU/HTP prefill 使用相同新输入，保存原始 logits 对照。

当前模型 Qwen2.5-VL 没有 DeltaNet；本基线不宣称覆盖 Qwen3.5 的递归状态迁移。此前单轮 smoke、模拟 A2O 与 ViT 后端探索保留为历史证据，不能替代多轮验证。


## 当前四路径入口

- `configs/four-path-session.json`＋`scenarios/four-path-memory.json`：首轮GPU截图，随后历史追问，依次HH/HC/CH/CC；snapshot来自实际Decode后端。
- `configs/path-grid.json`＋`scripts/profile_prefill.py --config configs/path-grid.json`：同一结构化合成历史，两种初始驻留位置，小P与短动作的四路径校准。
- `scripts/analyze_paths.py results/NAME`：校验后导出成本与选路，单列完整动作和Decode计算前缀；Gap只是离线理想重叠。
- `scripts/plot_paths.py results/NAME`：只画通过校验且样本足够的完整动作成本，导出PNG/PDF及布局检查。

真实连续截图曾再次触发FastRPC停顿，尚未根治；完整记录见VALIDATION.md。四路径运行器与成本表不等于已完成在线成本调度。

## 本次基础机制验证

异步准备最小实现、串行短回放与验收边界见 [ASYNC.md](ASYNC.md)。本次手机存在并行竞争，耗时不用于性能结论；不运行长期实验。大型模型、状态和张量不纳入 Git，原始日志和检查记录保留。
