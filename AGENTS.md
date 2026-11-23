# 项目约束

- 今后手机操作默认使用 Termux SSH：`ssh -p 8022 -i ~/.ssh/a2o_xiaomi14pro_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/hui/code/exp/2026-11-a2o-gap-profiling/configs/termux-known-hosts u0_a320@10.29.131.50`，实验期间持锁且不回退到 ADB。

- 维护一个工程：代码放 src/，执行入口放 scripts/，配置放 configs/，场景放 scenarios/，每次结果放 results/ 的具名目录。历史实验保留在 archive/，不增加 EXP 编号目录。
- src/a2o 按机制分包，原生代码放 native/ 并按 backend/state/scheduling/session/vision/profiling 划分。scripts 与 docs 的正文入口必须放二级目录；根目录只留 README.md 和 AGENTS.md，docs/README.md 作为文档索引。
- 中文说明，复用 shared/ 下已有运行时、模型与工具链，不复制大资产。默认文本成本实验使用 Qwen3-8B Q4_0（configs/model-default.json），明确记录 CPU / HTP offload；Qwen2.5-VL-3B 与 F16 mmproj 仅保留为历史视觉验证入口。
- 远端编译用 `ssh wins`（`10.29.243.183`）的缓存镜像 `ghcr.io/snapdragon-toolchain/arm64-android:v0.3`：SDK 6.4.0.2、Hexagon Clang 19.0.04；原源码/产物在 `/home/wins/ai-workspace/tools/llama.cpp-hexagon-build/`，探针另建私有目录。
- CPU / HTP / hybrid 必须显式配置并记录，设备不可用时直接失败，不静默回退。模型设备实验串行执行，避免不同条件争抢手机资源。
- 论文端侧正式实验默认使用 CPU 4 线程（Prefill 和 Decode 均为 4）及 HTP 主机 4 线程。线程专项诊断单独记录偏离配置。
- 默认关闭 KV 整理和预搬移；先通过正常响应和状态迁移检查，再比较策略与延迟。
- 同一条件的 prefill/decode 使用一个进程内的上下文；模型初始化、视觉初始化、观察处理、状态迁移和 decode 分别计时。
- 保留失败输出与原始日志；退出码成功不等于响应通过。实际图像输入、JSON 响应、历史记忆和状态迁移分别验证。
- 不更改手机设置、CPU 频率或用户数据。回放工具不自动执行生成动作。

- `main` 后续研究范围限定为 CPU/HTP、Tool Gap 内状态准备、返回后选路与 P 分布投机价值。GPU/OpenCL 对照仅保留在 `codex/gpu-opencl-study` 独立实验分支，不合入主线、不纳入主线默认候选后端或后续结论，除非用户重新要求。
