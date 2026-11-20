# 主线实验总控台

本文档作为论文封版前主线实验的统一索引与状态看板。

## 一、封版构建与统一开关口径

- **手机端封版构建（已核验源码一致性）**：[`results/20261119-splitlm-main-merge-build`](../../results/20261119-splitlm-main-merge-build/manifest.json)
  - 源码核对结论：`manifest.json` 记录的全部原生源文件 SHA256 与当前 `main` 主线（`03ca4e4`）**100% 一致**，可直接复用，无需重复交叉编译。
  - 构建镜像：`ghcr.io/snapdragon-toolchain/arm64-android:v0.3`
  - 产物 SHA256：
    - `agent`: `7cec7889b851d8fedac9eec7562ff06ccaecd5a345f5c26e01c365493c607e73`
    - `libggml-cpu.so`: `1c8ee90c7cc23a87f72dd8928e3a4aefde3d1861018cea5f7dd6cd8fea723e91`
    - `libllama.so`: `6048ea85a684dc5fa2bfd5ab555da203818a2c3a1160e152520c22838cd0d855`
- **主线默认运行参数（M1 + M2 完全体）**：
  - 共享权重：`A2O_SHARED_WEIGHTS=1`（`--shared-weights`）
  - M2 KV 裁剪：`A2O_KV_REDUCE=1`、`A2O_KV_POLICY=prefill_prior`、`A2O_COMPACT_ANCHOR=1`
  - 采集税优化与内存保护：`A2O_KV_LAYER_STRIDE=4`、`A2O_KV_MIN_EVICT_TOKENS=256`、`A2O_KV_HWM_TRIM=1`（高水位差值门槛 `3072` tokens）

## 二、文档与结果目录索引

- 已有实验数据全量基线：[已有结果梳理](existing-results-summary.md)
- 待补关键实验计划与记录表：[主线实验计划与记录表](mainline-plan.md)
- 本地原始结果根目录：`results/mainline-eval/`（详见 [results/mainline-eval/README.md](../../results/mainline-eval/README.md)）

## 三、主线实验进度总表

| 编号 | 实验名称 | 优先级 | 执行端 | 当前状态 | 对应结果目录 | 记录位置 |
|---|---|---|---|---|---|---|
| **A** | `main-4b-ablation`（19 轨 `shared_prior` 封版重跑 + `static_hc` / `cold_route` 扩展） | **P0** | 手机（SM8650） | **第一阶段已完成**（`shared_prior` `1,220.9 s`）；第二阶段待扩展脚本 | `results/mainline-eval/main-4b-ablation/` | [实验 A 记录](mainline-plan.md#a-main-4b-ablation4b-多-benchmark-封版主表--p0) |
| **B** | `host-accuracy-align`（`tau2-ratio-8b` 收尾 + `stride=4` 准确率对齐） | **P1** | 主机（`wins` 4090） | **`tau2-ratio-8b` 已结束**（`m2_prior_high` 23/36）；`stride=4` 待跑 | `results/mainline-eval/host-accuracy-align/` | [实验 B 记录](mainline-plan.md#b-host-accuracy-align主机准确率收尾与稀疏层对齐--p1) |
| **C** | `gap-sensitivity`（Tool Wait Gap 与输入形状敏感性扫描） | **P1** | 手机（SM8650） | **已完成**（`1231c04` 构建，5 档 × 3 条件 × 3 轨全通过） | `results/mainline-eval/gap-sensitivity/` | [实验 C 记录](mainline-plan.md#c-gap-sensitivitytool-wait-gap-敏感性扫描--p1) |
| **D** | `main-8b-scaling`（Qwen3-8B 长上下文 4–6 轨扩展验证） | **P1** | 手机（SM8650） | 待启动 | `results/mainline-eval/main-8b-scaling/` | [实验 D 记录](mainline-plan.md#d-main-8b-scalingqwen3-8b-长上下文扩展验证--p1) |
