# EXP-008: GUI 历史 KV — keep(上一动作 A) + 任务关注度加权 + history 相似度 merge

## 问题
在 A2O 空档内整理历史 KV：上一动作对应 KV **不动**；总任务相关 KV **保留但按关注度加权**（高分保留，中低分可进 merge）；其余历史按相似度聚类 merge。本轮跑通最小 host 侧链路并计时，对照 EXP-005 A2O 预算。

## 方法
- **体量**：Qwen2.5-VL-3B LLM KV f16，`bytes/token = 2×36×2×128×2 = 36864`（与 EXP-006/007 一致）。
- **输入**：无现成 dump → **合成**段列表（`task` / `action_prev` / `history_*`）；特征为 `FEAT_DIM=64` 代理向量（**非真实模型 KV 张量**），字节按真实公式计。可选 `--npz` 加载派生特征。
- **keep**：
  - `action_prev`：整段 `keep_full`，输出校验 checksum/token 数不变。
  - `task`：关注度 = 提供的 `importance` 或与 task 向量余弦；`score >= high_thr` 保留，其余进 merge 池。
- **merge**：history（含 task 低关注子段）段均值特征 → 纯 numpy k-means → 每类加权中心代表（代表 token 数≈成员均值）。
- **NPU DMA 预热**：`npu_dma_warmup_move()` **stub**（TODO：Hexagon DMA / fastrpc）。
- **未做**：重编译 llama.cpp、下载大模型、改手机设置/CPU 频率。

## 复现
```bash
cd ~/code/exp/2026-11-a2o-gap-profiling/EXP-008-kv-keep-a-merge
conda run -n visiontrim-qwen25vl python scripts/test_kv_keep_merge.py
conda run -n visiontrim-qwen25vl python scripts/kv_keep_merge.py --out-dir results
```

## 结果（host，Asia/Shanghai 2026-11-03 10:57）
见 `results/summary.md`、`kv_keep_merge_summary.json`、`dispositions.csv`、`timings.csv`、`a2o_comparison.csv`；更大场景 `results/gui5_like/`。

| 场景 | tokens in→out | MiB in→out | 压缩比 | 策略中位 ms | +host memcpy ms | action_prev 完好 |
|---|---:|---:|---:|---:|---:|---|
| 默认（6 hist×80） | 560→284 | 19.69→9.98 | 0.507 | 0.832 | ≈1.059 | 是 |
| gui5_like（8×120） | 1064→344 | 37.41→12.09 | 0.323 | 1.070 | ≈1.281 | 是 |

对照 EXP-005 A2O 最短预算 UI-TARS≈1345 ms：**均落入空档**。测试 `scripts/test_kv_keep_merge.py` 全过。

手机：adb 在线但 loadavg≈12–14，本轮**未**推送二进制/短测 memcpy（避免高负载干扰）；未改设置/频率。

## 下一步
1. 接入真实 llama.cpp / agent KV dump（替换合成特征）。
2. 实现 NPU DMA 预热搬移（替换 stub）。
3. 与截图/UI 采集并发时的调度干扰实测。
