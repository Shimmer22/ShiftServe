# 主线关键实验计划与记录表

> 封版构建：`results/20261119-splitlm-main-merge-build`（与 `03ca4e4` 源码 SHA256 完全匹配）。  
> 运行原则：手机端实验串行独占，**绝不与主机以外的手机脚本争抢设备**；主机侧实验仅在 `wins` 执行，与手机端互不干扰可并行。

---

## A. `main-4b-ablation`（4B 多 Benchmark 封版主表 — P0）

### 1. 实验目标与安排
- **第一阶段（立即可跑，`~2.5–3 h`）**：使用封版构建重跑 `20261118-multibench-frozen-ablation` 同一批 **19 条轨迹 / 64 个 warm 轮次** 的 `shared_prior` 条件（默认开启 `kv_layer_stride=4`、`kv_min_evict_tokens=256`），用干净无 Swap、低采集税的数据替换旧表中的 `1,353.3 s`。
  - 数据复用：直接复用 `results/20261118-multibench-frozen-ablation/` 的 `fixtures.json`、`renderer-audit.json`、`route-costs.json` 到 `results/mainline-eval/main-4b-ablation/`，确保与已有 `full_cpu / shared_cc / shared_m1 / shared_recent` 完全同轨同标定。
- **第二阶段（扩展强基线，需先在 `frozen_ablation.py` 补充对应 arm）**：
  - `static_hc`（SKIM-style 对照：无 Gap 异步准备、固定 HTP Prefill + CPU Decode、不裁剪 KV）；
  - `shared_prior + cold_route`（加载 H=0 冷选路模型，补充“含首轮推理合计”同样胜出的证据）。

### 2. 启动命令（第一阶段 `shared_prior` 重跑）
```bash
mkdir -p results/mainline-eval/main-4b-ablation
cp results/20261118-multibench-frozen-ablation/{fixtures,renderer-audit,route-costs}.json results/mainline-eval/main-4b-ablation/
python3 scripts/run/frozen_ablation.py \
  --fixtures results/mainline-eval/main-4b-ablation/fixtures.json \
  --build results/20261119-splitlm-main-merge-build \
  --cpu-registry configs/model-qwen3-4b.json \
  --model configs/model-qwen3-4b.json \
  --output results/mainline-eval/main-4b-ablation \
  --conditions shared_prior \
  --gap-ms 1000
```

> 注：`scripts/run/frozen_ablation.py` 当前内置条件仅含 `full_cpu`、`shared_cc`、`shared_m1`、`shared_recent`、`shared_prior`；扩展第二阶段 `static_hc` / `cold_route` 时再为其补入条件分支并单列说明。

### 3. 预留记录表

#### 表 A-1：19 条轨迹 / 64 Warm 轮次总览对比（`gap_ms = 1000`）
| 条件 | 来源 / 状态 | 工具返回后合计 | 相对 `full_cpu` | Prefill | Decode (`ms/t`) | Handoff | Wait | Restore | 峰值 Swap (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_cpu` | 已有 (`20261118-multibench`) | 1,514.1 s | 基线 | 783.3 s | 661.6 s (216.1) | 0.0 s | 0.0 s | 43.6 s | 0.0 |
| `shared_cc` | 已有 (`20261118-multibench`) | 1,395.8 s | −7.8% | 746.1 s | 581.9 s (190.1) | 0.0 s | 0.0 s | 41.8 s | 0.0 |
| `shared_m1` | 已有 (`20261118-multibench`) | 1,264.8 s | −16.5% | 544.7 s | 614.3 s (200.7) | 63.0 s | 13.5 s | 4.3 s | 0.0 |
| `shared_recent` | 已有 (`20261118-multibench`) | 1,175.3 s | −22.4% | 513.3 s | 569.2 s (185.9) | 59.3 s | 6.9 s | 3.8 s | 0.0 |
| **`shared_prior` (封版)** | *待重跑* | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `static_hc` *(扩展)* | *待脚本支持后补跑* | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

#### 表 A-2：`shared_prior`（封版）分 Benchmark 拆解
| Benchmark | 轨迹 / Warm 轮 | `shared_prior` Post-Return | 相比 `full_cpu` | 相比 `shared_m1` | 相比 `shared_recent` | 触发 `prefill_prior` 轮次 |
|---|---:|---:|---:|---:|---:|---:|
| `tau2-airline` | 5 / 10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `tau2-retail` | 3 / 6 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `tau2-telecom` | 6 / 39 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `toolsandbox` | 5 / 9 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## B. `host-accuracy-align`（主机准确率收尾与稀疏层对齐 — P1）

### 1. 实验目标与安排（仅在 `wins` 执行，与手机并行不冲突）
1. **步骤 B-1（收尾）**：等待当前后台运行中的 `host_kv_accuracy_8b_pipeline.py`（`results/20261118-host-kv-accuracy/tau2-ratio-8b`）自然跑完，补齐 8B τ²-bench 比例裁剪 5 条件准确率。**在此期间勿覆盖或重建 `wins` 上的 `phone_service_host` 二进制。**
2. **步骤 B-2（口径对齐）**：待 B-1 结束后，在 `wins` 用 `03ca4e4` 主干源码重建 CUDA 版 `phone_service_host`（携带 `A2O_KV_LAYER_STRIDE=4`、`A2O_KV_MIN_EVICT_TOKENS=256`），在 **BFCL 4B**（`configs/host-kv-accuracy-bfcl.json`，输出到 `results/mainline-eval/host-accuracy-align/bfcl-stride4/`）复验 `m2_prior_mid` 与 `m2_prior_high`，证明 **36 层采 9 层（`stride=4`）不损害任务准确率**。

### 2. 预留记录表

#### 表 B-1：`tau2-ratio-8b` 完整结果（Qwen3-8B, 36 题）
| 条件 | 比例 | 完成 / 总题 | Reward 合计 | `reward_mean` | Retail / Airline / Telecom | 状态 |
|---|---:|---:|---:|---:|---|---|
| `full_kv` | −1.0 | 36 / 36 | 4.0 | 0.1111 | _待提取分域_ | 已完成 |
| `recent_only` | 0.50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 运行/排队中 |
| `m2_prior_low` | 0.25 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 运行中 |
| `m2_prior_mid` | 0.50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 排队中 |
| `m2_prior_high` | 0.75 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 排队中 |

#### 表 B-2：BFCL 4B 稀疏层采样口径对齐（`stride=1` vs `stride=4`）
| 条件 | `kv_layer_stride` | `kv_min_evict_tokens` | Complete | Valid | Accuracy | 相对 `full_kv` / `recent_only` |
|---|---:|---:|---:|---:|---:|---|
| `full_kv` *(已有基线)* | — | — | 17 / 60 | 6 | 0.1000 | 基线 |
| `recent_only` (`0.50`, 已有) | — | — | 13 / 60 | 3 | 0.0500 | −0.0500 |
| `m2_prior_mid` (`0.50`, 已有) | 1 | 0 | 15 / 60 | 4 | 0.0667 | +0.0167 vs `recent` |
| `m2_prior_high` (`0.75`, 已有) | 1 | 0 | 17 / 60 | 5 | 0.0833 | +0.0333 vs `recent` |
| **`m2_prior_mid` (对齐复验)** | **4** | **256** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **`m2_prior_high` (对齐复验)** | **4** | **256** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## C. `gap-sensitivity`（Tool Wait Gap 敏感性扫描 — P1）

### 1. 实验目标与安排
- **目的**：直观展示 M1 利用 Tool Wait Gap 隐藏准备开销 $M(H)$ 的收益曲线，以及在不同 `P/H` 形态下动态路由对 `Static HC` / `Full CPU` 的优势。
- **样本选择**（从 `fixtures.json` 抽取 3 条典型轨迹）：
  1. 短历史小 `P`（ToolSandbox）：`send_message_with_contact_content_cellular_off`
  2. 长历史小 `P`（τ²-airline）：`airline-c6f3ac57944a-9`
  3. 长历史大 `P`（τ²-retail）：`retail-1da51b8d8ff9-26`
- **扫描变量**：`--gap-ms` ∈ `{0, 250, 500, 1000, 2000}`（条件：`full_cpu`、`shared_m1`、`shared_prior`；以及扩展后的 `static_hc`）。输出目录：`results/mainline-eval/gap-sensitivity/`。

### 2. 预留记录表（Warm Post-Return 耗时，单位：秒）
| 轨迹类型 | `gap_ms` | `full_cpu` | `static_hc` *(可选)* | `shared_m1` | `shared_prior` (封版) | `shared_prior` 平均 `wait_ms` |
|---|---:|---:|---:|---:|---:|---:|
| 短历史 (`send_message`) | 0 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 短历史 (`send_message`) | 250 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 短历史 (`send_message`) | 500 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 短历史 (`send_message`) | 1000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 短历史 (`send_message`) | 2000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 长历史小 `P` (`airline-9`) | 0 / 250 / 500 / 1000 / 2000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 长历史大 `P` (`retail-26`) | 0 / 250 / 500 / 1000 / 2000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## D. `main-8b-scaling`（Qwen3-8B 长上下文扩展验证 — P1）

### 1. 实验目标与安排
- **目的**：按照项目默认文本模型要求（`configs/model-default.json`, Qwen3-8B Q4\_0），在 12GB RAM（实际可用 ~15GB MemTotal 手机）上用 `4–6` 条长短混合代表轨迹验证：**共享权重彻底消除双权重 Swap 崩溃，且封版 M1+M2 消除此前 `8b-joint-retry` 中的采集税倒挂，稳定取得全局最低延迟与最低 RSS**。
- **对比条件**：`full_cpu` vs `shared_cc` vs `shared_m1` vs `shared_prior`（封版构建 `20261119-splitlm-main-merge-build`）。输出目录：`results/mainline-eval/main-8b-scaling/`。

### 2. 预留记录表
| 条件 | 轨迹数 / Warm 轮 | 工具返回后合计 | 相对 `full_cpu` | Prefill | Decode (`ms/t`) | Handoff | 峰值 RSS / Swap (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_cpu` | _TBD_ | _TBD_ | 基线 | _TBD_ | _TBD_ | 0.0 | _TBD_ |
| `shared_cc` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0.0 | _TBD_ |
| `shared_m1` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **`shared_prior` (封版)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
