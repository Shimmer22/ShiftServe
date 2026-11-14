# Qwen3-1.7B Prefill 先验直接指导 CPU KV Reduce 验证

验证方向从“双模型/稀疏 NPU 投机解码 + CPU 验证”收敛为**直接利用 Prefill 阶段观察到的注意力先验（`kq_soft_max`）指导 CPU KV 裁剪与紧凑重排（KV Reduce）**，彻底消除投机验证税与双模型驻留冲突，并直接削减 `HC`（HTP Prefill + CPU Decode）状态迁移体量与 CPU Decode 扫 KV 开销。

## 1. CPU Decode 对 KV 长度的敏感度（`Qwen3-1.7B-Q4_0`）

配置文件：`configs/qwen3-1_7b-kv-verify-gate.json`，结果目录：`results/20261114-qwen3-1_7b-kv-verify-gate/`（`D=16`，`CPU_REPACK`）。

| 历史长度 `H` | `state_bytes` | CPU Decode 中位延迟 (`ms/token`) | 相对 `H=512` 变化 | 检查点位一致 |
| --- | --- | --- | --- | --- |
| `256` | `28.00 MB` | `34.19` | `-1.3%` | 通过 |
| `512` | `56.01 MB` | `34.63` | `1.00x` | 通过 |
| `4096` | `448.05 MB` | `66.30` | `+91.5%` | 通过 |
| `8192` | `896.09 MB` | `95.51` | `+175.8%` | 通过 |

长历史（`H=8192`）下 KV 扫描占据 `1.7B` CPU decode 约 `64%` 耗时；同单机对比中，即使在 `HTP` 开启 `kv_offload` 与 `flash_attention`（`results/20261114-qwen3-1_7b-decode-diag/`），`H=4096` 单模型 `HTP` decode 仍为 `95.27 ms/token`，明显慢于 CPU decode。

## 2. 机制实现：先验采集 + `seq_rm` + 紧凑重排

- 先验采集（[prefill_prior.h](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/src/a2o/native/profiling/prefill_prior.h)）：在 `kv_offload=false, flash_attention=false` 配置下，`kq_soft_max-<layer>` 在 CPU 上以 F32 形式产出；对本轮 prefill 所有头、层及 query 位置的 softmax 权重求和得到各历史位置的重要性分数，不做额外设备间大矩阵回拷。
- 紧凑重排的必要性（[llama-kv-cache.cpp](file:///Users/hui/code/exp/shared/runtime/llama.cpp/src/llama-kv-cache.cpp#L1129-L1143)）：`llama_kv_cache::get_n_kv` 计算依赖 `GGML_PAD(cells.used_max_p1(), max(n_pad, 256))`。若仅对中间 token 调用 `seq_rm` 留下空洞，尾部 recent window 仍在高位 cell，实际 attention 扫描宽度 `n_kv` 不会缩小。通过 [kv_reduce_probe.h](file:///Users/hui/code/exp/2026-11-a2o-gap-profiling/src/a2o/native/profiling/kv_reduce_probe.h) 保留 `sink + pool(prefill_prior top-K) + recent`，在 `seq_rm` 后经 `state_seq_get_data -> memory_clear -> state_seq_set_data` 将存活槽位紧缩至 `0 .. kept-1`（同时完整保留原始 RoPE `pos`），可天然复用现有 `HC` 迁移路径同时缩减状态传输字节与 CPU `decode_n_kv`。

## 3. 端到端 `decode_n_kv`、算子加速与长程记忆验证

配置文件：`configs/qwen3-1_7b-kv-reduce-verify.json`，结果目录：`results/20261114-qwen3-1_7b-kv-reduce-verify/`（`H=4096`，`kv_pool_window=5`，`kv_sink_tokens=4`，`kv_recent_tokens=64`）。

### 3.1 `decode_n_kv` 实测收缩与 `COMPUTE` 提速

| 条件 | `kept / total` | 实测 `decode_n_kv` | `state_bytes` | `evict + compact` | `CPU Decode` 中位 (`ms/tok`) | CPU 频点 `(cpu4, cpu7)` |
| --- | --- | --- | --- | --- | --- | --- |
| `attr-full-h4096` | `4136 / 4136` | `4352` | `452.42 MB` | `-` | `48.43` | `(2438400, 2035200)` |
| `attr-prior-496` | `496 / 4136` | `512` (`8.5x↓`) | `54.26 MB` (`8.3x↓`) | `0.91 + 70.41 ms` | `32.76` (**`1.48x` 加速**) | `(2323200, 1939200)` |
| `attr-prior-240` | `240 / 4136` | `256` (`17.0x↓`) | `26.25 MB` (`17.2x↓`) | `1.07 + 58.29 ms` | `48.96`（降频态） | `(2131200, 902400)` |

`attr-prior-496` 相比 `attr-full-h4096` 的 `D=16` 总 `COMPUTE` 耗时从 `772.65 ms` 降至 `522.21 ms`（`-32.4%`），紧凑后 `max_pos_preserved = 4135` 无损保留。

### 3.2 纯日志长程大海捞针 JSON 召回（`pos=1024 / 4146`）

| 条件 (`P=50`) | Prefill 后端 | Needle 保留率 | 解码中位 (`ms/tok`) | 状态大小 | 纯 JSON 召回结果 |
| --- | --- | --- | --- | --- | --- |
| `needle-full-h4096` | `CPU` | `20/20` | `78.39` `(2131200, 902400)` | `453.52 MB` | **通过**：`{"code":749281,"codename":"ORION"}` |
| `needle-recent-496` | `CPU` | **`0/20 (0.0%)`** | `50.20` `(2131200, 1478400)` | `54.26 MB` | **失败**：`{"code":123,"codename":"Sample"}` |
| `needle-cpu-prior-496` | `CPU` | **`20/20 (100.0%)`** | `52.45` `(2131200, 902400)` (**同频 `1.49x` 加速**) | `54.26 MB` | **通过**：`{"code":749281,"codename":"ORION"}` |
| `needle-htp-prior-496` | `HTP` | **`20/20 (100.0%)`** | `190.49` *(HTP 自解码)* | `54.26 MB` | **通过**：`{"code":749281,"codename":"ORION"}` |

结论：**Prefill 注意力先验（无论是 CPU 还是 HTP Prefill 执行）能在相同 `496` token / `54.26 MB` KV 预算下 `100%` 捕获并保留 `pos=1024` 的关键历史事实，完美通过精确 JSON 记忆校验；同频点下实际 CPU decode `n_kv` 由 `4352` 降至 `512`，获得 `1.48x–1.49x` 解码提速，且状态体积缩减 `8.3x`。** 后续可直接将同一机制迁移至 `Qwen3-8B` 与 `τ²-bench` `HC` 多轮场景。
