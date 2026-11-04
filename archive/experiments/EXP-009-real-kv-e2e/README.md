# EXP-009 真实 llama KV 端到端（dump → 策略 → 写回 → HTP0 prefill）

## 目标
把 EXP-008 的 keep(A)+任务关注度+history k-means merge 接到真实 llama KV：
手机 HTP0 上 dump → host 策略 → `llama_memory_seq_rm` 写回 → 压缩态再做下一次 HTP0 prefill。

## 状态
后续核查：本实验没有 CPU/NPU 往返，prefill/decode 均在 HTP0；7.9ms 不包含完整解析、mask 构造、导出/传输/加载。EXP-010 的 keep-all 对照同样立即 EOG，因此不能由原生成结果归因于压缩或稀疏 position；也不应未经验证直接 renumber M-RoPE。完整路线与证据见上级 `PIPELINE_REVIEW.md`。

**已跑通端到端（2026-11-03）**。数字见 `results/e2e_summary.json`（文件名：`e2e_summary.json`）。
- action_prev 段 intact：策略断言 + 压缩 seq 再检查均通过
- NPU DMA / Hexagon layout：**仍为 stub**（接口保留，结果标注）
- merge 写回：**exemplar-keep**（保留簇代表段 token，不重写质心 K/V）

## 设备 / 资产
- 手机：Xiaomi 15（SM8750），无线 adb；后端 **HTP0**（Hexagon v79）
- 模型：`/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf`
- 运行时：`shared/runtime/llama.cpp-hexagon-pkg` → 手机 `/data/local/tmp/llama.cpp`
- Host Python：conda `visiontrim-qwen25vl`
- 未改 CPU 频率/系统设置/用户数据；测完清理 `/data/local/tmp/exp009_kv` 与临时 bin

## 管道
1. **dump**（手机）：`tools/kv_gui_e2e_min.cpp` 多段可控文本模拟 task / history_0..5 / action_prev → `llama_state_save_file` + `llama_state_seq_get_data`
2. **parse**（Mac）：`scripts/parse_kv_seq.py` 解析 magic(`0xaf143cd8`)+M-RoPE `cell_ext`，取层均值 K→投影 64 维，按段切 `npz`
3. **policy**（Mac）：复用 `EXP-008/.../kv_keep_merge.py` → `scripts/policy_to_keep_mask.py` 产出 `keep_mask.txt` + action_prev 断言
4. **apply**（手机）：加载 state → 按 mask 区间 `llama_memory_seq_rm` → 存 `kv_state_compressed.bin` / seq
5. **prefill**（手机）：加载压缩 state → 追加 next prompt → greedy 若干 token

复现：
```bash
export DEVICE=10.29.11.213:42629   # 以 adb devices 为准
./scripts/run_e2e.sh all           # 或分阶段 dump|policy|apply|prefill|cleanup
```

## 关键实测（勿编造；摘自 results）
| 项 | 值 |
|---|---|
| dump n_tokens / seq_bytes | 171 / 6,308,052（≈36,864 B/tok） |
| dump 历史 prefill | 4228.32 ms（HTP0，含整段历史） |
| keep / drop | 115 / 56（token 压缩比 ≈0.673） |
| 压缩 seq_bytes | 4,242,548 |
| action_prev intact | true（32/32 仍在压缩 seq） |
| host policy_total | ≈7.87 ms |
| apply seq_rm | ≈0.027 ms（不含模型加载） |
| next prefill（20 tok） | 468.30 ms（HTP0） |
| greedy 输出 | `<|im_end|>`（见局限） |
| NPU DMA | stub |

与 A2O：KV prep（host policy + seq_rm ≈ 7.9 ms）远低于所列 A2O 预算（约 1.3–12.7 s）。**A2O 是 wait+截图预算，不要和 NPU 推理时间比。** 模型加载 / 全量 dump 不计为 prep 预算本体。

## 局限 / 阻塞点（诚实）
1. 历史是**可控多轮文本段**，不是真多图 GUI dump（工具可扩 mmproj，本轮未接图像）。
2. `seq_rm` 后 **position 稀疏**（pos_max 仍 170，有洞）；未做 compact/renumber，M-RoPE 连续性有风险。
3. 压缩态 greedy 立刻 `<|im_end|>`：只证明 prefill/decode 路径可跑，**不证明输出质量**；未套 chat template。
4. merge 无法在不发明张量的情况下写「质心 KV」→ exemplar-keep。
5. Hexagon DMA/layout 搬移未实现（stub）。
6. 完整 session 写回依赖 `llama_state_*`；ON_DEVICE 路径未用。

## 建议下一步
- seq_rm 后 **compact positions / 重写 M-RoPE ext**，或只 drop 尾部再验证生成质量
- 接真实多图 GUI 轨迹 + mmproj
- 实现 Hexagon DMA warmup（替换 stub）
- 对照：同 prompt 在未压缩 KV 上的 greedy 输出

## 目录
- `tools/kv_gui_e2e_min.cpp`、`tools/build/kv_gui_e2e`（android aarch64）
- `scripts/{parse_kv_seq,policy_to_keep_mask,run_e2e}.sh/py`
- `results/`：dump/policy/apply/prefill 日志与 `e2e_summary.json`
