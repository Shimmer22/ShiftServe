# 主实验平台迁移验证（2026-11-05）

主设备已切换为小米 23116PN5BC（SM8650，Android 14，设备序列号
`207d26dd`），默认无线连接为 `10.29.131.50:42369`。旧设备实验结果和
manifest 保留原样。配对凭据不入库。

## 部署

- 复用 `../shared/runtime/llama.cpp-hexagon-pkg`，未重新构建共享运行时。
- 部署 Qwen2.5-VL-3B-Instruct Q4_0 和 F16 mmproj，未复制到工程目录。
- 手机端 16 个文件 SHA256 全部一致，见
  [部署清单](../../results/20261105-160303-platform-deployment/deployment.json)。
- 首次连接端口 46089 中途离线，更换为用户提供的 42369 后续传；保留
  [首次失败记录](../../results/20261105-160110-platform-migration/deployment.json)。

## 文本与状态

所有条件关闭 KV 整理、预搬移和 CPU 权重 repack，batch128、context2048，
HTP queue/opbatch 均为 1；不执行模型动作，不改变手机设置。文本验证期间
视觉投影仍在传输，因此这些时间数据不能用于平台性能比较。

| 条件 | 状态检查 | 响应检查 |
|---|---|---|
| CPU READY | 无历史迁移 | `READY`，通过 |
| CPU 历史，同后端往返 | 完整 sequence 字节一致 | `WIFII`，失败 |
| CPU 历史，不做往返（诊断对照） | 不执行迁移 | `WIFII`，失败 |
| HTP 历史，同后端往返（独立诊断） | 完整 sequence 字节一致 | `WIFII`，失败 |
| hybrid 历史（独立诊断） | CPU→HTP 与 HTP→CPU 字节一致 | `WIFII`，失败 |

原历史场景要求 `WIFI` 或 `WI-FI`，未修改验收标准。主序列在 CPU 历史
响应失败后停止，随后分别运行直接 CPU 对照及 HTP/hybrid 独立诊断。
结果说明当前没有观察到迁移引入的回答变化，但不能据此宣称历史记忆验收通过。

证据：[主序列](../../results/20261105-161300-platform-validation/summary.json)、
[HTP/hybrid 诊断](../../results/20261105-161300-platform-validation/backend-diagnostics.json)、
[直接 CPU 对照](../../results/20261105-161452-280635-migration-cpu-history-direct/result.json)。

## 图像

独立执行 `scripts/validate/platform.py --stage vision`，使用保存的真机截图
`inputs/phone_screenshots/shot0.png`，固定 CPU ViT，并分别显式指定 CPU、HTP、
hybrid LLM。三种路径均通过实际图像与严格 JSON 检查，输出为
`{"screen":"Developer Options","action":"wait"}`；hybrid 的跨后端
状态字节核验也通过。未使用外部视觉 embedding 回放。

证据：[图像序列](../../results/20261105-161851-platform-validation/summary.json)、
[迁移总记录](../../results/20261105-160110-platform-migration/summary.json)。

## 复现与后续

```bash
python3 scripts/run/deploy_platform.py
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate/platform.py
# 需要分别检查文本或图像时，使用 --stage text / --stage vision
```

迁移整体尚未通过：先定位历史响应失败，再验证常驻 Tool 跨轮记忆。
基础检查全部通过后，在新平台重采四路径成本与 Oracle。旧平台 Router
标定仅可作为机制夹具，不能作为 SM8650 的性能预测依据。
