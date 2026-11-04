# EXP-006 Hexagon smoke

Xiaomi 15 (SM8750 / Hexagon v79) 上 llama.cpp Hexagon 包冒烟。

| 子项 | 结果 | 记录 |
|---|---|---|
| 文本 HTP0 | ok | `results/smoke_20261103.md` |
| 视觉 mmproj + HTP0 | ok（ViT=OpenCL，LLM=HTP0） | `results/vision_htp0_20261103.md` + `.log` |

运行时/包引用：`shared/runtime/llama.cpp-hexagon-pkg`（勿在本目录复制大文件）。截图源：`EXP-003-phone-server/inputs/shot0.png`。
