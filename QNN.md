# QNN 视觉编码接入

**已搁置。** 当前按用户新范围使用 GPU ViT，先验证多轮 CPU/HTP prefill 状态链路；不继续本页的编译/设备测试。主线见 [MULTITURN.md](MULTITURN.md)。以下保留已有产物和手动入口。

参考 Qualcomm 官方实现，复用固定形状视觉图、Conv3d→Conv2d、split Q/K/V 与 Linear→Conv2d 适配。代码来源和 BSD 许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。保持本机/手机原有 Qwen2.5-VL-3B，输出宽度为 2048；不使用官方 7B 的 3584 维输出替代。

## 当前入口

远端复用已有 CUDA / PyTorch Docker 基础环境。模型放共享 `models/`，QAIRT SDK 放共享 `tools/qairt/`，工程放 `projects/a2o-gap-profiling/`，产物放 `outputs/a2o-gap-profiling/`。SDK 和模型通过 `/workspace` 挂载，不复制进镜像。环境定义为 `configs/qnn.Dockerfile`；当前适配使用官方要求的 Transformers 4.56.2。

导出并对照原始视觉模型（CUDA）：

```bash
python scripts/export_qnn_vision.py \
  --checkpoint /workspace/models/Qwen2.5-VL-3B-Instruct \
  --image inputs/phone_screenshots/shot0.png \
  --output /workspace/outputs/a2o-gap-profiling/NAME-export \
  --height 980 --width 448 --device cuda
```

只加载 safetensors 中的 visual 权重，不构建 LLM。图片尺寸为 448×980、grid `[1,70,32]`，输出 `[560,2048]`。输出已做一次逆 window 排序，接入端不能再逆排一次。导出 FP32 ONNX，由 QAIRT 转换为浮点16位 HTP 图。直接将整个 PyTorch 适配模型转半精度的尝试未通过对照，原始结果保留，当前入口不采用该方法。

转换和编译（CPU 编译工具，目标 HTP）：

```bash
python scripts/compile_qnn_vision.py \
  --sdk /workspace/tools/qairt/2.51.1.261002 \
  --input /workspace/outputs/a2o-gap-profiling/NAME-export \
  --output /workspace/outputs/a2o-gap-profiling/NAME-compiled --opt-level 1
```

使用 QAIRT 的 ONNX→DLC→HTP context 路线，目标为 SM8750（soc_model=69）、v79，不需要生成/编译大型 C++ 模型库。每阶段命令与日志留在产物目录。默认优化级别 1 用于功能验证；更高优化级别留到机器空闲后再编译。`--dlc /PATH/vision_encoder.dlc` 可复用已有转换产物。

手机视觉单测：

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_qnn_vision.py \
  --context /ABS/PATH/vision_encoder.bin --inputs results/qnn-vision-reference
```

Android QAIRT 运行库统一放 `../shared/runtime/qairt-2.51.1/android/`，编译模型放共享模型目录。手机工具使用 `libQnnHtp.so`，不加载 QNN CPU 后端。该单测使用 CUDA host 按真实截图生成的预处理输入，记录 QNN 输出与适配模型的差异；尚不能代表包含手机图像预处理的完整时延。

## 与现有 Agent 的接口

`vision_embeddings` 可指定自然空间顺序、行优先的 float32 原始文件。native 通过原 mtmd image chunk 的 M-RoPE 信息将其交给 HTP LLM prefill，之后继续原有 KV 回传和 CPU decode。该诊断模式在结果中明确记录 `vision_provider=embedding_replay`；不会将离线编码时间算作 NPU 加速。

`configs/hybrid-embedding-replay.json` 验证已导出的 3B 适配 embedding。QNN 真机编码完成后可用其输出替换该文件路径，验证端到端响应。当前 replay 仍初始化 mmproj 以取得 mtmd chunk/position 元数据；连续服务接入时应以固定视觉输入元数据及 QNN 常驻上下文替代。

## A2O 中可提前完成的工作

固定形状下，QNN context 装载、权重准备、图 IO 缓冲区、RoPE、window/full masks 均不依赖下一截图像素。未来将它们与历史 KV 上传放入 A2O；像素预处理和 ViT 执行等截图到达后进行。目前先验证 QNN 视觉编码及其与现有 Agent 的接口，未把独立 CLI 单测称为已实现同进程常驻 QNN 调度。

## 本轮状态与后续继续

远端环境已保存为 Docker 镜像 `a2o-qnn:20261103`，容器 `a2o-qnn-export` 已停止。FP32 导出目录为 `/workspace/outputs/a2o-gap-profiling/20261103-qnn-vision-fp32`；已转换的浮点16位 DLC 为 `/workspace/outputs/a2o-gap-profiling/20261103-qnn-vision-dlc/vision_encoder.dlc`。默认优化和级别1编译均未完成，按用户安排留到空闲时继续，原始日志保留。当前没有可部署的 HTP context，也尚未执行手机 QNN 编码。

空闲时直接复用 DLC：

```bash
docker start a2o-qnn-export
docker exec a2o-qnn-export python scripts/compile_qnn_vision.py \
  --sdk /workspace/tools/qairt/2.51.1.261002 \
  --input /workspace/outputs/a2o-gap-profiling/20261103-qnn-vision-fp32 \
  --dlc /workspace/outputs/a2o-gap-profiling/20261103-qnn-vision-dlc/vision_encoder.dlc \
  --output /workspace/outputs/a2o-gap-profiling/NAME-context --opt-level 3
```
