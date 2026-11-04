# 运行

当前论文主线为 Tool Slack 投机准备与结果返回后延迟绑定；四路径用于手机成本校准，命令与验收见[ROADMAP.md](ROADMAP.md)。远端 BFCL 队列见 [BFCL.md](BFCL.md)。`--suite multiturn`保留为此前固定CPU Decode的回归测试，见本文末尾及[MULTITURN.md](MULTITURN.md)。以下单轮命令保留用于复现历史结果；QNN/NPU ViT 已搁置。

## Prefill 优势区间

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/profile_prefill.py
conda run -n visiontrim-qwen25vl python scripts/analyze_prefill.py results/NAME-prefill-grid
conda run -n visiontrim-qwen25vl python scripts/plot_prefill.py results/NAME-prefill-grid
```

矩阵与计时边界见 [ROADMAP.md](ROADMAP.md)。分析和绘图不访问手机。设备测试必须与其它模型实验串行。可用 `--histories 8192 --inputs 64 --repetitions 1 --direct-check` 对比原始 CPU 上下文直接继续与 snapshot 恢复后的结果；该诊断仍检查输出，不以耗时正常或退出码为成功依据。

## 环境

- Host：`conda run -n visiontrim-qwen25vl python ...`，Android NDK 已安装于 `~/Library/Android/sdk/ndk/`。
- 运行时包：`../shared/runtime/llama.cpp-hexagon-pkg`，使用包内头文件与动态库构建，避免本地源码版本与二进制 ABI 不一致。
- 手机：`/data/local/tmp/llama.cpp/lib`；模型 `/data/local/tmp/models/Qwen2.5-VL-3B-Instruct-Q4_0.gguf`；mmproj 同目录 `mmproj-model-f16.gguf`。
- `adb devices -l` 查当前设备；所有命令可用 `--device SERIAL` 覆盖默认无线地址。
- 默认地址读取 `configs/device.json`；`DEVICE` 可覆盖。当前为 `10.29.11.213:46533`。

## 默认基线

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_agent.py \
  --config configs/cpu-none.json --scenario scenarios/ready.json --tag cpu-ready

conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_agent.py \
  --config configs/npu-none.json --scenario scenarios/screenshot.json --tag npu-screen
```

runner 自动构建 `build/agent_runtime`，推送到本次独立临时目录，执行后取回结果并清理。使用同一个 C++ 文件构建全部条件，不修改共享运行时。

## 分组验证

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_agent.py --suite diagnosis
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_agent.py --suite matrix
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_agent.py --suite vision
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_agent.py --suite vision-history
conda run -n visiontrim-qwen25vl python -m unittest discover -s tests
```

纯本地汇总已有结果（不会访问手机）：

```bash
conda run -n visiontrim-qwen25vl python scripts/analyze_results.py
conda run -n visiontrim-qwen25vl python scripts/analyze_vision.py
conda run -n visiontrim-qwen25vl python scripts/plot_results.py
```

图表导出到 `results/figures/`（PNG、PDF、数据与来源清单），只读取已有结果，不访问手机。绘图使用本机 `figure-style` 技能样式与 Arial Unicode 字体。

`diagnosis`：原始上下文套模板后的响应、CPU READY、CPU/HTP 同后端 seq roundtrip 后历史记忆。

`matrix`：同一历史场景下的八个有效消融条件。预搬移不适用于没有后端交接的纯 CPU / HTP 条件。

`vision`：同一张已保存真机截图的 CPU / HTP / hybrid 响应，检查 JSON action；场景本身没有历史，适合验证图像和回传，不用于衡量历史预搬移收益。

`vision-history`：同一真机截图加文本历史，四个混合条件（有无整理 × 有无预搬移），统一 2000ms 模拟 A2O、batch128、CPU mmap、OpenCL 视觉 FA 关闭。检查页面内容、动作字段及跨后端状态一致性。

原始无模板问题可通过以下命令重现，预期响应校验失败（退出码 2），失败结果仍会保存：

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_agent.py \
  --config configs/diagnostic-raw.json --scenario scenarios/original-context.json --tag raw-original
```

## 结果

每次结果目录包含：`manifest.json`（完整配置、场景、设备和源码/二进制 hash）、`formatted_prompt.txt`、`runtime.log`、`runtime.json` 与经过 host 响应校验的 `result.json`。进程失败仍保留 manifest 和日志，不自动切换后端。

混合机制配置已设 `a2o_ms: 2000`。这是受控等待，不代表真实动作/UI 稳定性测量。真实动作循环接入前，用此矩阵验证状态和响应。`hybrid-mmap.json` 保留零等待的内存诊断条件。

## NPU–CPU 协同检查

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/run_agent.py \
  --config configs/hybrid-htp-probe.json --scenario scenarios/screenshot-history.json --tag hybrid-htp-probe
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_agent.py --suite npu-cpu
conda run -n visiontrim-qwen25vl python scripts/analyze_npu_cpu.py
```

第一条开启视觉节点诊断，仅用于检查分配。第二条顺序运行 CPU/OpenCL/HTP 视觉与 HTP 提前准备对照，均保持 HTP LLM prefill＋CPU decode。第三条离线汇总到 `results/npu-cpu-summary.json`，保留未完成/失败条件，不自动补跑。HTP 视觉目前应视为算子覆盖待验证路径，不能仅凭 CLIP 后端名称视为完整 NPU ViT。

## 当前多轮基线

```bash
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/validate_agent.py --suite multiturn
conda run -n visiontrim-qwen25vl python scripts/analyze_multiturn.py
```

主线见 MULTITURN.md。GPU ViT 固定，CPU→HTP→CPU→HTP 选择 prefill、CPU decode，同一进程四轮真实历史连续积累；关闭 KV 整理、预搬移及模拟等待。成对 prefill 是数值诊断，不作正式整轮性能对照。`configs/multiturn-cpu.json` 为同驻留条件下始终使用 CPU prefill 的四轮对照。
