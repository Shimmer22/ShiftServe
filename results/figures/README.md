# 设备测试图表

- `01_latency_breakdown`：最终四条件的观察后耗时组成；单次测量，包含视觉模型初始化，不包含 LLM 初始化和观察前 A2O。
- `02_kv_transfer_volume`：压缩前后的上行/下行完整序列化状态体积，MiB；不同于实际 buffer 分配量。
- `03_encoder_lifetime`：保留/释放视觉编码器的单次诊断对照，CPU decode 耗时与进程 major fault 增量；相同截图、历史和输出，负载与缓存未控制。

各图同时提供 PNG 与 PDF。`sources.json` 保存运行来源和排版检查，`plot_data.json` 保存实际绘图数据。未完成和主动终止的运行不进入统计。

复现：在项目根目录执行 `conda run -n visiontrim-qwen25vl python scripts/plot_results.py`。
