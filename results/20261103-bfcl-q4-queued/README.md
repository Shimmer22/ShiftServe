# 已排队：BFCL Qwen2.5-VL-3B Q4_0

远端持久队列 PID `1850472`。此目录为安排时的本地快照，实时状态在远端：

`/home/wins/ai-workspace/outputs/a2o-gap-profiling/20261103-bfcl-q4-queued/status.json`

CUDA llama.cpp 已构建，模型 SHA-256 与本地 shared/models 一致。等待 SRNO oracle 队列退出、GPU 无 compute PID 且低占用连续3次；间隔60秒。当前不是采集完成状态。

同一组20例做1/4/8并发对照，随后800例按8并发采集。正式输出留在远端同名目录；字段、边界与命令见 [BFCL.md](../../BFCL.md)。本地14项测试通过，远端4项采集契约测试通过，官方800例与 checker 导入检查通过。模型 GPU 推理与全量轨迹只能在资源释放后验证。

第一次 CUDA 构建因共享源码缺少 cstdarg 声明失败，原始日志留存；独立 CMake 构建参数补充标准头后构建成功，`prepared.json` 保存二进制 hash。此前 `failure.json` 是准备阶段失败记录，不代表当前等待队列失败；实时状态以 `status.json` 为准。
