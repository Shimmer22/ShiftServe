# 图1的CPU线程补充实验（P≤64）

本次只扫描CPU prefill线程数；decode、共同历史建立均固定4线程。SIMD沿用原共享运行时，权重repack仍关闭。CPU/HTP双模型和上下文同时驻留，保持图1的预算与生命周期。

## 协议

- 设备：Xiaomi 14 Pro / SM8650；H2048/4096、ctx6144、batch128、F16 KV，关闭KV整理和预搬移。
- 独立选线程：1–8线程，P16/31/32/64。每个H/P先以4线程建立logits参考，每个候选线程做1次独立数值与完整输出校准，然后随机顺序完整测量3次。全部线程、全部条件通过才选择，不抽取成功子集。
- 选择指标：各H/P的prefill中位数相对4线程归一化，再对8个H/P条件等权几何平均。选择一个全局固定线程数，不逐点选择，也不使用后续曲线数据选线程。
- 独立补曲线：选定线程与新4线程对照，P16/24/31/32/33/40/48/56/64；每条件1次数值校准、3次正式测量。G目标200ms，主指标为设备实际返回到完整JSON发布，CLOCK_BOOTTIME含挂起；各阶段另记。
- 未修改绑核、手机设置或CPU频率；不执行模型动作。

## 实现与复现

原生TRIAL协议在既有pack参数后增加可选prefill线程字段，缺省4。通过`llama_set_n_threads(cpu.ctx, 4, prefill_threads)`设置多token计算线程，单token decode始终4线程。图1其他原生文件、模型与库hash需一致；新一轮内部还要求二进制一致。

```bash
python scripts/profile/cpu_threads.py --device <当前adb地址> --output results/<具名线程实验>
```

扫描配置为`configs/cpu-thread-sweep-h2048.json`和`cpu-thread-sweep-h4096.json`。执行器串行运行，自动选线程并保存选择结果，再生成独立曲线配置及测量；全部成功后调用`scripts/plot/cpu_threads.py`。

图1原文件不覆盖。补充图保留原4线程CPU与HTP线，叠加最佳线程CPU的独立复测；另给出新二进制4线程控制线，避免把轮次漂移全部当作线程收益。所有数值、响应、HTTP正文、事件时序、原始行与完整矩阵由`a2o.profiling.cpu_threads`复核，数据和失败输出只留在本地results。

选择结果属于1–8线程与本次负载域，不代表普适最优CPU配置；只改变prefill线程数，不代表已调优decode、repack、batch或绑核。
