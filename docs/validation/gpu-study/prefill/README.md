# 第一张曲线补充 OpenCL GPU

复用手机已有共享OpenCL运行库与Qwen2.5-VL-3B Q4_0。GPU路径为OpenCL prefill→完整状态交接→CPU decode，与原HTP/CPU路径一致。图中GPU使用`OpenCL/CPU*`，星号表示独立会话补测；CPU/HTP来自2026-11-05晚间Termux SSH正式重跑数据。图形入口为`scripts/plot/opencl_boundary.py`。

固定H2048/4096，P16/24/31/32/33/40/48/56/64/128/256/512/1024；ctx6144、batch128、F16 KV、FA关闭、CPU repack关闭，无KV整理和预搬移，无视觉输入。OpenCL设备固定GPUOpenCL（Adreno 750），设备不存在即失败；日志确认37/37层权重与KV分配至OpenCL。已有后端内部的少量CPU图分割不等同于设备选择失败后的静默回退。

每个H/P先3次CPU+3次GPU完整响应校准，按cosine≥0.99准入，NRMSE保留；再独立3次GPU正式测量。全扫描均强制候选路径，未使用Router择优。提前准备从HTTP发起开始，Gap目标200ms；第一张图纵轴从实际Tool返回至完整JSON发布，包含未隐藏准备、观察处理、同步完成的GPU prefill、带字节核验的状态交接和CPU decode，不包含模型初始化及共同历史建立。初始化与历史建立单独保留。生成动作不执行。

同一进程内双上下文，CPU权威状态；完整输出、token IDs、EOG与CPU参考一致，GPU→CPU迁移后逐字节核验状态。固定历史夹具与原曲线相应H的历史字节相同，HTTP正文逐P核验hash。实际图像与独立记忆任务不在本次文本微基准范围内。

所有手机条件均通过Termux SSH串行运行，宿主机与设备持模型锁。不改手机设置、用户数据或CPU频率。失败预检与传输日志保留在各具名results目录，失败轮未进入正式测量，不纳入图中。原图不覆写，新图在本目录figures中。

计时主口径为CLOCK_BOOTTIME；同时保留steady_clock诊断。两种时钟非原子读取，GPU预检曾出现0.1075ms的Gap读取偏差；新GPU扫描仅将此诊断容差显式设为1ms，不修改原值，不改变主时钟0.01ms恒等式与事件顺序验收。原CPU/HTP数据沿用原0.1ms诊断容差。

跨会话图比较用于性能范围筛查，没有控制温度或频率，也没有测能耗；3次重复不能证明微小差异稳定。各H同一会话的CPU校准成本另保留，可用于检查历史曲线漂移，不混作正式测量点。

## 复现

使用工程已有NDK入口编译仅负责选择设备的agent驱动，OpenCL运行库复用shared及手机现有版本，不重建/替换后端库：

```sh
PYTHONPATH=src python -c 'from a2o.environment import build; print(build())'
python scripts/profile/opencl_boundary_campaign.py --output results/<新的具名目录>
```

配置：`configs/opencl-boundary-h2048-termux.json`、`configs/opencl-boundary-h4096-termux.json`。协议沿用HC作为候选路径名称，设备原始行额外记录`candidate_backend=opencl`，manifest记录其实际含义；图中不将HC误标成HTP。

## 正式结果
共234条记录：156次独立校准、78次GPU正式测量。26/26条件准入，78/78次正式返回时准备已ready。全部响应、token/EOG、HTTP正文及迁移核验通过；跨H/跨后端完整响应一致。13项相关回归检查通过。图形文字碰撞/越界检查与视觉检查通过。
26/26个H/P条件中GPU的三次中位数低于原CPU/HTP曲线。这是已有会话上的观测比较，不是跨会话统计显著性结论。
| H | P | CPU/CPU 秒 | HTP/CPU 秒 | OpenCL/CPU 秒 |
|---|---|---:|---:|---:|
| 2048 | 32 | 2.233 | 1.804 | 1.221 |
| 2048 | 1024 | 55.376 | 25.177 | 10.494 |
| 4096 | 32 | 2.518 | 2.062 | 1.566 |
| 4096 | 1024 | 66.714 | 34.731 | 13.932 |

图：`figures/01_backend_boundary.png`、矢量`figures/01_backend_boundary.pdf`；逐点数值`figures/boundary.csv`，带源文件hash的数据`figures/plot-data.json`。

原始GPU目录：
- `/Users/hui/code/exp/2026-11-a2o-gap-profiling/results/20261106-144555-894164-http-oracle-grid`
- `/Users/hui/code/exp/2026-11-a2o-gap-profiling/results/20261106-145744-737602-http-oracle-grid`
