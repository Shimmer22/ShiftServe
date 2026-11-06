# CPU / HTP / OpenCL decode 独立比较

本实验单独测decode，不把prefill优势混进decode结论。配置为H2048/4096、P32/1024，decode前KV分别2080/3072/4128/5120 token；Qwen2.5-VL-3B Q4_0、F16 KV、ctx6144、batch128、4个CPU线程、FA关闭、无CPU repack、无KV整理/预搬移。文本历史与输入复用上一张曲线的夹具，历史按原分段token边界构造并核对导出字节相同。

同一个原生进程先用CPU建立每个H/P的相同prefill状态，随后CPU、HTP、OpenCL分阶段测量；加速器模型逐一加载/释放，没有HTP/GPU同时执行。每个条件均从相同CPU快照恢复，显式指定HTP0或GPUOpenCL，不可用就失败，不静默回退。初始化、历史建立、tokenize、prefill、快照恢复、返回CPU的迁移与decode分别记录。手机通过Termux SSH持宿主机/设备锁，不更改设置、CPU频率或用户数据。

## 两个decode口径

- 自然JSON：从CPU prefill提供的首个候选token开始，后续在所测后端greedy生成，直到EOG。与旧入口一致，记录6次单token评估，包含EOG写入；首token的prefill计算不属于decode。完整JSON、token IDs、EOG必须与CPU参考一致。
- 64步固定回放：从同一prefill快照重新开始，将CPU参考JSON的非EOG token循环组成64个固定输入，逐次调用单token评估并同步。它是计算微基准，不是自然长响应。所有64步都计入，不丢弃开头步骤；原始token IDs与逐步耗时保留。

所有自然/回放步骤均在计时区间之外抓取完整logits，并与CPU同位置参考比较，cosine≥0.99、NRMSE记录。短JSON与64步回放分别准入，不能用短响应通过替代长序列通过；一项长回放的预热或任一重复失败，就排除整个后端/H/P的有效回放中位数，仅保留明确标注的空心原始点。图中decode计算时间排除数值审计，但审计发生在步骤之间，不等同于无审计的连续生成吞吐。CPU参考每条件一次，各后端每条件一次完整预热，再各测3次；预热与参考不进入正式汇总。每次候选decode后迁回CPU，并逐字节核对KV/位置状态。回放不执行任何生成动作。

主时间为CLOCK_BOOTTIME，同时保留逐步steady_clock诊断。第一张图的完整响应包含prefill/交接/CPU decode，本图是已驻留KV上的decode计算，二者不能直接相加推导未经实测的端到端路径。

单会话、每条件3次，后端顺序为CPU→HTP→OpenCL，条件顺序在每次重复中随机化。未进行跨会话或后端顺序交叉验证，未控制温度/频率，不能将微小差异推广成稳定优势。没有图像输入或独立历史记忆任务验收。

## 文件与复现

```sh
PYTHONPATH=src python -c 'from a2o.environment import build; print(build())'
python scripts/profile/decode.py --config configs/decode-backends-termux.json
/Users/hui/miniforge3/envs/visiontrim-qwen25vl/bin/python scripts/plot/decode.py results/<本轮decode目录> --output results/<本轮decode目录>/figures
```

`runtime.log`保留后端分配与迁移日志；`device/trials.jsonl`为原始记录，`checked-rows.jsonl`附加响应验收；`manifest.json`记录配置、源文件/二进制/输入hash；图与逐步CSV另存于figures。失败预检`results/20261106-154715-364986-decode-backends-smoke`因导出历史文本重tokenize边界不一致退出；改用原分段构造后`results/20261106-154925-435421-decode-backends-smoke`的13条预检全部通过，均不进入正式图。

## 数值诊断修订

首轮64步主实验`results/20261106-155219-860183-decode-backends`在HTP预热时整体最低cosine=0.989187，门槛未通过而中止，21条记录保留，不进入最终图。本轮将自然JSON与固定长回放的cosine/NRMSE和逐步cosine分别记录；短响应/迁移失败仍终止，长回放失败则继续采集明确标记的诊断数据，不降低0.99门槛。`completion.json`区分completed、response_and_state_passed与all_numerical_passed，进程正常退出不代表所有回放通过。

## 正式结果与验收
52条记录完整采集：4个CPU参考、12次后端/条件预热、36次正式重复。52/52条自然JSON、token IDs/EOG、状态迁移与短decode数值验收通过。64步回放按条件准入10/12组；HTP的两个H2048条件未准入，completion的passed/all_numerical_passed明确为false，不将退出码0视为全部通过。

### 6次单token评估中位数（ms，含EOG写入）
| H | P | CPU | HTP | OpenCL GPU |
|---|---|---:|---:|---:|
| 2048 | 32 | 644.1 | 1247.3 | 424.2 |
| 2048 | 1024 | 695.2 | 1433.2 | 508.5 |
| 4096 | 32 | 739.0 | 1665.3 | 580.4 |
| 4096 | 1024 | 774.6 | 1950.4 | 658.4 |

### 64步回放：每次评估平均耗时的三次中位数（ms/token）
| H | P | CPU | HTP | OpenCL GPU |
|---|---|---:|---:|---:|
| 2048 | 32 | 106.71 | 未准入 | 70.98 |
| 2048 | 1024 | 114.88 | 未准入 | 83.39 |
| 4096 | 32 | 122.90 | 275.81 | 95.44 |
| 4096 | 1024 | 127.56 | 323.02 | 108.52 |

本轮OpenCL的自然decode和已准入回放观测中位数均最低，但只能限定在这四个条件、当前参数和顺序执行协议，不能推广为所有decode最优。GPU的回放评估速率为9.22–14.09次/s，CPU为7.84–9.37次/s；这是排除审计的同步计算口径，不是完整应用生成吞吐。

HTP H2048/P32、P1024的回放最低cosine分别为0.989187、0.988229，第一次低于0.99分别在第62、57步；图中用空心三角保留原始试验，不计算合法中位数。短JSON未失败，不能据此称HTP无法生成自然长响应。

两轮主实验的CPU回放结果相近（前一轮因HTP长数值检查中止，未混入本图），支持主实验CPU耗时可重复；预检只有一个条件和8步，不能拿其CPU耗时与本轮64步结果直接当同协议比较。后端顺序/温度限制仍存在。

图：`figures/decode_backends.png`和矢量`figures/decode_backends.pdf`；逐点汇总`figures/decode-summary.csv`、逐步耗时`figures/decode-traces.csv`、带逐步cosine和输入IDs的`figures/plot-data.json`。9项相关已有回归检查通过，编译、设备预检和最终数据矩阵/输入/原始行/二进制/数值准入核对通过。几何与视觉检查通过。
