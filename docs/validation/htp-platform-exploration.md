# HTP 平台与64-token边界探索

## 直接证据与当前状态

已完成8个诊断点：31、32、63、64、65、66、68、72，每点一次。私有DSP布局探针实际记录1984个MUL_MAT的执行路径和量化HMX分块。原版运行库的同批次P60至72逐整数十轮扫描已完成，130个双后端请求、260条测量；正式统计以完成后的 `results/20261111-htp-platform-exploration/README.md` 和 `direct-layout-summary.json` 为准，诊断时间不混入原版性能数据。

固定H4096、CPU6线程、HTP主机4线程、batch128、F16状态、Qwen3.5-9B-Q4_0。完整状态缓存复用原六线程服务的CPU权威快照，恢复包含字节一致性检查。无KV整理、预搬移或手机设置/频率改动；回放不执行动作。

## 31/32：HVX到HMX

P31的248个HTP矩阵算子全部走HVX。P32及本次63至72的各点，245个走HMX、3个仍走HVX。每个MUL_MAT日志均带探针标签，未发现HMX错误标签。此结果是私有探针中的直接分支记录，不再只依赖总耗时猜测。

## 64/65：模型分块和硬件布局同时变化

主导矩阵在64、65均走HMX；第二次上涨不是重新回退HVX。量化HMX内部mc从64变为96，32行tile从2组变3组，nc也会缩小。例如N4096/K12288的FFN Down，nc从160变128，N块数从26变32。

更关键的是Qwen3.5 DeltaNet使用64-token chunk。源码 `src/models/delta-net-base.cpp` 中 `CS = kda ? 16 : 64`，通过pad和n_chunks构造内部张量。该模型采用非KDA路径，P64为1个块，P65补到128、变为2个块。诊断日志直接看到：

```text
key_gdiff_t: 64:128:1:32 → 64:128:2:32
SOLVE_TRI:   64:64:1:32  → 64:64:2:32
```

| 一次诊断指标 | P64 | P65 |
|---|---:|---:|
| HMX / HVX矩阵算子数 | 245 / 3 | 245 / 3 |
| 量化HMX典型mc | 64 | 96 |
| DSP提交批次数 | 1660 | 1756 |
| CONT累计时间 | 0.389秒 | 0.763秒 |
| 全部DSP矩阵算子累计时间 | 1.415秒 | 1.480秒 |
| 带探针HTP prefill墙钟 | 3.065秒 | 4.171秒 |

因此，硬件tile的变化不能单独解释整个台阶；DeltaNet内部块数、拷贝重排、CPU侧图节点和调度也要纳入分析。DSP算子时间不包含CPU侧节点与全部RPC/等待，不能相加后冒充完整墙钟归因。每点一次诊断也不能替代原版十轮的波动估计。

## 探针实现与失败记录

`src/a2o/native/profiling/htp-layout-probe.patch` 只在DSP内部上下文添加字段，记录原有分支与原有HMX标准量化路径实际选择的mc/nc/layout；不改计算门槛与分块算法，不扩大RPC wire ABI。使用现有8个PMU槽传输 `route,M,N,K,type,mc,nc,layout`，魔数0xA2100000；这些槽不再是硬件PMU计数。F16等未采集内部布局的路径mc/nc保持0，不据此编造布局。

首次复用了旧v79脚本默认值，但设备实际为v75，导致HTP session启动失败。失败运行保存在 `results/20261111-htp-platform-route-diagnostic/`，无正式测量计入。已显式改为v75重编译，使用私有单目录ADSP路径加载；共享源码与安装库未更改。原版v75 DSP库的手机/主机构建SHA-256匹配，记录在 `original-dsp-identity.json`。DeltaNet源码摘录和libllama校验保存在 `source-evidence.json`。

## 执行入口和结果

- 诊断配置：`configs/htp-platform-route-diagnostic-v75.json`。
- 正式配置：`configs/htp-platform-p60-72-cpu6-t10.json`。
- 串行接续：`scripts/collect/htp_platform_campaign.py`，等诊断完整停止并释放锁后启动原版库正式测量。
- 诊断响应与逐请求profile日志：`results/20261111-htp-platform-route-diagnostic-v75/`。
- 正式原始结果：`results/20261111-htp-platform-p60-72-cpu6-t10/`。
- 总进度与分析：`results/20261111-htp-platform-exploration/`，`status.json` 区分diagnostic与baseline，正式完成后自动更新分析。

后续若要做因果干预，应把DeltaNet chunk size、HMX布局和CPU侧算子分别控制；本次仅收集原算法的路径证据与同批次性能，不宣称已经完成单因素A/B或获得端到端加速。
