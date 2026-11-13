# Qwen3-8B 默认模型与关键成本扫描

2026-11-13起，默认文本成本实验使用Qwen3-8B Q4_0，身份配置在`configs/model-default.json`。纯文本模型没有图像能力；Qwen2.5-VL的旧图像验证不能继承到本轮。历史Qwen3.5文档、原始数据与配置保留原模型名称，不批量替换。

## 模型与执行边界

- 上游：`QuantFactory/Qwen3-8B-GGUF`，固定revision `1a85ad16e1e80725404ad707448c10dd414ed045`，文件`Qwen3-8B.Q4_0.gguf`，来源为`Qwen/Qwen3-8B`。
- 文件大小4774749376字节，SHA-256 `7578dd44850f97bea0f89d6553a2b3260de07185aa363852dbbe582539ce8066`。本地复用`../shared/models/`，手机存放在Termux私有模型目录，不放入Git。
- 已读取实际GGUF张量目录：253个Q4_0、145个F32、1个Q6_K；Q6_K正是`output.weight`，形状4096×151936。文件名Q4_0不等于全张量Q4_0。
- CPU显式6线程，HTP主机4线程，CPU/HTP-only registry，默认安装Hexagon主机库与v75 DSP库；OPBATCH1024/OPQUEUE16、HMX开启。禁用KV整理、预搬移和稀疏。
- 首批保留`offload_kqv=false`、Flash Attention关闭，以移除DeltaNet后的现有offload路径建立成本表；CPU attention和输出头仍存在，绝不命名为全算子纯HTP。KV offload与FA另设条件验收，不能静默改动本表配置。
- context4608、batch/ubatch128、F16 KV；4608覆盖H4096、最大P128或P40+D320。与旧context8192及1/1提交参数的表分开保存，不能把差值单独归因于换模型。

## 先准入，再采集

构建使用缓存工具链、只读共享源码和私有探针目录；临时主机地址显式传入，不修改全局SSH配置：

```sh
python3 scripts/build/qwen35_preflight.py --service \
  --host wins@100.120.152.75 \
  --output results/20261113-qwen3-8b-migration/build-single-resident \
  --remote-directory /home/wins/ai-workspace/tools/llama.cpp-hexagon-probes/20261113-qwen3-8b-single-resident

python3 scripts/collect/model_cost_campaign.py --phase preflight \
  --binary results/20261113-qwen3-8b-migration/build-single-resident/agent

python3 scripts/collect/model_cost_campaign.py --phase collect \
  --binary results/20261113-qwen3-8b-migration/build-single-resident/agent
```

首个脚本保留历史文件名，但构建服务不绑定模型。采集配置为`configs/qwen3-8b-key-costs.json`。准入要求CC/HH/HC/CH四路径分别通过真实自由生成JSON、同一上下文两轮任务码记忆、EOG提交和检查点恢复后的状态/logits一致检查；随后验证短历史以及实际H4096的完整状态恢复和有限logits。准入成功时服务和手机锁保留，供采集阶段继续使用；失败尝试停止自己的服务并保存清理结果。所有手机操作仅Termux SSH，无ADB回退。

本轮选用`multi_turn_miss_func_62/t00-s03`的较长响应来覆盖新tokenizer下的D320；原BFCL轨迹的prompt文本及响应文本使用新模型重新分词。只复用文本内容，不复用旧token ID或旧KV快照。固定token计时不评价新模型在原任务上的回答质量，跨后端logits差异单独报告。

## 正式矩阵与产物

H=4096；P为24至64逐整数以及66、68、72、80、96、128，共47点。每后端每点十次，轮内乱序，每点CPU/HTP先测各五次；不将几何补点放到独立追加批次。完整473个配对请求的顺序落盘到`schedule.json`，固定seed20261113，轮内使用seed+round。

decode固定P40，每后端连续逐token执行至D320，在1、8、16、32、48、64、96、128、192、256、320记录累计时间，共三轮，嵌入前三轮P扫描。纯阶段耗时不含初始化、历史建立、恢复和检查点验证；这些开销单列。逐token计时包含调用与同步，不含采样。

状态成本另测H512/2048/4096、P40，各预热一次加正式五次，记录快照导出、CPU→HTP准备、准备实际导入工作、验证开销、HTP→CPU交接和CPU恢复及字节数。该表包含现有helper的字节核验，不能称为纯DMA带宽。

`results/20261113-qwen3-8b-key-costs/`保存：

- `preflight.json`及每路径原始响应，`fixture.json`为新token夹具；`plan.json`冻结模型、构建、配置和文本来源。
- `status.json`为进度，`background.log`为独立后台日志；`measurements/`逐请求原子保存，可按相同配置跳过已完成项。
- `stage-times.csv`预期1006行：940条prefill耗时、66条decode检查点；`state-times.csv`含预热及正式状态成本。`cost-table.csv`与`state-cost-table.csv`保存均值/中位数等聚合，缺页与swap单独保留。
- 每阶段记录进程RSS、swap和缺页计数；数值一致性标记与原始耗时都保留，默认不自动剔除慢点。
- `completion.json`仅在全部采集、汇总并停止本次服务、释放锁后生成；失败另存原始响应和清理状态。

正式采集可用`caffeinate -i python3 -u ... --phase collect`在独立进程会话启动。聊天关闭不终止它，但Mac和手机仍需联网；先核对PID与首条正式测量落盘再交接。网络故障有界重试，协议/计算故障停止；不承诺机器重启后自动恢复，也不把只启动成功当成采集完成。

## 本轮独立后台流程

`scripts/run/model_migration_campaign.py`将模型断点续传、SHA-256校验、基础准入和正式采集串行连接；任一阶段失败都停止，不会跳过准入。通过`caffeinate -i`与独立进程会话启动，主进程脱离聊天。当前阶段与子进程PID在`results/20261113-qwen3-8b-migration/pipeline-status.json`，主日志为同目录`background.log`，分阶段日志为`installing_model.log`、`preflight.log`、`collecting.log`。部署期间不应将后台已启动写成基础验收已通过。

```sh
python3 scripts/run/model_migration_campaign.py
```

模型续传重传最后16MiB以覆盖未完成SFTP窗口；之后使用32MiB有限长度分块，每块SHA-256核验后才提交进度，网络失败最多重试8次，重试前核对owner并停止本文件的接收器。最终仍须核对完整文件SHA-256，清理错误单独保存，不能覆盖传输原始错误。状态为`collecting`时，详细测量进度读取`results/20261113-qwen3-8b-key-costs/status.json`；最终以`completion.json`及锁释放记录验收。

## 双驻留换页诊断与单驻留测量

原双驻留准入结果保存在`results/20261113-qwen3-8b-dual-resident-diagnostic/`。四路径JSON/记忆/检查点、H4096状态与logits均通过，但H4096预热的CPU decode窗口新增71510次major faults、swap约2.2GiB，不作为无换页纯阶段基线。

正式P/D扫描使用`single_resident=true`：仍在同一个进程按既定逐点配对顺序执行，每次需要切换后端时，先销毁旧context/model，再加载目标模型。新加载的后端执行32-token预热，然后恢复同一个CPU权威历史快照，最后才计时prefill及连续decode。模型释放、模型/context初始化、32-token预热、历史建立和恢复均单列，不计入纯阶段成本。并非把CPU和HTP各集中跑一整批；每个条件的prefill/decode始终共用同一个context。

单驻留短检查的CPU/HTP窗口均为0新增major faults、0 swap；H4096仍须独立验收，不由短检查外推。单驻留阶段表不能直接当成两个模型常驻的在线路由成本表，更不能把反复重载模型的初始化耗时从完整E2E中隐藏。

P/D表完成后先落盘`stage-summary.json`和阶段CSV，再显式切换双驻留测状态交接并重验HC/CH。转换记录在`state-residency.json`；状态请求前后另存进程资源边界，包含该请求的bootstrap/预热，不冒充逐重复的资源计数。两种驻留条件不混为一种成本口径。

## 默认attention排查已停止

开启KV offload的原H4096、短上下文、双HTP会话和缩小映射块均未通过准入，未启动新的473次扫描。失败日志保留在原results目录，调查见[映射排查记录](../validation/htp-kv-mapping-investigation.md)。2026-11-13将专用入口、配置和补丁移至`archive/diagnostics/20261113-default-htp-abandoned/`；主线继续使用单会话、KV offload关闭及FA关闭的已验证基线。Qwen3迁移与已完成成本表保留。
