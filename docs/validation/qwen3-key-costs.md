# Qwen3-8B固定配置成本表：完成验收

2026-11-13。只验收本轮固定配置，不扩展后端开关或补跑其他条件。运行入口和范围见[采集指南](../guides/qwen3-model-costs.md)。

## 完整性与随机重复

473个配对请求全部完成：47个P点各十次，共940条prefill耗时；连续decode三轮，每后端11个累计检查点，共66条decode数据。另有H512/2048/4096各一次预热、五次正式状态测量，共18条状态记录。完整顺序与固定seed、每点CPU/HTP先测5/5、所有H/P/D和线程数均已逐请求核对。

阶段CSV逐行与原始JSON一致；状态CSV及均值/中位数表与原始重复重新计算一致。全部473组prefill logits数值标记通过，最低cosine为0.99798848。单驻留CC/HH及状态测量前的双驻留HC/CH准入均通过；这些检查不等价于完整任务质量或严格跨后端无损证明。

结束时另通过Termux SSH只读核实：无模型进程、模型锁不存在。

## 图表与过滤

输出目录：`results/20261113-qwen3-8b-key-costs/report/`。

- `figures/prefill-all.{png,pdf,svg}`：全部940个样本参与均值。
- `figures/prefill-cleaned.{png,pdf,svg}`：按各P、各后端独立的Q3+1.5IQR上界排除50个慢点，每组保留8–10次，仅作为敏感性分析。
- `figures/decode.{png,pdf,svg}`：三轮全部保留，检查点为同一连续运行的累计时间，不是独立重复。
- `excluded-prefill.csv`逐样本记录来源、轮次、耗时和阈值；原始数据不修改。`prefill-summary.csv`同时保存全部样本与清理版统计。
- `audit.json`、`raw-identities.json`记录验收与原始文件哈希；`figure-qa.json`记录文字边界/重叠检查。三个图已完成全图与面板像素审阅。

复现：

```sh
conda run --no-capture-output -n visiontrim-qwen25vl python scripts/plot/model_costs.py \
  --input results/20261113-qwen3-8b-key-costs
```

## 本批原始均值

| 条件 | CPU秒 | HTP offload秒 |
|---|---:|---:|
| Prefill P24 | 1.866 | 3.462 |
| Prefill P31 | 3.275 | 4.374 |
| Prefill P32 | 2.486 | 3.770 |
| Prefill P40 | 3.055 | 4.042 |
| Prefill P64 | 4.931 | 4.942 |
| Prefill P128 | 10.094 | 7.423 |
| 连续Decode D320 | 60.708 | 124.013 |

CPU prefill仍有明显锯齿，不能用单一P阈值概括全部逐整数点。上述P64的两后端均值接近；P128的HTP阶段更快。D320下CPU的累计decode更快。以上只适用于本批固定配置。

状态成本中，H4096的CPU→HTP准备均值305.819ms、HTP→CPU交接543.171ms；准备导入工作本身46.809ms，其余包含现有字节核验。不得把整个helper耗时解释为纯DMA或独立KV格式转换成本。

## 资源与口径

P/D扫描为单模型驻留、逐点配对。模型释放、重载、32-token预热及历史恢复单列，不进入纯阶段耗时；同一条件的prefill/decode共用context。状态交接测量另为双驻留，不能将两种驻留条件混成一个在线E2E成本表。

Prefill窗口新增major faults中位数为0，CPU最大186、HTP最大12；对应swap中位数约4.6–4.7MiB，最大约32.4MiB。六次连续decode运行的decode窗口均无新增major faults。并非所有prefill样本绝对无换页，未按资源标记自动删除任何耗时。

本轮标签固定为“HTP offload（CPU attention）”：`offload_kqv=false`、FA关闭，输出头Q6_K留CPU。验收与图表不将其表述为默认启用NPU或纯HTP的性能。
