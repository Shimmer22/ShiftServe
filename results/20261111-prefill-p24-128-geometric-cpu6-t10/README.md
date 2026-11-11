# CPU/HTP Prefill 完整扫描与锯齿调查数据

这是本次最详尽版本：H=4096，P=24–64逐整数，再取66、68、72、80、96、128，共47点；每点每后端10次，CPU6线程、HTP主机4线程。共470个双后端请求、940条耗时。Qwen3.5-9B-Q4_0、8192上下文、batch128、F16状态；无KV整理与预搬移。

## 数据入口

- [原始940条阶段耗时](stage-times.csv)：包含轮次、P、线程数、准备恢复时间和来源批次。
- [原始数据图](figures/prefill-dense.png)、[PDF](figures/prefill-dense.pdf)。
- [清理长尾后的图](figures-cleaned/prefill-dense.png)、[PDF](figures-cleaned/prefill-dense.pdf)。
- [保留931条](figures-cleaned/cleaned-stage-times.csv)、[剔除9条](figures-cleaned/excluded-stage-times.csv)、[规则与数量](figures-cleaned/cleaning-summary.json)。
- [原始请求/响应归档](evidence/phone-records.tar.gz)：493份原始文件，包含470个完整测量JSON、健康信息、预热、正常响应、服务日志与启动清单。[SHA-256清单](evidence/phone-records-manifest.json)可逐文件复核；不含模型、服务可执行文件或状态快照。
- [精确token输入源](evidence/token-source.response.json)、[服务构建清单](evidence/service-build-manifest.json)、[二进制SHA-256](evidence/service-binary-sha256.txt)。
- [完整调查](../../docs/validation/cpu-prefill-sawtooth.md)、[源码证据和拟合结果](../20261111-cpu-prefill-sawtooth-audit/summary.json)。

## 口径

这是长度受控token微基准，非完整Agent任务。准备、历史状态恢复、模型初始化和迁移均不进入纯prefill时间。数值相似度保留在原始记录中；性能图不按数值标记区分。

第一批测24至64及128，随后追加66、68、72、80、96；每批次每轮P乱序，每点CPU/HTP先测各5次。两批次并非在同一时段交错测量，逐行source_run保留批次来源。原始数据一律保留。

清理上尾时，每个P/后端独立计算线性插值Q1/Q3，剔除高于Q3+1.5IQR的点，不处理低尾，共9/940；各组保留9至10次。删除点不进入清理后均值。

## 离线复核和复现

```sh
python scripts/validate/prefill_sawtooth_results.py
python scripts/analyze/cpu_prefill_sawtooth.py
python scripts/plot/pd_prefill_dense.py --input results/20261111-prefill-p24-128-geometric-cpu6-t10/stage-times.csv
python scripts/plot/pd_prefill_dense.py --input results/20261111-prefill-p24-128-geometric-cpu6-t10/stage-times.csv --clean-upper-tail
```

验收仅需Python标准库；分析需要numpy；绘图还需matplotlib、Pillow、本机figure-style技能的kernel.py及Arial Unicode MS字体。原环境复用 `data` Conda环境。

原两批次设备配置在 `configs/prefill-p24-64-128-cpu6-t10.json` 和 `configs/prefill-geometric-supplement-cpu6-t10.json`。后续单批次重新测47点可用 `configs/prefill-sawtooth-reproduction-h4096-cpu6-t10.json`，它引用本目录已保存的精确token源；该配置产生的是新实验，不等同于本次两批次采集结果。构建、手机模型路径、私有运行库、Termux SSH配置与CPU/HTP registry依赖见[执行指南](../../docs/guides/pd-stage-sweep.md)。
