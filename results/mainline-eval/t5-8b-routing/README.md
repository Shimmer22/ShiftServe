# T5 小米 14 Pro 三路真值与路由回放

结果对应 `1120TODO.md` 的 T5 和 Fig. 7。采用 T1/T2 已验收的 8B 配对计时构建，构建 ID 为 `t1-t2-paired-truth-21d2e5ce0abe46fcaa47dd77e08398ce`。原始 `t1-frozen-20261120` 构建中的标定接口没有冻结 Decode 序列及 Tool Gap 返回后总耗时协议。采集使用 Termux SSH 的设备模型锁串行运行。

## 真值切片

Qwen3-8B Q4_0，CPU 4 线程、HTP0 4 host 线程，共享权重。固定上下文 4352、Gap 1000 ms、Decode 32 token、紧凑预算 1536 token、稀疏层步幅 4、最少淘汰 256 token，开启 KV 字节回读。H 为 2048 和 4096，P 为 16、24、40、56、80、128。P=16 取自已有采集入口的最短默认档位，其余来自 8B 成本配置。用现有 8B 历史曲线夹具的 token 构造输入，从固定位置取相同的 32 token 强制 Decode。每个点做三次配对，同一 P 的三次重复共享构建一次的输入历史快照。封版接口每次重复依次执行 CC、HC-Full、HC-Compact，并在各路径前恢复原始快照。

返回后总耗时包括实际等待、恢复、Prefill、裁剪、交接和 Decode。快照前缀构建与模型加载未计入返回后延迟。HC 的准备时间单列，只有溢出 Gap 的时间计入总耗时。每个 P 的中位数用于 Fig. 7a，同时记录每条路径的胜出次数和前两条路径的中位数差。可选 D 缩放用历史曲线的 CPU 单步线性拟合对 D=32 实测 Decode 做归一化，其他阶段保持实测中位数。D=16/64/96 的数值为外推。

## 路由回放

第一次配对用于校准，后两次用于回放。所有策略按同一由种子 20261120 决定的顺序回放后两次采样。每步代价来自该次三路测得的返回后耗时。Oracle 可逐样本选择三条路径中的最短者。逼近率为累计 Oracle 耗时除以相同样本上的策略累计耗时。

- Static-Threshold 沿用 P=32 阈值，低于阈值走 CC，其余走 HC-Full。
- Offline-LUT 用第一次实测为每个 H/P 单元固定最短路径。它预先知道测量网格的覆盖点，两次回放样本没有进入查表。
- Greedy-Online 和 M1-LCB 共用两组校准样本 (2048, 24)、(4096, 56)、历史 Decode 先验和同一紧凑开销估计。固定 K=1536，保证选择的每条路径在网格中都有真值。两者分别用后验均值和受限 LCB 决策，每步只观测自己选择的路径。该回放按固定 D 和已有 HTP 候选计费，包含实测的超 Gap 等待；运行时的等待预算与跨轮锚点变化由其他任务验证。

`fig7a-truth.csv`、`fig7a-decode-projections.csv`、`fig7b-ratios.csv`、`fig7b-steps.csv` 和 `summary.json` 由 `scripts/analyze/t5_routing_truth.py` 生成。每个切片下留有计划、逐条响应、健康检查、启动清单和原始运行日志。

## 复跑

```sh
root=results/mainline-eval/t5-8b-routing
for h in 2048 4096; do
  python3 scripts/collect/p_cost_truth_sweep.py \
    --output "$root/h$h" --build results/20261120-paired-truth-build \
    --fixtures results/20261114-qwen3-history-decode/fixture.json \
    --force-hc-compact --h "$h" --p-points 16 24 40 56 80 128 \
    --decode-tokens 32 --keep-tokens 1536 --context-size 4352 \
    --order-seed 20261120 --reps 3 --batch-reps || exit $?
done
python3 scripts/analyze/t5_routing_truth.py "$root" "$root/h2048" "$root/h4096"
```

`pilot-h2048` 保留批量模式启用前的接口试采。首个 (2048, 56, 32) 点的三路 Decode 序列一致，KV 字节回读正常。该点用于估计历史预填充的设备机时。试采在下一点计算期间中止，状态文件仍保留中止前的快照。

正式采集的首批 (2048, 56, 32) 已保存 3 份配对响应，每份均通过冻结 Decode 校验，HC-Compact 每次保留 1536 个、淘汰 568 个 token。切片与路由逼近率在两个 H 均采完后自动汇总。

## 五点重采

首版于 23:32 停在 H=2048、P=80。三次 Prefill 余弦相似度均为 0.9759，低于 0.99。原始失败响应保留在首版切片目录。重采见 `retry-p5/`，沿用同构建、夹具和校验口径，P 网格为 16/24/40/56/128，各点三次配对。依据重采网格，在线策略校准点为 (2048, 24) 和 (4096, 40)。

## 01:55 回访与短历史补测

五点重采的两条切片均已通过。30 组三路冻结样本全部通过数值和 token 校验，原始日志共保留 60 次 HTP→CPU 字节回读。Fig. 7b 使用 20 个留出样本，Oracle 为 228.17 秒。Static-Threshold 为 92.72%，Offline-LUT 为 99.72%，Greedy-Online 与 M1-LCB 均为 96.19%。

Fig. 7a 的网格内 HC-Full 在 H=2048、P=56 连胜三次，HC-Compact 在 7 个网格点连胜。CC 在 H=2048、P=16/24 有单次胜出，尚无稳定胜点。补测 H=1792、P=1/8/16，各三组，沿用封版、夹具、固定 D=32、K=1536 及 0.99 数值门槛。P=1/8 用于观察极短工具返回，P=16 与主网格重叠。补测只用于 Fig. 7a，Fig. 7b 继续使用原五点网格。

补测原始记录见 `short-h-supplement/h1792`。验收时执行：

```sh
root=results/mainline-eval/t5-8b-routing
python3 scripts/analyze/t5_routing_truth.py "$root/acceptance" \
  "$root/retry-p5/h2048" "$root/retry-p5/h4096" \
  --supplement "$root/short-h-supplement/h1792"
```

## 02:25 短历史切片回访

H=1792、P=8 的三次配对均通过，CC 三次胜出。P=16 的采集日志记录到两条 HC 路径的三次回读，但 HTTP 响应连接被重置，未保存可复验 JSON。已核对手机模型锁释放和同一采集器 SHA-256，并按原计划续跑 P=16 和 P=1。原日志保留。

## 02:50 传输恢复

第二次 P=16 的六次 KV 回读在 02:31 已写入设备日志，随后设备可响应同一互斥锁保护的健康检查，宿主的长时间 HTTP 请求仍等待响应。已中断该采集器，由原清理入口拉回日志并释放模型锁，再通过 Termux SSH 本地转发续跑，HTTP 请求经设备 loopback 发送。采集参数、构建及采集器哈希沿用原计划，已通过的 P=8 响应保留。

## 最终验收

2026-11-21 03:15 复核通过。三条切片合计 39 组三路配对。CC 在 H=1792、P=1/8/16 各三次均胜；HC-Full 在 H=2048、P=56 三次均胜；HC-Compact 在 H=2048 的 P=40/128 和 H=4096 的全部五档各三次均胜。各样本三路冻结 Decode token 一致，两条 HC 路径的实际 HTP0 上下文和 KV 回读已核验。

Fig. 7a 和 Fig. 7b 的最终数据见 `acceptance/`。Fig. 7b 保持使用主网格的 20 个回放样本，M1-LCB 与 Greedy-Online 的累计耗时和逼近率分别同为 237.20 秒和 96.19%，没有观测到 LCB 相对贪心的改进。原始日志保留在各具名采集目录，汇总提供计划和日志哈希。
