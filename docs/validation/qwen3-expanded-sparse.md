# Qwen3扩展激活裁剪与接受率（2026-11-13）

Qwen3-8B Q4_0可以通过联合裁剪FFN及Query heads明显加速当前HTP offload decode，但这次无需训练的prefill能量选择破坏了预测质量：两个候选各8个锚点、每块8 token，连续接受前缀均为0。不能将速度收益当作可用draft收益，也不能由单条轨迹否定所有先验驱动子网络。

## 数学依据与边界

实际模型36层，隐藏维度d=4096，FFN宽度f=12288，32个Query heads、8个KV heads、每head128维。每层FFN gate/up/down约3df=151.0M MAC；Q/O投影约2d²=33.6M MAC；K/V约2d×1024=8.4M MAC；H4096 attention的QK及AV约2H×4096=33.6M MAC。词表151936，输出头约dV=622.3M MAC/token，当前留CPU。MAC不是实测耗时或DRAM流量，量化、缓存、提交、CPU/HTP交错和kernel利用率都会改变延迟。

若FFN保留比例r、Query heads保留比例q，每层主体近似为3rdf+2qd²+2d×1024+2Hqd。保持所有K/V heads、RoPE维度与历史KV不变，Query按每个KV组等数保留，并配对选择Q行与O列；因此减少Q/O及attention，不能声称减少KV容量或K/V投影。仅降低FFN会撞上其余成本下界；扩展到Query有独立的计算节省空间。裁剪残差通道会牵连所有层、RMSNorm与状态布局；KV heads会改变历史状态；词表候选集有漏掉正确token的风险，本轮未实施这些路径。

## 速度

Termux SSH持锁串行，CPU6/HTP4，context4608、H4096/P40、F16 KV，KV offload关闭、FA关闭，OPBATCH1024/OPQUEUE16。每条件独立进程，完整prefill后才裁剪，同一context decode；40秒纯decode预热，3×16固定token。初始化、历史恢复、prefill、打包与检查点验证不计入decode。恢复原成本表的同模型/同token/context/batch H4096快照并逐字节检查；首组CPU重新建立历史。未改共享库、设备设置或频率。

| 条件 | ms/token中位数 |
|---|---:|
| CPU前置/后置 | 217.33 / 193.41 |
| 固定FFN保留12.5% | 201.55 |
| 固定FFN保留6.25% | 189.66 |
| 固定FFN12.5%＋Query50% | 144.16 |
| 固定FFN12.5%＋Query25% | 125.16 |
| prefill先验FFN12.5%＋Query50% | 137.45 |
| prefill先验FFN12.5%＋Query25% | 121.20 |

先验条件另批串行，不能宣称比固定选择有确定速度收益；对CPU前后中位数范围，分别约1.41–1.58倍及1.60–1.79倍。所有正式窗口新增major faults为0、swap为0；CPU有明显前后漂移，原始重复均保留。先验采集在P40完整prefill读取FFN SwiGLU输出及各Query head attention输出的平方能量，FFN按256通道块选前12.5%，Query在每个KV组内选前2或1个head。仅使用当前prefill，不读未来token；选中的索引保存于响应JSON。保持Q4_0原量化块，不重新量化。

FFN与Query打包合计约3.6–3.7秒，先验观察的完整P40 prefill约3.8秒；后者包含原有prefill，不是独立测得的增量。回调关闭后执行计时；权重buffer在进程内修改，不改GGUF。所有条件裁剪前正常响应及裁剪后有限logits、检查点续跑的最终状态逐字节一致；完整CPU另验JSON/记忆。稀疏模型没有通过JSON或记忆质量验收。

## Greedy接受率

速度筛选通过后，使用同一原始轨迹以Qwen3重新分词得到的参考续写。锚点0/8/16/24/32/48/64/96：恢复相同完整prefill检查点，逐token输入参考前缀，再让稀疏模型自由greedy生成最多8 token。CPU完整模型在同样锚点逐token喂入候选，比较下一token argmax；第一次不匹配后该块的连续接受前缀不再增长。这里是greedy前缀接受率，不是采样拒绝接受率，且参考轨迹不是CPU自由生成轨迹。

两个先验候选各8块、64个提议token，连续接受token均为0，平均每块接受长度均为0。首token全部不匹配。完整CPU自生成/自验证对照在锚点0和16均接受8/8，共16/16，支持验证的token位置对齐正确。不能把固定token的可重复logits或checkpoint通过当作质量通过。该选择准则忽略输出权重、跨层误差累积和子网络适配；每层同时裁掉87.5% FFN及50%/75% Query是很强的近似，原始激活能量不足以保证预测保真。

当接受前缀为0，草稿只增加计算与打包成本；本轮没有必要进一步跑完整投机E2E。一般串行近似需满足E[L]+1 > (k·t_d+t_v(k)+状态开销+摊销准备成本)/t_c，其中L是接受前缀长，t_d是草稿单token成本，t_v是目标批验证成本。不能只要求t_d<t_c；CPU attention仍参与HTP草稿，还需考虑与CPU验证争用，不能当纯NPU并行。

下一步更合理的是保守/分层保留、按输出贡献加权的通道选择和轻量校准或蒸馏，再沿速度→接受率顺序筛选。当前结果不支持直接把这两个激进候选接到在线draft。

## 复现

构建`python3 scripts/build/decode_attribution.py --tag build-qwen3-prefill-prior-v2`，依次使用`configs/qwen3-expanded-sparse.json`、`configs/qwen3-prefill-prior-speed.json`、`configs/qwen3-prefill-prior-drafts.json`、`configs/qwen3-prefill-prior-verify.json`运行`scripts/run/decode_attribution.py`；verify配置中的草稿必须由对应draft响应产生，不应替换为其他模型token。`scripts/analyze/qwen3_expanded_sparse.py`汇总速度和接受率。模型、快照、构建和原始结果复用本地results/shared，不纳入Git。
