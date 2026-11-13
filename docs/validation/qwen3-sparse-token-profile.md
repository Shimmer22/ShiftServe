# Qwen3激进稀疏单token与输出质量

Qwen3-8B Q4_0，prefill能量选择保留12.5% FFN、25% Query heads。H4096/P40、context4608、CPU6/HTP4、KV offload与FA关闭，OPBATCH1024/OPQUEUE16。不是降低量化bit数。

无插桩前后中位数108.92/110.46 ms/token；细粒度中位数115.02 ms/token。每组40秒decode预热、5次正式重复，每次连续4个单token调用。表为同一正式重复的平均每token耗时。初始化、历史恢复、prefill、打包和状态验证另计。

| 部分 | ms/token |
|---|---:|
| CPU attention其他 | 0.023 |
| CPU其他节点 | 0.005 |
| CPU attention核心 | 34.264 |
| CPU输出头 | 25.527 |
| CPU KV写入 | 7.870 |
| DSP norm/残差等 | 0.198 |
| DSP Q/K/V/O投影 | 7.732 |
| DSP attention其他 | 1.131 |
| DSP FFN | 9.884 |
| CPU派发/线程池/插桩差额 | 15.841 |
| HTP提交/缓存/等待/profile差额 | 11.939 |
| 显式张量copy | 0.051 |
| 外层显式同步 | 0.074 |
| 剩余主机工作 | 0.485 |

CPU外层83.53 ms，HTP调用30.88 ms；CPU38段、HTP37段，DSP每token37批提交。代表性重复的4个token为130.60/108.52/107.28/113.69 ms。内层节点不能再与外层执行段相加。CPU节点含既有barrier等待，KV写入在5次重复间波动5.8–17.7 ms，不能全当复制时间；CPU派发差额不能全叫OS调度；HTP差额不能全叫RPC。显式copy很小不等于无数据搬运。未测DDR带宽或缓存miss，矩阵时间含读取、解量化与计算。诊断中位数比控制高约4.6%–5.6%，不将比例精确套到无插桩。正式窗口major fault=0、swap=0。

FFN仅约9.9 ms，占8.6%；即使完全移除，也只能从这份分解再省约9.9 ms。主要剩余成本在CPU attention、大词表输出头及调度。仍是CPU/HTP混合。

## 自由生成样例

每题最多48 token，greedy。稀疏索引来自原任务P40先验，清空状态后用于新提示；不是针对每道新题重新选择先验，因此该测试检验固定子网络迁移到新提示的质量。此前原任务锚点接受率也为0，是独立证据。

### cpu-full

提示：用一句中文解释为什么天空看起来是蓝色的。
输出：`因为大气中的分子会散射太阳光中波长较短的蓝光，使蓝光向各个方向传播，因此我们看到的天空呈现蓝色。`
正常结束：True

提示：计算 17+26，只输出数字。
输出：`43`
正常结束：True

提示：请只输出JSON：{"code":314159}
输出：`{"code":314159}`
正常结束：True

### sparse-query50

提示：用一句中文解释为什么天空看起来是蓝色的。
输出：` Lub Lub Lub Lub Lub Lub Lub Lubutautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautautauta`
正常结束：False

提示：计算 17+26，只输出数字。
输出：` Lub Lub Lub Lub Lub Lub Lub Lub Lub浔浔浔浔浔浔浔浔 Lub浔浔 Lub浔 Lub浔 Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub Lub`
正常结束：False

提示：请只输出JSON：{"code":314159}
输出：` Lub Lub Lub hé Lub hé hé Lub hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hé hénoinspection`
正常结束：False

### sparse-query25

提示：用一句中文解释为什么天空看起来是蓝色的。
输出：` HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF HF`
正常结束：False

提示：计算 17+26，只输出数字。
输出：`批发市场批发市场批发市场批发市场批发市场批发市场批发市场晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦晦`
正常结束：False

提示：请只输出JSON：{"code":314159}
输出：`escapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescapeescape`
正常结束：False

完整CPU解释通顺、算术43正确、JSON有效；两组稀疏输出均为重复碎片，三题均不合要求且不正常结束。不能作为有效draft。
