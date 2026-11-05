# 首批 Observation 与系统收益图

**计时口径更新（2026-11-04晚间）：只读设备探针中 steady_clock 经过5.00秒，同机 CLOCK_BOOTTIME/实时时钟经过14.34秒。旧记录的 steady_clock 排除系统挂起，不能保证覆盖完整用户等待；旧数值保留为 active-clock 诊断。新主图统一使用包含挂起的 CLOCK_BOOTTIME，并保留双时钟对照，不改手机设置。探针见 results/20261104-observation-main/clock-check。**

首轮优先复用一次统一矩阵制作五张图，不加入 KV tiles、执行计划准备、自预热或压缩。原始图前的数据校验由 `scripts/plot/observations.py` 完成。

## 最小采样方案

这是所选覆盖与三次重复要求下的精简方案，不是任意论文结论的统计下限。两个 H 对应两次原生进程启动，所有手机模型条件串行：

| 维度 | 固定或扫描 |
|---|---|
| H | 2048、4096 |
| P | 16、24、31、32、33、40、48、56、64、128、256、512、1024 |
| 共同预算 | ctx6144、batch128、相同 CPU/HTP 上下文与 F16 KV，无额外历史 tiles |
| 后端 | 显式 hybrid；CC=CPU prefill/CPU decode；HC=HTP prefill/CPU decode |
| 全域扫描 | 每个 H/P 在 G目标200ms，测不提前准备 CC 和提前准备 HC，各3次 |
| 系统条件 | 每个 H 的 P24/31/32，G目标0/50/200ms，3种准备策略×CC/HC/Router，各3次 |
| 独立校准 | 每个 H/P 的 CC/HC 各3次，生成同一完整 JSON 至 EOG，HC 按 cosine≥0.99 准入，NRMSE仅记录 |

全域扫描和系统条件相交时只测一次。每个 H：13×2×3=78次校准；其余10个P×2路径×3=60次扫描；3个系统P×3个G×9种重放×3=243次。合计每批381，两批**762次响应试验=156次校准+606次正式测量**。若所有13个P都跑完整三G九重放，则需2262次，精简方案减少1500次（约66%）。同组数据同时提供交界、分解、Gap、策略及Oracle，不为五张图分别启动五套实验。

两条 H 曲线只能证明这两个历史长度的对照。13点是稀疏扫描，不等同于每个整数；存在更远交界或非单调变化时再加点。三次重复用于筛查，不能给微小收益作统计定论。

## 五策略定义与计时

- CC：不提前准备，全部 CPU。
- HC：返回后准备 HTP 历史，HTP prefill 后交接 CPU decode。
- Post-tool Routing：返回后根据独立校准的 CC 与 HC+准备成本选择，可等待按需准备。
- Always Prepare：发起 Tool 时准备，返回后根据实际 P、ready 和独立校准选路。
- Cost-aware：发起时根据所有校准 P 是否有潜在收益决定准备，返回后按实际 P 和 ready 选路。

后两策略沿用现有实现。当前规则只要校准域有一个 HC 可获益，就会提前准备，因此可能和 Always Prepare 实际完全重合；这是应报告的结果，不能通过改名制造策略差异。Post-tool 对照专用开关 `post_tool_routing` 只改变 after_return/router：把独立 HC 准备中位数计入校准成本，并允许选中后执行准备。设备记录的 ready_at_return 仍保持事实值，不伪报 ready。

所有主策略的完整延迟统一从设备发起 HTTP 到完整响应发布。初始化、共同历史建立与 Tool 前状态恢复不在该段，初始化/历史时间另列。响应后回收独立记录。Gap 图只能用 actual_gap_ms；服务器目标只用于分组。对照使用同一 H/P 输入和内存分配预算，实际 HTTP Gap 有波动，串行反事实不是同轮配对真值。

## 运行

```bash
# 地址需使用手机当前 adb 端口；先检查 adb devices -l，无设备直接失败。
python scripts/profile/oracle.py --device 10.29.131.50:42369 --config configs/observation-main-h2048.json
# 第一批结束且无模型占用后才启动第二批
python scripts/profile/oracle.py --device 10.29.131.50:42369 --config configs/observation-main-h4096.json
python scripts/plot/observations.py results/<第一批> results/<第二批> --output results/<具名主图目录>
```

每批保留 checked-rows.jsonl、设备原始trials、HTTP正文hash、同一设备时钟事件、完整响应与ID、数值校准、源文件及二进制hash。出图前核验完整矩阵、原始记录一致性、输出、计时恒等式、发布顺序和数值准入；HC失败按整个H/P排除。图像和独立历史记忆验收不属于该文本微基准。

选路 Oracle 只在准备状态匹配时比较，无法匹配记空值。联合 Oracle 从包括实际策略轨迹的完整实测候选取经验最小值，明确允许等待；不通过阶段拼接构造重叠收益，也不与严格 ready Oracle 混为一谈。
