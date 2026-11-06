#set document(title:"Xiaomi 14 Pro：Observation与系统收益实验报告",author:"A2O 实验记录")
#set page(paper:"a4",flipped:true,margin:(x:10mm,top:10mm,bottom:12mm),footer:context [#set text(size:8pt,fill:rgb("#64748b")); #grid(columns:(1fr,auto),[A2O · Xiaomi 14 Pro · 2026-11-05],[#counter(page).display("1") / 7])])
#set text(font:("PingFang SC","Arial"),size:10pt,lang:"zh",fill:rgb("#172b45"))
#set par(spacing:0pt,leading:4pt,justify:false)
#set block(spacing:0pt)
#let ink=rgb("#172b45")
#let muted=rgb("#526579")
#let blue=rgb("#0072b2")
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[Xiaomi 14 Pro｜Observation 与系统收益]\
#text(size:8pt,fill:muted)[2026-11-05 · SM8650 / Hexagon v75 · 固定文本 HTTP 微基准 · 每条件 3 次重放]]
#v(2mm)
#v(2mm)
#grid(columns:(1fr,1fr,1fr),gutter:4mm,
  block(fill:rgb("#eef4fa"),inset:4mm,width:100%)[#text(size:20pt,weight:"bold")[762] 条主矩阵记录],
  block(fill:rgb("#eef4fa"),inset:4mm,width:100%)[#text(size:20pt,weight:"bold")[26 / 26] 组 HC 数值准入],
  block(fill:rgb("#eef4fa"),inset:4mm,width:100%)[#text(size:20pt,weight:"bold")[5 / 5] 张图验收通过])
#v(5mm)
#text(weight:"bold",size:12pt)[核心判断]
#v(2mm)
#text(size:10pt)[CPU/HTP 的优势区域及 P32 阶跃清晰。提前状态准备能够与真实 Tool 等待重叠，但当前 Cost-aware 尚未稳定优于 Always Prepare 或 Post-tool Routing；本轮支持性能表征，不能直接宣称新的稳定系统增益。]
#v(4mm)
#text(weight:"bold",size:11pt)[策略完整响应中位数（ms；服务器 G 目标为 200ms）]
#v(2mm)
#text(size:9pt)[#table(columns:(28mm,1fr,1fr,1fr,1fr,1fr,1fr),inset:(x:2.5mm,y:1.8mm),stroke:0.35pt+rgb("#d8e0e8"),align:(x,y)=>if x==0 {left} else {right},fill:(x,y)=>if y==0 {rgb("#eef3f8")} else {none},
table.header(text("H / P",weight:"bold"),text("CC",weight:"bold"),text("HC",weight:"bold"),text("Post-tool",weight:"bold"),text("Always Prepare",weight:"bold"),text("Cost-aware",weight:"bold"),text("相对 CC 缩短",weight:"bold")),
text("2048 / 24"),text("2125.0"),text("2614.0"),text("2135.7"),text("2159.0"),text("2182.2"),text("-2.7%"),
text("2048 / 31"),text("3039.0"),text("3052.3"),text("2596.8"),text("3016.2"),text("3041.0"),text("-0.1%"),
text("2048 / 32"),text("2481.7"),text("1945.8"),text("1968.9"),text("1919.4"),text("2039.4"),text("+17.8%"),
text("4096 / 24"),text("2400.2"),text("3932.3"),text("2421.3"),text("2403.6"),text("2417.3"),text("-0.7%"),
text("4096 / 31"),text("3370.3"),text("4680.9"),text("2996.4"),text("3393.0"),text("3369.0"),text("+0.0%"),
text("4096 / 32"),text("2835.7"),text("2198.2"),text("2202.3"),text("2102.3"),text("2078.9"),text("+26.7%"),)]#v(3mm)
#text(size:9pt)[CC = CPU prefill + CPU decode；HC = HTP prefill + CPU decode。所有策略从设备发起 Tool 到完整 JSON 发布统一计时，使用相同输入与内存分配预算。负百分比表示 Cost-aware 更慢；组间实际 G 有波动，非同轮严格配对。]
#v(4mm)
#text(size:9pt)[矩阵：H = 2048 / 4096；P = 16、24、31、32、33、40、48、56、64、128、256、512、1024。156 次独立校准 + 606 次正式测量；另有 18 次新平台预检，不进入五张主图。]
#v(3mm)
#text(size:9pt)[阅读顺序：第 2-4 页为 Observation，第 5 页为完整策略比较，第 6 页为 Oracle Gap，第 7 页列出配置、计时与验收边界。每个条件只有 3 次重复，不能以小幅差异替代统计证据。]

#pagebreak()
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[01 / CPU/HTP 性能交界]\
#text(size:8pt,fill:muted)[全域 P 扫描及 31 / 32 / 33 细节；横轴 P，纵轴为 Tool 返回后的完整响应。]]
#v(2mm)
#block(height:111mm,width:100%)[]
#v(3mm)
#text(size:9pt)[固定服务器 G 目标 200ms；线为各点中位数，浅色点为三次原始测量。实线为 CC，虚线为提前准备的 HC。H 仅取两个历史长度；P 为 13 点稀疏扫描，并非每个整数。]
#v(3mm)
#text(size:10pt)[可观察到 P32 附近的非平滑阶跃；同时，全域扫描显示延迟继续随大 P 增长，不能把 31/32 的核行为当作整个性能曲面的规律。H2048/P31 两路径接近，不应将小差异夸大为稳定边界翻转。]

#pagebreak()
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[02 / Cost Breakdown：优势区域为何不同]\
#text(size:8pt,fill:muted)[阶段墙钟分解；每根柱采用阶段均值，黑点为三次完整响应。]]
#v(2mm)
#block(height:105mm,width:100%)[]
#v(3mm)
#text(size:9pt)[Observation 为观察处理；Blocking preparation 只计 Tool 返回后的阻塞准备；State handoff 包含状态迁移与检查。后台总准备不能再次加到响应堆叠中。Other 为同次完整时间减已列阶段的余额。]
#v(3mm)
#text(size:9pt)[#table(columns:(22mm,55mm,1fr),inset:(x:2.5mm,y:1.8mm),stroke:0.35pt+rgb("#d8e0e8"),align:(x,y)=>if x==0 {left} else {right},fill:(x,y)=>if y==0 {rgb("#eef3f8")} else {none},
table.header(text("H",weight:"bold"),text("准备方式",weight:"bold"),text("P24 / 31 / 32 的总准备中位数范围（ms）",weight:"bold")),
text("2048"),text("返回后准备"),text("50.8 - 51.5"),
text("2048"),text("提前准备"),text("72.1 - 76.2"),
text("4096"),text("返回后准备"),text("84.8 - 92.2"),
text("4096"),text("提前准备"),text("103.9 - 112.4"),)]#v(3mm)
#text(size:9pt)[Prefill 主导大部分后端差异。上表总准备可被等待窗口隐藏，但隐藏成功不等于把准备耗时全部转化为完整响应收益；初始化和共同历史构建另列，不计入每轮 Tool 延迟。]

#pagebreak()
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[03 / Tool Gap 重叠收益]\
#text(size:8pt,fill:muted)[横轴使用设备 CLOCK_BOOTTIME 测得的实际 G，绝不使用 HTTP 服务端目标代替。]]
#v(2mm)
#block(height:114mm,width:100%)[]
#v(3mm)
#text(size:9pt)[固定 P32、HC 路径；灰色在 Tool 返回后准备，蓝色在 Tool 等待期间准备。点为每次实际 G 与实际返回后响应；线连接各目标组的实测中位数。两种策略的实际 G 并不完全相同。]
#v(3mm)
#text(size:10pt)[提前准备在这些样本中降低了部分返回后等待，但幅度随 H / G 条件波动。仅有 0 / 50 / 200ms 三个服务端目标组、每组 3 次，不能据此宣称精确的隐藏阈值或稳定的单调关系。]

#pagebreak()
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[04 / End-to-End 策略比较]\
#text(size:8pt,fill:muted)[完整响应 = 设备发起 Tool 至完整 JSON 发布；每个 H/P 的五策略共享输入和预算。]]
#v(2mm)
#block(height:158mm,width:100%)[]
#v(3mm)
#text(size:9pt)[服务器 G 目标为 200ms；点为三次原始测量，横线为中位数。Post-tool 按包含准备的独立校准选路；Always Prepare 与 Cost-aware 均使用实际 Router。当前证据未证明 Cost-aware 稳定优于两者。]
#v(3mm)
#text(size:8pt)[]

#pagebreak()
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[05 / Oracle Gap：仍有多少调度空间]\
#text(size:8pt,fill:muted)[左：同准备状态下的严格选路；右：允许等待的完整联合重放。两幅纵轴计时起点不同。]]
#v(2mm)
#block(height:108mm,width:100%)[]
#v(3mm)
#text(size:9pt)[严格候选只包含状态匹配且可用的路径；灰色叉号表示无法识别。联合候选包括全部九路实测重放及实际策略本身，保留候选与所选 trial ID，不拼接阶段最小值。]
#v(3mm)
#text(size:9pt)[#table(columns:(65mm,95mm,1fr),inset:(x:2.5mm,y:1.8mm),stroke:0.35pt+rgb("#d8e0e8"),align:(x,y)=>if x==0 {left} else {right},fill:(x,y)=>if y==0 {rgb("#eef3f8")} else {none},
table.header(text("Oracle 口径",weight:"bold"),text("本轮范围",weight:"bold"),text("计时起点",weight:"bold")),
text("严格选路 Oracle"),text("16 / 18 个条件状态匹配"),text("Tool 返回至完整响应"),
text("联合经验 Oracle（可等待）"),text("最大观测差距 666.3 ms"),text("Tool 发起至完整响应"),)]#v(3mm)
#text(size:9pt)[联合差距最大出现在 H4096/P32、G 目标 0ms。该条件的 Cost-aware Router 三次返回时均未 ready，因而走 CPU；允许等待的 HC 重放更快。这是后续核实短 Gap / pending 候选准入的线索，不是已经获得的加速；实际 G 波动与三次中位数取最小带来选择偏差。]

#pagebreak()
#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)[方法、验收与复现边界]\
#text(size:8pt,fill:muted)[本报告为固定文本微基准结果，不包含新增图像任务、外网 Tool、能耗或通用 Agent 正确率结论。]]
#v(2mm)
#text(size:9pt)[#table(columns:(44mm,1fr),inset:(x:2.5mm,y:1.8mm),stroke:0.35pt+rgb("#d8e0e8"),align:(x,y)=>if x==0 {left} else {right},fill:(x,y)=>if y==0 {rgb("#eef3f8")} else {none},
table.header(text("项目",weight:"bold"),text("设置或验收标准",weight:"bold")),
text("模型 / KV"),text("Qwen2.5-VL-3B-Instruct Q4_0 / F16 KV"),
text("设备 / 后端"),text("Xiaomi 14 Pro，SM8650，Android 14，Hexagon v75；显式 hybrid"),
text("共同预算"),text("context 6144；batch 128；HTP queue / opbatch 均为 1"),
text("策略采样域"),text("H2048 / 4096，P24 / 31 / 32；G 目标 0 / 50 / 200ms；三策略 × CC / HC / Router"),
text("关闭的机制"),text("KV 整理、通用预搬移、CPU repack；无 tiles、执行计划、自预热或在线自校准"),
text("响应与资格"),text("相同完整 JSON 与 6 个生成 ID（含 EOG）；cosine 至少 0.99，NRMSE 仅记录"),
text("统一主时钟"),text("CLOCK_BOOTTIME，包含系统挂起；steady_clock 仅作为诊断"),
text("双时钟读取抖动"),text("允许 0.1ms 非原子读取差；主时钟恒等式仍为 0.01ms，原值不改写"),)]#v(4mm)
#text(size:9pt)[#table(columns:(35mm,1fr,1fr),inset:(x:2.5mm,y:1.8mm),stroke:0.35pt+rgb("#d8e0e8"),align:(x,y)=>if x==0 {left} else {right},fill:(x,y)=>if y==0 {rgb("#eef3f8")} else {none},
table.header(text("H",weight:"bold"),text("模型 / 上下文初始化（s）",weight:"bold"),text("共同历史构建（s）",weight:"bold")),
text("2048"),text("12.35"),text("65.22"),
text("4096"),text("17.52"),text("181.02"),)]#v(3mm)
#text(size:9pt)[共同初始化、历史建立和 Tool 前恢复不计入每轮完整响应；观察处理、准备、prefill、交接、decode 与响应后回收分别记录。两批在同一设备串行运行，逐条件完整重放；不执行模型输出动作，不改变手机设置、CPU 频率或用户数据。]
#v(3mm)
#text(size:9pt)[数据验收：完整矩阵、原始记录一致性、HTTP 正文 hash、JSON / 完整 ID / EOG、迁移字节核验、发布顺序、主时钟恒等式和二进制来源检查均通过。26/26 个 H/P 条件接纳 HC；没有挑选成功子集。图形几何与视觉检查已完成。]
#v(3mm)
#text(size:8pt)[本地复现链：两份原始运行目录 → figures/plot-data.json、comparisons.json、measurements.csv → 五张 PNG/PDF → 本汇总。原始目录：20261105-165933-954653-http-oracle-grid；20261105-173045-235354-http-oracle-grid。原始行、失败出图日志、验收记录和本报告源文件均保留。]
