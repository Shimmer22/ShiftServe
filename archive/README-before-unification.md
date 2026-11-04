# 2026-11 A2O gap profiling（动作→观察间隙 / 单步 VLM 耗时拆分）

## 目标
完整目标路线与代码核查见 [PIPELINE_REVIEW.md](PIPELINE_REVIEW.md)：**手机 CPU decode → A2O 内压缩历史 KV 并使 HTP 可访问 → 新截图后 HTP prefill → KV/状态同步回 CPU → CPU decode**。当前 EXP-009 只跑通 HTP0 状态处理子链路，尚未实现这个跨后端循环。

移动端 GUI Agent 每步动作后需等 UI 稳定才能截图。该空窗（A2O gap）可能可用于 KV 管理与 NPU 预备。
首个动机实验：
- **M1**：A2O gap 时长分布。
- **M3**：单步 VLM 推理拆分为 ViT 视觉编码 / prefill（图像 token、文本 token）/ decode。

## 状态一览（诚实声明）
| 项 | 状态 |
|---|---|
| M1 工具链（设备端脚本、编排、分析、作图） | 已写好；分析逻辑用合成数据做了逻辑校验（`scripts/test_analyze_synthetic.py`，非实测） |
| M1 **真实测量** | **未完成，被阻塞**：手机息屏锁屏（`mWakefulness=Dozing`，keyguard 显示），且 `adb shell input` 被 `INJECT_EVENTS` SecurityException 拒绝（HyperOS「USB 调试(安全设置)」未开）。我没有也不应绕过/修改这些设置。→ **无 M1 的 mean/median/p90 数据** |
| M3 | 已实测（见下），CPU-only |
| EXP-007 KV prep 拷贝上界 | 已完成（host+phone userspace memcpy；非 DMA/非真实 prune） |
| EXP-008 KV keep(A)+任务关注度+merge | 已完成最小 host 试跑（**合成 KV**；NPU DMA=stub） |
| EXP-009 真实 KV dump→策略→写回→HTP0 prefill | **已跑通**（文本多段模拟历史；merge=exemplar-keep；NPU DMA=stub；压缩后生成质量未验证，见 EXP-009 README） |
| EXP-010 keep-all / 原策略诊断 | 两组均立即 EOG，故原 EOG 不能归因于压缩；见 EXP-010 README 与 PIPELINE_REVIEW.md |

## 设备
Xiaomi 15（24129PN74C，SM8750，Android 16），无线 adb `10.129.178.171:45021`，屏幕 1200x2670。全程未改 CPU 频率/调速器/系统设置。
测量期间手机负载很高（`/proc/loadavg` 1 分钟值约 14–27，内存 MemAvailable 约 4–5GB，swap 已用约 4GB），**M3 时间含此干扰，偏大且有噪声**。

## 复用来源（引用，未拷贝）
- `shared/runtime/llama.cpp` 的 Android arm64 构建，手机上已存在的 `/data/local/tmp/llama-mtmd-cli`（sha256 与 Mac 侧 `build-android/bin/llama-mtmd-cli` 一致 `c5c78a52…ea24`）。
- 手机上已有 `models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`、`models/mmproj-model-f16.gguf`（与 `shared/models/` 同 sha256 记录）。未下载任何新文件。
- 输入截图：`2026-10-qwen38-prune-launch/results/androidworld-setup-20261024-v1/home.png`（AndroidWorld 模拟器主屏，1080x2400，真实移动端 UI）与同目录体系下 ScreenSpot-v2 桌面截图 `pvst-thinking-screenspot-inputs-20261017-v2/originals/pc_*.png` 前 2 张。
- 文档参考：`shared/docs/Qwen2.5-VL-Android-部署指南.md`、`…-vit-timing-patch.md`，`2026-10-qwen38-prune-launch/docs/PHONE_RUNTIME.md`。

## EXP-001 M1 方法（`EXP-001-m1-gap/`）
- `scripts/a2o_trial.sh`（在手机上运行）：注入前取基线帧 → 注入动作（记录注入起止时间）→ 连续 `screencap | md5sum`，记录每帧采集起止时间（同一时钟 `date +%s%N`），直到连续 N=3 帧相同且与基线不同。
- 稳定判定：首个 k 使 md5[k..k+2] 相同且 ≠ 基线。最后一次变化落在 F(k-1) 采集与 F(k) 采集之间：
  - `lower = s(k-1) − t_inj_end`；`upper = e(k) − t_inj_end`（= 轮询截图的 Agent 实际观察到的等待，**含截图开销**）；`mid = (lower+upper)/2`（去除截图开销后的点估计，误差 ≤ (upper−lower)/2）。
  - 截图开销单独记录为每帧 `capture_end − capture_start`（`mean_capture_ms`），与真实等待分开报告。注入开销 `inject_ms`（`input`/`am` 进程启动）亦单独记录。
- 动作类（仅导航/查看，不改设置值）：`open_settings`、`scroll_down`、`scroll_up`、`open_submenu`（用 intent 打开「显示」设置页，仅查看）、`back`、`home_from_app`、`recents_open`、`recents_close`；每类计划 ≥20 次。
- 未采用 SurfaceFlinger `--latency` / gfxinfo framestats 作主方法：对空闲 launcher 层仅返回刷新周期、无帧；其时钟(MONOTONIC)与设备 realtime 另有偏移。作为后续交叉验证项。
- 运行：`python scripts/run_m1.py --preflight`（须通过）→ `python scripts/run_m1.py --run 20` → `python scripts/analyze_m1.py results/m1_raw_*.json --out-prefix results/m1`（输出 CSV/summary/分布图 png）。
- 当前仅有的结果文件 `results/m1_*screenoff*`：**息屏状态下的空操作冒烟测试**（5 次，注入 `:`），只证明工具链可运行；5/5 为 `no_change`，该状态下 `capture_cost` 均值 247.9ms **不代表**亮屏截图开销，不得当作 M1 结果。

## EXP-002 M3 方法与结果（`EXP-002-m3-step-breakdown/`）
- 手机上 `llama-mtmd-cli`，CPU only（`-ngl 0`，无 GPU/NPU 后端），`-t 4`，`-c 2048 -n 48 --temp 0`，每次新进程（模型加载时间单列）。固定 GUI-Agent 提示词（文本 39 token，含模板）。
- 阶段：ViT = 日志 `image slice encoded in`；图像 prefill = `image decoded … in`（n_tokens_batch=图像 token 数）；decode = perf `eval time`；文本 prefill 为**推导残差** = perf `prompt eval time` − ViT − 图像 prefill（见未验证项）。
- 输入：AW 主屏缩放到 3 种分辨率（336x756/448x980/560x1232），及 2 张桌面截图 784x476。

| step | 图像 token | 文本 token | 次数 | ViT ms(均值) | 图像 prefill ms | 文本 prefill 残差 ms | decode ms | decode token |
|---|---|---|---|---|---|---|---|---|
| aw_home_S | 324 | 39 | 3 | 11735（10346–12830） | 21222（19593–22818） | 3121 | 2691 | 27 |
| aw_home_M | 560 | 39 | 3 | 22757（22496–22891） | 38178（36655–40033） | 3300 | 2715 | 27 |
| aw_home_L | 880 | 39 | 1 | 40563 | 73988 | 3912 | 3287 | 27 |
| desk0_M | 476 | 39 | 1 | 23962 | 40157 | 4226 | 2575 | 21 |
| desk1_M | 476 | 39 | 1 | 23634 | 40645 | 4114 | 2198 | 19 |

原始：`results/m3_steps.csv`、`m3_summary.json`、`m3_runs_t4.json`、`raw/*.err.txt`；图：`results/m3_stage_breakdown.png`。

## 已验证 / 未验证
已验证：设备可连、已有资产可直接跑 Qwen2.5-VL-3B；上表数据均来自 `raw/` 日志解析；M1 分析的稳定判定逻辑在合成数据上正确。
未验证 / 局限：
1. **M1 无真实数据**（锁屏 + 输入注入被拒）。
2. M3 文本 prefill 是残差推导；唯一的纯文本对照失败（`llama-mtmd-cli` 无图像时进入聊天模式后 `selfchecker … sig6` abort，见 `raw/text_only_*`；手机上的 `llama-cli` 对照 3 分钟未返回，已终止，无数据）。perf 的 `prompt eval time` 与日志中 ViT/图像 prefill 的口径关系未用独立手段核实。
3. M3 重复次数少（S/M 各 3 次，其余 1 次），设备高负载，数字应视为上界/量级参考；线程数仅测 4。
4. 输入不是本机实拍截图（手机锁屏），移动端仅 1 张 AW 模拟器主屏；桌面图只用于不同内容对照。decode 输出内容（点击坐标）未评估正确性。
5. 未拆 ViT 内部（Patch/Layers/Projector），需 `MTMD_VIT_TIMING=1`（会使 ViT 成本翻倍，见 `shared/docs/Qwen2.5-VL-vit-timing-patch.md`）。

## EXP-003 手机作推理服务器（可行性，`EXP-003-phone-server/`）
- 仅无线 adb（`10.129.178.171:45021`）；不做点击、不改设置。
- 服务端（手机）：复用 `shared/runtime/llama.cpp/build-android/bin/llama-server`（sha256 `b521671f…49b2`，commit 58e68df，已有构建，无需编译）。临时推到 `/data/local/tmp/llama-server-a2o`，测完已删除；模型/mmproj 用手机上已有文件。启动命令：
  `./llama-server-a2o -m models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf -t 4 -c 4096 -ngl 0 -np 1 --host 127.0.0.1 --port 18080`
- Mac：`adb -s 10.129.178.171:45021 forward tcp:18080 tcp:18080`；`scripts/client.py` 用 `adb exec-out screencap -p` 抓图 → Mac 上缩放到 448x980（560 图像 token）→ `/v1/chat/completions`（base64 图像，temperature 0，max_tokens 64）。`scripts/summarize.py` 合并服务端日志的 ViT/图像解码时间。
- 结束：已 `pkill` 服务、`forward --remove`，手机上无 llama-server 进程、无遗留转发。
- 输入：3 张手机实拍（当前屏为「开发者选项」页），存于 `inputs/shot{0,1,2}.png`（缩放后）。

| 请求 | 截图 s | 缩放 s | HTTP 端到端 s | 服务端 prompt_ms（prompt_n） | 其中 ViT ms / 图像解码 ms | predicted_ms（n） | HTTP−服务端 s | loadavg1 前→后 |
|---|---|---|---|---|---|---|---|---|
| shot0 | 0.565 | 0.077 | 52.721 | 49639（627） | 17928 / 28173 | 2979（41） | 0.103 | 13.31→15.34 |
| shot1 | 0.452 | 0.060 | 60.743 | 56573（627） | 19487 / 32310 | 4083（41） | 0.087 | 15.34→15.94 |
| shot2 | 0.490 | 0.060 | 68.601 | 64640（627） | 22025 / 37170 | 3871（41） | 0.090 | 15.94→16.43 |
| shot2_cache0（cache_prompt=true，同图） | 0.490 | 0.050 | 4.349 | 114（1；cache_n=626） | 无（未重新编码） | 4164（42） | 0.071 | 16.43→16.39 |
| shot2_cache1（同上） | 0.490 | 0.050 | 4.813 | 113（1；cache_n=626） | 无 | 4616（42） | 0.084 | 16.39→16.36 |

原始：`results/requests.json`、`requests_table.csv`、`server.log`、`client.log`。
说明：服务端 `prompt_ms` 包含 ViT + 图像 token 解码 + 文本 prefill（shot0：49639−17928−28173=3538ms 为推导的文本部分）。截图 PNG 原图约 0.29MB（1200x2670），请求 JSON 约 184kB，经 adb forward 的传输/序列化开销 ≈ 0.07–0.10s，相对推理可忽略。
局限：仅 3 个冷请求 + 2 个缓存请求；同一次启动内 ViT 耗时逐次上升（17.9→19.5→22.0s），原因未查（可能为负载/温控，未验证）；手机 loadavg 本就 13–16。cache 请求命中是因为同一图像/提示的前缀仍在 slot 中（含上一个 `cache_prompt=false` 请求），不代表不同截图间的缓存收益。

## EXP-004 A2O（意图触发）与 NPU 可行性（2026-11-02，`EXP-004-a2o-intent-npu-feasibility/`）
### A2O：未取得有效数据（0 次 trial）
- 探测记录：`EXP-004-.../results/probe_20261102.md`。
- 无需 INJECT_EVENTS 的触发均被系统接受、无权限错误：`am start`（Settings / DISPLAY_SETTINGS / HOME / 浏览器 VIEW URL）、`cmd statusbar expand-notifications|collapse`。`cmd input tap` 仍报 SecurityException（INJECT_EVENTS）。`uiautomator dump` 可用但只读，无点击途径。`cmd accessibility` 无可用接口。
- 阻塞：手机再次处于灭屏锁屏（Dozing, keyguard, 焦点 NotificationShade）。锁屏下连续两次 screencap md5 相同（24d5ec95…），意图触发不产生可见变化，A2O 稳定判定无意义。需用户唤醒并解锁、保持亮屏后才能跑（脚本复用 EXP-001 `a2o_trial.sh` / `analyze_m1.py`，需新写意图触发编排器）。
- 偏差方向（待测，未实测）：意图触发 ≠ 点击。am start 走 ActivityManager，跳过触摸分发与控件响应延迟，且冷/热启动与点击同页切换差异大，预期低估点击型 A2O 的下界；通知栏展开/收起为系统动画，偏短。结论仅作方向性，不能代替真实点击。
### NPU：当前不可运行
- llama.cpp 源码含 `ggml-hexagon`（htp），但已有 `build-android` 为 `GGML_HEXAGON=OFF`，无 libggml-hexagon/libggml-htp 产物。
- 手机具备 `libQnnHtp*.so`(含 V79 Stub)、`libcdsprpc.so`、`/dev/fastrpc-cdsp*`，SoC SM8750（Hexagon v79，源码支持 v79）；`/vendor/dsp` 无读权限。
- Mac 无 QNN/QAIRT/Hexagon SDK；Docker 为 OrbStack，守护进程未运行，无 toolchain 镜像；其它实验目录无 NPU 脚本。
- 构建需 `ghcr.io/snapdragon-toolchain/arm64-android`（含 Hexagon SDK、NDK、OpenCL；docs 为 v0.8，本地源码文档为 v0.3，须 `--platform linux/amd64`，Apple Silicon 上需模拟）。镜像大小未实测（未下载）。
- 模型格式缺口：HTP matmul 仅支持 Q4_0 / Q8_0 / MXFP4（及 F16/F32 算子），本地 Qwen2.5-VL-3B 为 Q4_K_M，需重新量化为 Q4_0（`llama-quantize --allow-requantize`，有损；Mac 有 llama-quantize）。mmproj(ViT) 为 F16。
- 因需下载镜像并重编译、重量化，且未验证非 root 手机上 unsigned PD 是否可用，按约定未做；无 NPU vs CPU 对比数字。
### 清理
手机 /data/local/tmp 仅余原有文件，无后台进程，无端口转发；临时 /sdcard/x.xml 已删。

## EXP-005 A2O = 框架固定等待 + 截图开销（用户重新定义 A2O；NPU 按用户决定暂不做）
- 用户改定义：不测屏幕稳定，A2O = 框架动作后的等待 + 截图开销。EXP-004 的 intent 稳定性测量已停止，已有数据保留。NPU（下载/编译/重量化）按用户决定跳过，无数据。
- 设备：Xiaomi 15 (24129PN74C)，无线 adb 10.129.178.171:45021，屏幕 1200x2670。
- 源码等待（file:line、默认值、commit）：`EXP-005-a2o-fixed-wait/results/source_waits.md`；汇总表：`results/a2o_summary_table.md`。
- 更正前提：AndroidWorld 标准 run.py（非 MiniWoB）transition_pause=None，走条件等待（UI 树连续 3 次不变，间隔 0.5 s，上限 6.0 s），并非纯固定等待；M3A 另有动作后固定 2.0 s。
- 截图开销实测：`scripts/bench_screencap.py`，各方法 30 轮（另有 2 轮预热丢弃），Mac 端 wall-clock，含 adb 客户端启动；原始数据 `results/screencap_locked_rounds.csv`，统计 `results/screencap_locked_summary.json`。
- **局限：测量全程手机为息屏/锁屏（Dozing, isKeyguardShowing=true, 前台 NotificationShade），截图是锁屏画面：每次 PNG 都是 18650 B，md5 与 EXP-001 冒烟一致（24d5ec95…），内容近乎纯色。设备端 PNG 编码与传输量远小于真实应用界面，因此以下数字是下界。** 需解锁/亮屏后重测才能得到真实界面的开销。
- 未测：AndroidWorld（模拟器 gRPC + a11y forwarder）与 DroidRun（Portal app）的截图方式，真机不装 app 无法复现。
- 清理：手机 /sdcard 临时文件已删，/data/local/tmp 仅余原有文件，无转发、无后台进程；Mac 端 /tmp 源码克隆与临时图已删。

## EXP-007 A2O 空档 KV 整理+搬移耗时（`EXP-007-a2o-kv-prep-timing/`）
- 状态：已完成第一版。Host/phone userspace 拷贝上界（非 Hexagon DMA、非真实 prune）。GUI/ ctx4096 prep ≪ A2O 预算（最短≈1.3s）。详见该目录 README 与 `results/a2o_vs_kv_prep_table.md`。

## EXP-008 GUI KV keep(A)+任务关注度+history merge（`EXP-008-kv-keep-a-merge/`）
- 状态：已完成最小 host 试跑（**合成 KV**，非真实 dump）；action_prev 整段保留；task 按关注度加权；history k-means merge；NPU DMA=stub。详见该目录 README 与 `results/`。

