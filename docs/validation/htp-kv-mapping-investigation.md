# HTP KV offload 映射失败调查（2026-11-13）

当前证据优先指向单会话下的FastRPC映射资源及布局问题，尚未确认本机底层errno。不能由此断言设备不支持KV offload；也不能只用3.5GB DSP虚拟地址上限解释所有失败。

## 本机已确认事实

Qwen3-8B Q4_0、context4608、batch128、F16 KV；入口显式只传HTP0，Hexagon默认NDEV=1。HTP权重REPACK分配约3726.02MiB。KV offload开启后，648MiB KV共享缓冲区的rpcmem分配和FD获取完成，随后fastrpc_mmap返回0x1。FA=AUTO和关闭都失败；全新HTP-only进程也失败。context512时72MiB KV映射成功，后续25431040字节计算缓冲区映射失败。

原始日志见results/20261113-qwen3-8b-key-costs-default-attention/、results/20261113-qwen3-default-attention-htp-direct/、results/20261113-qwen3-default-attention-short-smoke/、results/20261113-qwen3-8b-key-costs-kv-offload-fa-off/。失败均已清理，不启动473次全量。

## 三条关键证据

1. 当前wins缓存源码自带README.md第228–230行建议：默认单会话，8B使用两个会话。其developer.md说明每个Hexagon PD约3.5GB映射窗口，通过多个虚拟会话按层分割大模型。我们的Engine仅有devices={HTP0,nullptr}，没有使用该路径。现行官方仍提供[单物理NPU、多虚拟会话分层示例](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md#2-layer-split-mode-across-virtual-sessions-htp0htp1-or-htp00htp01)。这是一条待本机验证的可行方向，不是成功保证。

2. Qualcomm把[错误1定义为General failure](https://github.com/qualcomm/fastrpc/blob/development/inc/AEEStdErr.h#L24-L27)。[fastrpc_mmap_helper实现](https://github.com/qualcomm/fastrpc/blob/development/src/fastrpc_mem.c#L463-L537)把多数ioctl映射错误归并为1，具体errno仅在FARF日志中。现有错误不能单独证明ENOMEM、DSP VA耗尽或权限错误。Termux SSH按FastRPC标签读取logcat为空，底层errno仍未拿到。

3. 官方[issue #26123](https://github.com/ggml-org/llama.cpp/issues/26123)中2026-10-14的报告在QCS9075/v73 Linux复现相同错误，内核跟踪定位到IOVA allocator失败。映射大小、向2次幂对齐与碎片使失败早于理论容量；减小映射块等配置能改善。该报告支持本机候选机制，但不是v75 Android同根因的证明。[PR #30534](https://github.com/ggml-org/llama.cpp/pull/30534)只修VMEM=0探针，不能当作我们默认路径的直接修复。

## 必须区分两层映射

DSP虚拟地址窗口与SMMU/IOVA窗口不是同一个资源。FASTRPC_MAP_FD_DELAYED延迟的是DSP端HAP_mmap，不能据此假设没有消耗SMMU映射。参见[Qualcomm flags定义](https://github.com/qualcomm/fastrpc/blob/development/inc/remote.h#L619-L626)与[驱动实现](https://android.googlesource.com/kernel/msm.git/+/91fde529708585fcda4f7a8da34f5eb9da9b163a/drivers/char/adsprpc.c)。权重总分配字节数也不能直接等同于同时驻留的DSP VA。

缓存版本host alloc立即注册每个缓冲区；当前master实现已改变非pinned缓冲区映射时机并增加其他机制。不能把新版文档能力直接视作已安装库行为。缓存没有.git且build-info为unknown，本次保存文件及SHA256，未臆造版本号。

## 后续验证顺序

先在私有入口记录映射尺寸、flags、顺序和能获取到的底层errno；检查HTP1是否能创建。优先验证NDEV=2并显式选择HTP0与HTP1、按层拆分，保持模型/H4096/F16 KV/offload开启；这仍是一颗物理NPU，结果需记录为双虚拟会话条件。仅设置NDEV=2而Engine仍只选择HTP0不会实现分层。

另以单会话独立对照缩小MBUF，检验布局/碎片假设。缓存版本MBUF用整数MiB解析，不能照抄新版的128M,128M语法；调整时考虑额外4KiB guard page以及单个tensor不可切分的限制。OPBATCH1024/OPQUEUE16也需作为独立映射压力对照，不能悄悄改掉再与旧成本混合。

两个方向均先做JSON/记忆/状态恢复准入，成功后才开启同款473次扫描。无需先降低HTP层数或关闭KV offload；降低层数是后续候选，不是已证明唯一解。调查阶段未修改共享库、手机设置或执行新模型实验。

本次源码快照与哈希记录在results/20261113-htp-kv-mapping-investigation/。


## 补测与暂停（2026-11-13）

双会话配置`qwen3-8b-kv-dual-htp.json`失败：HTP0成功，HTP1在open阶段返回0x200。128MiB映射块配置`qwen3-8b-kv-htp-mbuf128.json`失败：KV已拆块，约126MiB块仍在fastrpc_mmap返回0x1。两轮均保存failure-cleanup.json，服务停止并释放锁。

进一步调查：0x200为AEE_ERPC通用FastRPC错误，不能直接断言权限或硬件不支持多会话。v75有Linux多会话运行记录，但未找到同小米14Pro/Termux路径的已解决case。上游64位DMA扩展PR29197只适用于v81+，不能直接解决v75。参考[Qualcomm同代手机相似问题](https://github.com/qualcomm/ai-hub-models/issues/85)、[v75多会话记录](https://github.com/ggml-org/llama.cpp/issues/30309)、[DMA扩展](https://github.com/ggml-org/llama.cpp/pull/29197)。

[Termux vendor路径报告](https://github.com/ggml-org/llama.cpp/issues/27677)提供了新线索，但本机去vendor目录的模型准入被现有库的libOpenCL.so依赖挡在链接阶段。无模型会话探针记录在results/20261113-htp-session-probe/：带vendor路径单会话成功、双会话重现HTP1不可用；不带vendor路径两种条件均启动失败。不能把这次clean条件视作成功执行了FastRPC对照，需以后先解决驱动库加载依赖才可比较。

用户要求停止排查。本次探针已自然结束且cleanup.json确认释放设备锁和wake lock；不再启动试验，473次新扫描未启动。既有KV offload关闭的CPU/HTP历史结果保留，不据此宣称其为硬件上限。

## 清理记录（2026-11-13）

已移除主线中的多会话、映射块、vendor路径和FA AUTO专用改动；原差异、六份配置、会话探针及专用测试归档至`archive/diagnostics/20261113-default-htp-abandoned/`。原始失败日志仍在results；未改共享运行时。清理前SSH复核无模型进程及模型锁。后续验证顺序仅为当时的候选记录，当前没有待运行任务。
