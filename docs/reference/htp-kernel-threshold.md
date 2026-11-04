# Hexagon P32 性能阶跃与路由验证

2026-11-04，用户要求 Luna 只读调研。主实验继续串行运行，调研不启动设备模型或更改手机设置。

本轮 neutral_padding_v1 独立校准中，H4096/P24 的 HTP prefill 中位数约 2526ms，P32 约 1017ms，P40 约 1056ms；完整输出及生成 ID 一致，D=6，准备仅约几十毫秒。阶跃位于 prefill，不能归因于隐藏状态准备。

shared 源码 `../shared/runtime/llama.cpp/ggml/src/ggml-hexagon/htp/matmul-ops.c:2947` 的 `op_matmul` 存在对应候选机制：

- HMX 编译/运行时启用；权重行数 N 按 32 对齐，Q4_0 等量化权重要求 K 按 256 对齐，并满足布局等条件。
- `m_total = src1->ne[1]`，`m_hmx = m_total & ~31`。M<32 直接执行 `op_matmul_hvx`，M≥32 才进入 HMX 路径（2994–3001行）。
- 部分 Qwen 投影/FFN 矩阵的激活行数对应输入 token 数 P，因此预测 P31 慢、P32/33 快。
- HMX 的 pipeline 另有 M≥128 门槛，不适合解释 P24→32。

这是性能路径切换的源码候选解释，尚不能认定为 kernel 正确性缺陷，也不能仅由总延迟确定具体算子。shared 源码、预编译包和手机安装库的版本一致性未证实，需用实际库与 profiling 证据进一步核实。

后续验证固定 H/P/D 及其余配置，细扫 P31/32/33，正式计时不启 profiling。若另跑 profiling，须单独目录和显式配置，不能与正式时延混合；优先查看逐算子 MUL_MAT 时长及可用的 HMX/HVX 标识。

## 二进制与工具链核查

正式粗扫的设备日志明确报告 Hexagon v79，并加载 `libggml-htp-v79.so`。在粗扫完成、下一组模型启动之前，已读取手机 host bridge 与 v79 skel 的 SHA256；两者均与本地 shared 预编译包完全相同，记录见 `results/20261104-104406-450656-http-oracle-grid/runtime-library-hashes.json`。

该 v79 库保留 `op_matmul`、`op_matmul_hvx`、`hmx_mat_mul_permuted_qk_0_d16a32` 等符号，证明包含两类实现；没有可用 DWARF 源码调试信息。系统 objdump 与现有 Android NDK llvm-objdump 均不支持 Hexagon 指令解码，因此不能仅从这些符号证明库中具有源码相同的 M32 分支。

`GGML_HEXAGON_PROFILE=1` 为每个 GGML op 输出形状、类型和耗时，使用通用 `MUL_MAT` 名字，不能直接标记 HVX/HMX。源码的 FARF(HIGH) 路径提示受 debug 编译开关控制，当前包的 `GGML_HEXAGON_HTP_DEBUG` 为 OFF，运行时环境变量不能恢复被编译裁掉的 HIGH 日志。

正式 H/P 测量结束后，先单独记录 op profile 并以 `GGML_HEXAGON_USE_HMX=0` 做私有进程诊断对照，不混入正式计时。直接分支探针需要匹配的 Hexagon SDK/tools 重建测试用 skel；若具备工具链，应该放入本次实验私有目录并显式设置加载路径，不覆盖共享安装库。用户另指定对 `10.29.243.183` 仅作工具链只读检查，Luna 不编译、不修改远端。

## 10.29.243.183 只读检查结果

Luna 通过既有 SSH 配置只读检查，没有启动容器、拉取镜像、修改文件或编译。远端 `/home/wins/ai-workspace/tools/llama.cpp-hexagon-build/` 保留 `src`、`build-snapdragon`、`pkg-snapdragon`。其 pkg/build 的 v79 HTP 库和 host bridge SHA256 均与手机安装版本相同；远程 `matmul-ops.c` 与本地 shared 同名源码 SHA256 也相同（`df0cfe65068041c5ab6a4d6b0808d75135de8177eda5b0c5e362fcb9c7094ae1`）。源码目录没有 `.git`，不能提供提交ID，也没有完整构建源文件hash清单，因此不把相同源码文件直接当成完整source-to-binary证明。

宿主机没有 `/opt/hexagon/6.4.0.2`，但已经缓存官方工具链镜像 `ghcr.io/snapdragon-toolchain/arm64-android:v0.3`，镜像ID `sha256:e0c11123e0104737c5c11643ec64933a11431b5fb95aeb54f85eefa757172332`，约8.04GB。官方构建README与历史日志均指向该镜像中的SDK6.4.0.2及Tools19.0.04。没有运行中的对应容器，本次未启动新容器，所以镜像内编译器路径和版本尚未直接核实。不能将宿主机未找到SDK误写为远端没有工具链；也不能把镜像存在误写为已验证可成功编译。

## 远端编译及直接路由验证（已完成）

用户授权远端编译后，在已有容器中实际验证了 Hexagon Clang19.0.04（SDK6.4.0.2）、CMake/Ninja。构建记录见 `results/20261104-133409-158979-htp-route-probe-build/`，AGENTS.md 已用一行记录环境。

新增私有 v79 探针，只在原分支处记录路由及门槛原因，不改变原计算内核和返回逻辑。利用现有profile=2的PMU槽作为诊断数据通道，未改变RPC布局；本次这些槽不再是硬件PMU计数。只给本次进程设置私有ADSP库路径，不覆盖安装库。

`results/20261104-133528-960312-http-oracle-grid/` 共18条CPU/HC校准，通过完整JSON、生成ID、EOG、cosine≥0.99及状态迁移。2916个MUL_MAT全部收到DSP实际分支标签，attention其中648条：P31的216条均为HVX，原因为M_lt_32；P32和P33各216条均为HMX且返回成功；未见HMX失败。原手机host bridge/v79库SHA验后未变，模型进程与reverse均已清理。

因此，原发布库的profiling/HMX关闭对照，加上保持原dispatch的重建探针库实际分支记录，相互印证了F16 attention的M32路由门槛。直接标签来自重建探针库，不能冒称原发布库本身新增了日志；探针耗时不并入正式性能结果。测试41项通过。
