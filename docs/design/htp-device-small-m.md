# HTP设备内小M补齐原型

目标是在DSP内部为小M矩阵选择较有利的计算路径，避免在主机复制整块激活、传输补行或读回补行。本分支仅开发M31的最小原型，未接入主线默认运行库。

## 实现

复用缓存llama.cpp中已有的HMX尾块逻辑。`transfer_activation_chunk_fp32_to_fp16` 在读取有效激活时，将无效行直接填零到VTCM的32行tile；`transfer_output_chunk_threaded` 按实际`n_rows`写回。外部输入输出形状始终是31行，RPC和主机图不增加节点，不向prompt插入dummy token，不裁剪recurrent/conv状态。

只修改DSP `op_matmul` 的M门槛，其他类型、对齐、batch及布局检查保留。策略谓词位于 `src/a2o/native/backend/htp_small_m_policy.h`：

- `original`：原有M≥32门槛。
- `force`：允许M31的Q4非batch矩阵走HMX，仅用于负对照。
- `guarded`：只允许M31、Q4、非batch、K4096且N8192或12288。它对应已测试的QKV和FFN Up；K12288/N4096的FFN Down保留HVX。

白名单是一条由单设备、单层、合成激活微基准得到的实验规则，不代表其他层输入或其他M已验证。该范围内HMX仍需计算32行tile，无法消除无效行的算术成本，也不能保证比HVX逐位相同。新路径执行失败显式返回错误，不递归尝试HMX或静默更换后端。

## 构建与复现

在仓库根目录，选择新的结果目录，按下列顺序运行。构建仅使用wins缓存镜像和私有源码目录，手机运行持Termux SSH模型锁，运行库不覆盖共享安装。

```bash
python scripts/build/htp_small_m.py --policy guarded --output results/20261112-htp-device-small-m/build-guarded
python scripts/build/htp_small_m.py --policy force --output results/20261112-htp-device-small-m/build-force
python scripts/build/htp_small_m.py --policy guarded --profile --output results/20261112-htp-device-small-m/build-guarded-profile
python scripts/build/htp_small_m_bench.py --output results/20261112-htp-device-small-m/build-bench
python scripts/run/htp_small_m.py --output results/20261112-htp-device-small-m
/Users/hui/miniforge3/envs/data/bin/python scripts/analyze/htp_small_m.py --input results/20261112-htp-device-small-m
```

已有结果拒绝覆盖；复现请整体替换上述结果目录。路由诊断与计时分开编译运行，诊断标签不参与正式时间。原版与guarded按ABBA顺序，每进程每形状每条件10次，强制版本补负对照。主机缓冲区均提前分配并复用，计时包含输入上传、计算同步、有效输出读回；主机补齐对照另包含复制及完整32行传输。

后续模型集成可在 `scripts/run/route_phone.py start` 的原有参数上显式添加 `--htp-skel <guarded库路径>`，其profile保持0，启动清单记录库SHA-256。该选项与诊断用 `--probe-skel --htp-profile 2` 互斥，默认仍使用原运行库。本轮只使用独立微基准，未通过该服务选项做模型级验收。

## 验证边界

主机测试直接编译实际C谓词，覆盖M1至128、三种模式、类型/batch与形状边界。手机测试使用真实layer0 Q4权重及固定合成激活，检查每次输出后方紧邻的128个F32哨兵。跨进程保存所有有效输出并校验：内部HMX与外部补32行的前31行应逐位一致；guarded中未选择的FFN Down应与原版逐位一致。内部HMX相对原HVX的差异单独记录，不当作数值不变。

首次算子结果在 `results/20261112-htp-device-small-m/README.md` 与 `summary.json`。原始输出以SHA-256标识，日志、原始计时与构建清单保留。算子验证不代表模型级通过；后续固定历史的完整模型验证见下一节。该原型不安装到默认手机服务。

## 固定历史后的完整模型验证

后续验证使用 `src/a2o/native/profiling/htp_small_m_e2e.cpp`，原版与guarded只更换私有DSP，其余可执行文件、模型及运行库完全一致。直接读取此前H4096、CPU6线程权威快照，检查模型路径/大小/mtime、前缀token、context、batch、状态字节数和checksum；不冒用旧可执行文件的缓存身份，也不重新计算不同历史。该快照187MB，完整恢复后重新导出并逐字节核验。

固定P31真实输入、D16固定后续token，两阶段都在同一HTP上下文执行。四个进程按原版、guarded、guarded、原版运行；各进程完成状态/续写验证、预热一次，再计时10次。记录完整模型prefill、HTP decode，以及状态恢复工作时间。恢复中的完整字节核验、其余验证、模型初始化、日志及文件传输均在性能段之外。

验证额外导出完整prefill状态，比较直接D16续写与恢复该状态后的D16续写；逐元素比较全部最终logits，并比较完整最终状态checksum。跨版本保存prefill和D16后的完整logits、状态checksum，以及16步自由greedy延续。固定token回放的速度与greedy延续的响应一致性分别报告，不能据此宣称整个BFCL任务或JSON语义通过。

```bash
python scripts/build/htp_small_m_bench.py --e2e --output results/20261112-htp-small-m-e2e/build-bench
python scripts/run/htp_small_m_e2e.py --output results/20261112-htp-small-m-e2e
/Users/hui/miniforge3/envs/data/bin/python scripts/analyze/htp_small_m_e2e.py --input results/20261112-htp-small-m-e2e
```

设备入口复用前面guarded构建结果；完整模型结果见 `results/20261112-htp-small-m-e2e/README.md`。这里的推理段总时间不包含工具、网络、视觉和冷启动，不等于整个Agent的用户等待时间。所有计时原样保留，不去长尾；ABBA只有两个进程对，不能把进程内20次重复当作20次独立设备环境。

本轮40次正式计时没有观察到稳定的整体收益：合并prefill均值慢0.48%，恢复＋prefill＋D16均值慢3.93%；但两个进程对的总时间分别慢11.24%、快2.22%，存在明显顺序漂移，不能把合并均值解释为确定的机制因果减速。完整logits相对L2差异在prefill后为5.93%，D16后为7.96%；自由greedy从第6个token开始分叉。两版本各自重复结果一致，完整状态恢复后续写检查通过，但跨版本不等价，不作为无损优化启用。
