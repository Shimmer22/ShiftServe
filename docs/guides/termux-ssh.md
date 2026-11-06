# Termux SSH 与灭屏持锁

2026-11-05 在小米 14 Pro 上完成连接修复与短探针验收。原始记录位于
`results/20261105-termux-ssh-setup/`；没有新增正式模型性能数据。

## 连接

```sh
ssh -p 8022 -i ~/.ssh/a2o_xiaomi14pro_ed25519 \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/Users/hui/code/exp/2026-11-a2o-gap-profiling/configs/termux-known-hosts \
  u0_a320@10.29.131.50
```

主机 ED25519 指纹已经通过 ADB 显示的本机指纹核对：
`SHA256:B9EGQldo4DFLCoSc/e3scajQkrTmcDdkujAASnvPc/I`。
本机私钥只保存在 `~/.ssh/a2o_xiaomi14pro_ed25519`，手机仅追加对应公钥。
IP 改变后应重新核对设备身份，不关闭主机密钥验证。

## 本次修复

- `sshd` 原先因缺少 `libcrypto.so.3` 不能启动；更新 OpenSSL 与 Termux
  工具依赖后恢复。包配置提示均保留原有版本，没有进行全量升级。
- 使用 `ssh-keygen -A` 补齐缺失的 SSH 主机密钥，启动 `sshd` 并验证公钥登录。
- 原 `termux-wake-lock` 报 `Aborted`；更新后系统实际观察到
  `PARTIAL_WAKE_LOCK 'termux:service-wakelock'`。未修改省电白名单等手机设置。

## 灭屏验收

通过 SSH 在 Termux UID 10320 下运行独立时钟探针。前后均为 `Dozing`，
最近唤醒时间未改变，Termux 持锁始终存在。MONOTONIC 与 BOOTTIME 均经过
约 5.00138 秒，差约 0.000000417 秒；本次短测试未观察到明显系统挂起。
它只证明该短窗口可执行，不代表长时间运行或 HTP 已验收。

原动态链接探针缺少 `libc++_shared.so`，失败输出保留；改用同一源文件、
静态链接 C++ 标准库的小探针完成上述检查，未复制模型或大资产。

Termux 已实际读取现有模型的 `GGUF` 文件头，也能列出 Hexagon 库路径；
这不等于运行库加载、DSP 访问、JSON 响应或状态迁移已通过。下一步按
CPU、HTP、hybrid 串行验收，再重做 H4096/P32 的短 Gap 配对实验。

## 生命周期

在 Termux 中启动 SSH 服务使用 `sshd`。实验前获取锁：

```sh
termux-wake-lock
```

每次仍需检查系统实际持锁状态；不能只凭命令退出码判断成功。
实验结束后释放锁：

```sh
termux-wake-unlock
```

当前为后续灭屏实验保留 SSH 服务和持锁；没有配置开机自启。

## 原报告后台复跑

`scripts/profile/oracle.py` 默认使用 SSH，原 ADB 通道只在显式指定
`--transport adb --device SERIAL` 时可用，不发生自动回退。
Termux 无 DUMP 权限且不能读取受保护序列号：manifest 保留权限限制，
以持锁请求与双时钟短探针检查持续执行条件，不伪报 Awake 或序列号。

Android 禁止 Termux 执行映射 shell 目录下的部分动态库，因此已有运行库
按原字节放在 `~/.local/a2o/runtime/lib/` 并逐文件校验，与原库 hash 一致；
模型继续读取原路径。每轮私有目录另放与编译器匹配的 NDK C++ 运行库，
库搜索路径和 hash 写入 manifest。HTTP 通道明确变为 SSH 反向转发，
不能把新旧实际 Gap 的差异直接归因于策略。

2026-11-05 的 H256/P32 预检共6条，CPU/HTP各3条的JSON、完整ID/EOG、
状态和数值准入通过，原始目录为
`results/20261105-220919-612886-http-oracle-grid/`。
此前两次环境适配失败的输出仍保留。

后台入口（从仓库根目录，使用包含 numpy/matplotlib 的现有 Python 环境）：

```sh
/Users/hui/miniforge3/envs/visiontrim-qwen25vl/bin/python \
  scripts/run/start_termux_report.py \
  --preflight-run results/20261105-220919-612886-http-oracle-grid
```

它启动独立会话，并用本机 caffeinate 保持执行期间不自动闲置休眠；
主矩阵 H2048/H4096 各381条，再运行 P31 两个H各46条，共854条报告范围记录。
配置固化到结果目录，所有条件串行执行；完成后复用旧分析器验收，
仅写JSON与原始日志，不生成图表/PDF，不更改原报告。
任一运行或验收失败立即停止；正常结束或失败时释放实验wake lock，保留SSH服务。
`campaign.json` 每5秒记录当前运行与条数，`detached.json` 保存PID和命令。
无需保持当前对话打开；本机仍需开机、联网并可访问手机。
