# 目录与机制边界

## Python 与原生源码

| 路径 | 职责 |
|---|---|
| `src/a2o/environment.py` | 唯一的工程/共享运行时路径、设备默认值、Android 构建与递归源码 hash |
| `src/a2o/runtime/` | 单次回放与常驻会话编排、流式进程协议；不拥有代价或响应检查实现 |
| `src/a2o/scheduling/` | 离线代价公式和合格实测表导出 |
| `src/a2o/tools/` | 显式白名单的只读 ADB / HTTP 工具 |
| `src/a2o/validation/` | 配置、JSON/记忆/状态检查和生命周期验收 |
| `src/a2o/workloads/` | BFCL 轨迹采集与沙箱工具调用 |
| `src/a2o/vision/` | QNN 适配与上游许可头，保留复现但不进入当前主线 |
| `src/a2o/native/backend/` | CPU/HTP Engine、模板、token 评估和采样 |
| `src/a2o/native/state/` | sequence 导出/迁移、完整 snapshot、异步任务和单 lane 生命周期 |
| `src/a2o/native/scheduling/` | 返回时点的 C++ 成本 Router |
| `src/a2o/native/session/` | 新常驻 Tool 会话与旧多轮回放；复用同一份 snapshot/迁移实现 |
| `src/a2o/native/vision/` | 新图 embedding 与 M-RoPE 位置表示 |
| `src/a2o/native/profiling/` | Prefill 与四路径微基准 |

`native/agent_runtime.cpp` 保留统一命令入口及单轮回放逻辑，按上述机制组合执行；没有复制 shared 模型或运行时资产。原生 include root 为 `src/a2o/native/`。所有运行入口都调用 `environment.native_source_hashes()` 递归记录 `.h/.cpp`，避免目录拆分后漏记文件。

## 二级脚本与文档

`scripts/{run,validate,profile,collect,analyze,plot,vision}/` 中保留直接可执行入口，不保留旧扁平路径的兼容壳。常用迁移：

| 旧入口 | 新入口 |
|---|---|
| `scripts/run_tool_session.py` | `scripts/run/tool_session.py` |
| `scripts/run_agent.py` | `scripts/run/agent.py` |
| `scripts/validate_tool_session.py` | `scripts/validate/tool_session.py` |
| `scripts/validate_async.py` | `scripts/validate/async_preparation.py` |
| `scripts/profile_prefill.py` | `scripts/profile/prefill.py` |
| `scripts/export_router.py` | `scripts/profile/export_router.py` |
| `scripts/queue_bfcl_remote.py` | `scripts/collect/queue_bfcl_remote.py` |
| `scripts/collect_bfcl.py` | `scripts/collect/bfcl.py` |

分析和绘图入口统一为 `scripts/analyze/<主题>.py`、`scripts/plot/<主题>.py`；QNN 手动入口在 `scripts/vision/`。它们仍有原始实验复现用途，不当作无用脚本删除。普通 CLI 的 `--help` 只显示参数；旧的批量汇总/绘图脚本可能在顶层执行，不要用 `--help` 试跑它们。

文档只在 `docs/{design,guides,validation,reference}/` 维护正文，索引为 [docs/README.md](../README.md)。根目录仅保留项目 README 与 AGENTS 说明。

## 清理与验证

移除旧扁平文件路径、重复的构建/检查实现、未被读取的 host 事件列表和生成缓存。模型、输入截图、结果、失败日志和 archive 均保留。旧 `results/` 的源码 hash 与脚本路径反映当时版本，不为匹配新目录而改写。

本次目录重构只跑本地测试、结构/链接/CLI 检查与 Android 交叉编译，不需要占用手机。重部署远端 BFCL 时应同步完整的新版 `src/` 和 `scripts/`；本次不更新、重启或停止已排队的远端进程。

2026-11-04 验证：23项本地测试与 Android 构建通过，常用 CLI 参数入口、二级目录、文档链接及13个原生文件的递归 hash 覆盖检查通过。旧单轮/多轮汇总已排除新 Tool 会话格式，回归测试仅在临时目录写报告，已有 results/archive 未改动。
