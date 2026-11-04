# 三项基础开发验收

`summary.json` 与 `final-checks.json` 均 passed=true，所有耗时无效，不比较性能。

| 项目 | 证据 |
|---|---|
| 跨轮非阻塞 | turn1 旧任务在导入核验后的屏障中保持存活；turn1/turn2 CPU 结果均实际发布后才释放；busy 时未新建候选 |
| 成本 Router | turn3 ready但CPU更便宜，turn4 ready且HTP更便宜；分支成本明确为mechanism-fixture、样本数0 |
| 实际 Tool | 每轮实际执行只读adb wm size；native收到真实返回后计算P；没有预读下一观察或执行模型动作 |
| 状态正确性 | 两条件各六轮，JSON/记忆/step/EOG通过；原始工具结果、完整生成ID、prefix、计数及下一位置一致 |
| 回收 | 共启动3个后台任务，关闭时3个均回收；单lane上界1 |
| 普通自动模式 | 另三轮禁用所有机制夹具；coverage_missing/CPU路径通过，与对照前三轮一致 |
| 本地验证 | 22项测试及Android构建通过；实测表仅导出合格CC/HC条件，精确H/P/D匹配 |

核心六轮用例代码提交为 `0a59717`；三轮普通模式提交为 `573a541`。两次六轮的源码和二进制hash相同。普通模式与核心用例的代码存在覆盖缺口标记/诊断计时差异，未合并成同二进制对照。最终源码另外收紧合成成本准入，普通base表不得用fixture，只有显式机制RUN允许；该限制由本地C++测试覆盖。

原始目录：

- `../20261104-084949-792027-tool-session-control/`
- `../20261104-085032-508313-tool-session-mechanism/`
- `../20261104-085317-179323-tool-session-auto/`

backend API仍不可抢占、关闭允许等待；屏障位于真实导入之后，模拟不可抢占的返回边界，不代表验证了所有底层锁。实测表覆盖有限，普通Tool输入尚无对应代价。没有图像/逐步logits/能耗/长期实验或Gap翻转证据。
