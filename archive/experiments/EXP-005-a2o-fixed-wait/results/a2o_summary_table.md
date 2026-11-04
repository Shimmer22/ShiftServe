# A2O 汇总: A2O = 框架动作后等待 + 截图开销
截图开销取自本机实测中位数 (results/screencap_locked_summary.json, **锁屏/息屏画面, 是下界, 见 README**)。
标 (推) 为由源码常数 + 实测中位数相加推算, 非整体实测。单位 ms。

| 框架 | 动作后等待(源码) | 截图方式 | 截图开销(实测中位数) | A2O (推) |
|---|---|---|---|---|
| AndroidWorld run.py(非MiniWoB) | 条件等待: UI树连续3次不变, 间隔0.5s, 上限6000 | 模拟器gRPC+a11y | 未测 | 未知(等待 ≥1500, ≤6000 + 截图) |
| AndroidWorld M3A 动作后 | 固定 2000 | 同上 | 未测 | 2000 + 未测 |
| AppAgent | 固定 10000 | screencap+pull, 再 uiautomator dump+pull | 481.2 + 2222.4 | 10000+481.2+2222.4 = 12703.6 |
| AndroidLab | 固定 3000 (默认) | screencap+pull, 再 XML | 481.2 + 2222.4 | 3000+481.2+2222.4 = 5703.6 |
| Mobile-Agent-v3 | 固定 2000 (step0 8000) + 截图内 sleep 1000 | rm+screencap+pull | 546.6 | 2000+1000+546.6 = 3546.6 |
| Mobile-Agent-v2 / E | 固定 5000 + 截图内 sleep 1000 | 同上 | 546.6 | 5000+1000+546.6 = 6546.6 |
| Mobile-Agent-v1 | 固定 1000 + 截图内 sleep 1000 | 同上 | 546.6 | 1000+1000+546.6 = 2546.6 |
| UI-TARS-desktop (ADB) | 固定 1000 (loopInterval 默认); wait 动作 2000 | exec-out screencap -p | 345.4 | 1000+345.4 = 1345.4 |
| DroidRun | 固定 1000 (after_sleep_action) | Portal app | 未测 | 1000 + 未测 |
