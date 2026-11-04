# EXP-005 各框架"动作后→下一次观察"的等待与截图方式(源码引用)
源码在 2026-11-02 浅克隆(commit 如下), 行号对应该 commit。

| 框架 | 仓库@commit |
|---|---|
| AndroidWorld | https://github.com/google-research/android_world @55ddaf5edc6f6c25673aed343907d5b9dca66f84 |
| AppAgent | https://github.com/TencentQQGYLab/AppAgent @2c1900422caf6f9e94e96d5dd984b530e5a5fbf8 |
| AndroidLab | https://github.com/THUDM/Android-Lab @4bfe108cf3883b710e8e5c6fba288161c830e504 |
| Mobile-Agent(v1/v2/v3/E) | https://github.com/X-PLUG/MobileAgent @1d54cac967907607c7e73a0b59119dd03d2d5e42 |
| UI-TARS (模型/prompt) | https://github.com/bytedance/UI-TARS @582f3a7ea5d285ee8ed9e2e84048d1ab01453c49 |
| UI-TARS-desktop (ADB operator) | https://github.com/bytedance/UI-TARS-desktop @3ac2cb8d946d78d5ff92ad17b318eaafa18c178c |
| DroidRun (mobilerun) | https://github.com/droidrun/droidrun @11dcdebdec9ea99f172dd7e80af7ba1eadbeb6c8 |

## AndroidWorld  (注意: 并非纯固定等待)
- `android_world/agents/base_agent.py:52` transition_pause 默认 1.0 s; `:101-116` get_post_transition_state: transition_pause 为数值则固定 sleep, 为 None 则走条件等待 `_get_stable_state`。
- `android_world/env/interface.py:243-247` `_get_stable_state`: UI 元素树连续 3 次不变才返回, 检查间隔 0.5 s, 超时 6.0 s (条件等待, 上限 6 s)。`:292-295` get_state; `:297-306` execute_action。
- `run.py:142` `_MINIWOB_TRANSITION_PAUSE=0.2`; `run.py:218-222` 非 MiniWoB 任务 transition_pause=None(走上面的稳定等待), MiniWoB 为 0.2 s 固定。
- M3A: `android_world/agents/m3a.py:342` wait_after_action_seconds 默认 2.0; `:540` 动作后 sleep(2.0); `:542` get_state(截图+a11y); `:385` 每步开头 get_post_transition_state。
- `android_world/env/actuation.py:84,102` input_text 内 1.0 s; `:201` wait 动作 1.0 s。
- 截图: android_env 模拟器 gRPC + a11y forwarder app / uiautomator, 真机不装 app 无法复现 -> 未测。

## AppAgent
- `scripts/task_executor.py:281` sleep(REQUEST_INTERVAL); `config.yaml:8` REQUEST_INTERVAL: 10 (s)。
- `scripts/task_executor.py:145-146` 每步先截图再取 XML; `scripts/and_controller.py:104-130` `screencap -p /sdcard/..` + `pull`; `uiautomator dump` + `pull`。

## AndroidLab
- `evaluation/auto_test.py:246` sleep(request_interval); 默认 3 s (`docs/prepare_for_linux.md:102`, `configs/*.yaml`)。
- `page_executor/text_executor.py:190` swipe 后 sleep 1; `:217` wait 5 s; `page_executor/simple_vision_executor.py:147` wait 默认 5 s。
- `utils_mobile/and_controller.py:72-81, 88-95` screencap + pull; `recorder/json_recoder.py` update_before: 先截图再 XML。

## Mobile-Agent
- v3: `Mobile-Agent-v3/mobile_v3/run_mobileagentv3.py:233` step0 sleep 8 s; `:234` 之后 sleep 2 s; 然后截图(失败重试间隔 5 s)。`mobile_v3/utils/android_controller.py:11-24`: rm → sleep 0.5 → screencap → sleep 0.5 → pull (截图函数自带 1.0 s 固定 sleep)。
- v2: `Mobile-Agent-v2/run.py:367` sleep(5); `MobileAgent/controller.py:7-20` 同样 0.5/0.5 sleep; `run.py:174-175` 截图。
- E: `Mobile-Agent-E/inference_agent_E.py:108` SLEEP_BETWEEN_STEPS=5, `:1069-1070`; `MobileAgentE/controller.py:7-20` 0.5/0.5。
- v1: `Mobile-Agent-v1/MobileAgent/controller.py:17-20` 0.5/0.5; 动作后各 sleep 1 s (`:31, 41, 62` 等)。

## UI-TARS / UI-TARS-desktop
- UI-TARS 仓库: `codes/ui_tars/prompt.py:20` 提示词里 wait() "Sleep for 5s"; 无 ADB 执行代码。
- UI-TARS-desktop: `packages/ui-tars/operators/adb/src/index.ts:84-100` `adb -s <id> exec-out screencap -p` (5000 ms 超时); `:295-297` wait=2000 ms。
- `packages/ui-tars/sdk/src/GUIAgent.ts:424-431` 循环末 sleep(loopIntervalInMs); `:185` 截图; `:207` 截图无效时 1000 ms。
- `apps/ui-tars/src/main/store/setting.ts:28` loopIntervalInMs 默认 1000; `store/validate.ts:29` 范围 0–3000。

## DroidRun / mobilerun
- `mobilerun/config_manager/config_manager.py:132` after_sleep_action=1.0; `:133` wait_for_stable_ui=0.3 (未找到使用方); `:364-365`。
- `mobilerun/agent/fast_agent/fast_agent.py:706`, `mobilerun/agent/executor/executor_agent.py:260` sleep(after_sleep_action)。
- 截图经手机端 Portal app, 真机不装 app 无法复现 -> 未测。
