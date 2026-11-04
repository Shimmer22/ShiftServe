"""EXP-001 orchestrator: run a batch of SAFE GUI actions over wireless adb and collect per-trial frame logs.

Safe set only (navigation / viewing in Settings and launcher; no value changes, no send/buy/delete):
  open_settings, scroll_down, scroll_up, open_submenu (Display page via intent, view only), back,
  home_from_app, recents_open, recents_close (HOME from recents).
Modes:
  --preflight   check screen on / unlocked / input injection permitted (read-only + harmless HOME key)
  --overhead    N screencaps on a static screen -> capture cost (no input injected)
  --noop-smoke  run a no-op injection through the trial script (tooling smoke test; injects nothing)
  --run N       real measurement, N iterations of the 8-step cycle (requires a passing preflight)
"""
import argparse, json, pathlib, re, subprocess, sys, time

ADB = "/Users/hui/Library/Android/sdk/platform-tools/adb"
SER = "10.129.178.171:45021"
EXP = pathlib.Path(__file__).resolve().parents[1]
RES = EXP / "results"; RES.mkdir(exist_ok=True)
DEV = "/data/local/tmp/a2o_m1"

def adb(cmd, timeout=120):
    r = subprocess.run([ADB, "-s", SER, "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr

def preflight():
    p = adb("dumpsys power | grep -E 'mWakefulness=|Display Power'; dumpsys window | grep -E 'mDreamingLockscreen|isKeyguardShowing' | head -3; "
            "wm size; getprop ro.product.model; getprop ro.build.version.release")
    awake = "mWakefulness=Awake" in p
    kg = re.search(r"isKeyguardShowing=(\w+)", p) or re.search(r"mDreamingLockscreen=(\w+)", p)
    locked = (kg and kg.group(1) == "true") if kg else None
    inj = adb("input keyevent KEYCODE_HOME")  # harmless; exposes the INJECT_EVENTS SecurityException if gated
    inj_ok = "SecurityException" not in inj
    info = dict(awake=awake, keyguard_showing=locked, inject_ok=inj_ok, raw=p.strip()[:600], inject_msg=inj.strip()[:200])
    info["ok"] = bool(awake and not locked and inj_ok)
    return info

def screen():
    m = re.search(r"(\d+)x(\d+)", adb("wm size"))
    return int(m[1]), int(m[2])

def push():
    adb(f"mkdir -p {DEV}")
    subprocess.run([ADB, "-s", SER, "push", str(pathlib.Path(__file__).with_name("a2o_trial.sh")), DEV + "/"], capture_output=True)

def trial(cls, inj, it, noop=False):
    out = adb(f"sh {DEV}/a2o_trial.sh '{inj}' 3 8000 1500", timeout=60)
    return dict(cls=cls, iter=it, inject=inj, raw=out)

def cycle(W, H):
    x, y0, y1 = W // 2, int(H * .75), int(H * .30)
    return [("open_settings", "am start -a android.settings.SETTINGS"),
            ("scroll_down", f"input swipe {x} {y0} {x} {y1} 300"),
            ("scroll_up", f"input swipe {x} {y1} {x} {y0} 300"),
            ("open_submenu", "am start -a android.settings.DISPLAY_SETTINGS"),
            ("back", "input keyevent KEYCODE_BACK"),
            ("home_from_app", "input keyevent KEYCODE_HOME"),
            ("recents_open", "input keyevent KEYCODE_APP_SWITCH"),
            ("recents_close", "input keyevent KEYCODE_HOME")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true"); g.add_argument("--overhead", type=int)
    g.add_argument("--noop-smoke", type=int); g.add_argument("--run", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.preflight:
        print(json.dumps(preflight(), indent=1, ensure_ascii=False)); sys.exit(0)
    push()
    ts = time.strftime("%Y%m%d-%H%M%S")
    if a.overhead:
        rows = []
        for i in range(a.overhead):
            rows.append(adb(f"sh {DEV}/a2o_trial.sh ':' 1 0 0", timeout=60))  # no-op, 1-2 frames
        (RES / f"m1_overhead_raw_{ts}{a.tag}.json").write_text(json.dumps(rows))
        print("saved overhead raw", len(rows))
    elif a.noop_smoke:
        rows = [trial("noop", ":", i, True) for i in range(a.noop_smoke)]
        (RES / f"m1_noop_raw_{ts}{a.tag}.json").write_text(json.dumps(rows))
        print("saved noop raw", len(rows))
    else:
        pf = preflight()
        if not pf["ok"]:
            print("PREFLIGHT FAILED - refusing to measure:", json.dumps(pf, ensure_ascii=False)); sys.exit(2)
        W, H = screen(); rows = []
        for it in range(a.run):
            for cls, inj in cycle(W, H):
                time.sleep(1.5)  # let previous action fully settle before baseline
                rows.append(trial(cls, inj, it))
                print(it, cls, flush=True)
        (RES / f"m1_raw_{ts}{a.tag}.json").write_text(json.dumps(rows))
        print("saved raw", len(rows))
