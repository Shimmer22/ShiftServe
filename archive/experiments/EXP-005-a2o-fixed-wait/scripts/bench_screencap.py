#!/usr/bin/env python3
"""EXP-005: 截图开销实测(Mac 端 wall-clock, 含 adb 客户端进程启动)。
方法 (每轮按顺序交错, 以免时间漂移偏向某方法):
  execout   : adb exec-out screencap -p  -> Mac 内存 (UI-TARS-desktop adb operator 同款)
  cap_pull  : adb shell screencap -p /sdcard/x.png ; adb pull  (AppAgent / AndroidLab / Mobile-Agent 去掉其sleep后的同款)
  cap_only / pull_only : cap_pull 的两段分别计时
  rm_cap_pull : Mobile-Agent 同款多一次 adb shell rm (其代码另含 2x0.5s 固定 sleep, 不在此测量内)
  uidump_pull: adb shell uiautomator dump + pull (AppAgent/AndroidLab 每步还要的 XML)
只用 csv/numpy。不改设置。远端临时文件 /sdcard/a2o_exp005_*.png|xml, 结束后删除。
"""
import subprocess, time, csv, json, sys, os, numpy as np, argparse
ADB = "/Users/hui/Library/Android/sdk/platform-tools/adb"
DEV = "10.129.178.171:45021"
R_PNG = "/sdcard/a2o_exp005_cap.png"; R_XML = "/sdcard/a2o_exp005_ui.xml"
def run(args, out=False):
    t = time.perf_counter()
    p = subprocess.run([ADB, "-s", DEV] + args, capture_output=True)
    return (time.perf_counter() - t) * 1000, p
def one_round(tmp):
    r = {}
    ms, p = run(["exec-out", "screencap", "-p"]); r["execout_ms"] = ms; r["execout_bytes"] = len(p.stdout)
    ms1, p1 = run(["shell", "screencap", "-p", R_PNG])
    ms2, p2 = run(["pull", R_PNG, tmp + "/cap.png"])
    r["cap_ms"] = ms1; r["pull_ms"] = ms2; r["cap_pull_ms"] = ms1 + ms2
    r["pull_bytes"] = os.path.getsize(tmp + "/cap.png") if os.path.exists(tmp + "/cap.png") else -1
    ms0, _ = run(["shell", "rm", R_PNG])
    ms1b, _ = run(["shell", "screencap", "-p", R_PNG]); ms2b, _ = run(["pull", R_PNG, tmp + "/cap.png"])
    r["rm_cap_pull_ms"] = ms0 + ms1b + ms2b
    run(["shell", "rm", R_PNG])
    ms3, _ = run(["shell", "uiautomator", "dump", R_XML]); ms4, _ = run(["pull", R_XML, tmp + "/ui.xml"])
    r["uidump_pull_ms"] = ms3 + ms4; r["uidump_ms"] = ms3
    run(["shell", "rm", R_XML])
    return r
def stats(v):
    v = np.array(v, float)
    return dict(n=len(v), mean=round(float(v.mean()), 1), median=round(float(np.median(v)), 1),
                p90=round(float(np.percentile(v, 90)), 1), min=round(float(v.min()), 1), max=round(float(v.max()), 1))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmp = "/tmp/a2o_exp005"; os.makedirs(tmp, exist_ok=True)
    for _ in range(2): one_round(tmp)  # warm-up, discarded
    rows = []
    for i in range(a.n):
        r = one_round(tmp); r["iter"] = i; rows.append(r); print(i, {k: round(v) for k, v in r.items() if k.endswith("_ms")}, flush=True)
    keys = ["iter"] + [k for k in rows[0] if k != "iter"]
    with open(a.out + "_rounds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    summ = {k: stats([r[k] for r in rows]) for k in keys if k.endswith("_ms")}
    json.dump(summ, open(a.out + "_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1))
