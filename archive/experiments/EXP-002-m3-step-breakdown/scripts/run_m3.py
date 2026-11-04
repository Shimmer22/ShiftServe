"""EXP-002 runner: per-step VLM timing split on the phone via llama-mtmd-cli (CPU).

Uses existing phone assets in /data/local/tmp (llama-mtmd-cli, Qwen2.5-VL-3B Q4_K_M, mmproj f16);
nothing is downloaded or installed. Each run = one fresh process (model load is reported separately).
Raw stdout/stderr per run are saved under results/raw/. No credentials involved.
"""
import json, pathlib, re, subprocess, sys, time, shlex

ADB = "/Users/hui/Library/Android/sdk/platform-tools/adb"
SER = "10.129.178.171:45021"
EXP = pathlib.Path(__file__).resolve().parents[1]
RAW = EXP / "results" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
DEV = "/data/local/tmp"; WD = DEV + "/a2o_m3"
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PROMPT = ("You are a GUI agent operating an Android phone. Task: open the Gmail app. "
          "Look at the screenshot and output the next action as JSON.")
PROMPT_TXT = ("You are a GUI agent operating an Android phone. Task: open the Gmail app. "
              "The screen shows the home screen. Output the next action as JSON.")

def adb(cmd, timeout=900):
    r = subprocess.run([ADB, "-s", SER, "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout

def ctx():
    s = adb("cat /proc/loadavg; grep -E 'MemAvailable|MemFree' /proc/meminfo; "
            "cat /sys/class/power_supply/battery/temp /sys/class/power_supply/battery/capacity 2>/dev/null")
    L = s.split("\n")
    la = L[0].split()[:3]
    mem = {l.split(":")[0]: int(l.split()[1]) for l in L[1:3] if ":" in l}
    return dict(load1_5_15=la, mem_available_kb=mem.get("MemAvailable"), mem_free_kb=mem.get("MemFree"),
                battery_temp_deci_c=L[3].strip() if len(L) > 3 else None,
                battery_pct=L[4].strip() if len(L) > 4 else None)

def parse(err):
    g = lambda pat, flags=0: re.search(pat, err, flags)
    d = {}
    m = g(r"image slice encoded in (\d+) ms");            d["vit_ms"] = int(m[1]) if m else None
    m = g(r"decoding image batch \d+/\d+, n_tokens_batch = (\d+)"); d["image_tokens"] = int(m[1]) if m else None
    m = g(r"image decoded \(batch .*?\) in (\d+) ms");     d["image_prefill_ms"] = int(m[1]) if m else None
    m = g(r"load time =\s+([\d.]+) ms");                   d["load_ms"] = float(m[1]) if m else None
    m = g(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens"); d["perf_prompt_eval_ms"], d["perf_prompt_tokens"] = (float(m[1]), int(m[2])) if m else (None, None)
    m = g(r"\beval time =\s+([\d.]+) ms /\s+(\d+) runs");  d["decode_ms"], d["decode_tokens"] = (float(m[1]), int(m[2])) if m else (None, None)
    m = g(r"total time =\s+([\d.]+) ms");                  d["perf_total_ms"] = float(m[1]) if m else None
    return d

def run(name, image, tag, rep, prompt):
    img = f"--image {WD}/{image}.png" if image else ""
    out = f"{WD}/{name}_{rep}"
    cmd = (f"cd {DEV} && ./llama-mtmd-cli -m models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf "
           f"--mmproj models/mmproj-model-f16.gguf {img} -p {shlex.quote(prompt)} -c 2048 -t {THREADS} -ngl 0 -n 48 --temp 0 "
           f"> {out}.out 2> {out}.err < /dev/null; cat {out}.err > /dev/null")
    c0 = ctx(); t0 = time.time()
    adb(cmd)
    wall = time.time() - t0; c1 = ctx()
    err = adb(f"cat {out}.err"); so = adb(f"cat {out}.out")
    (RAW / f"{name}_{rep}.err.txt").write_text(err); (RAW / f"{name}_{rep}.out.txt").write_text(so)
    adb(f"rm -f {out}.out {out}.err")
    d = parse(err)
    d.update(step=name, kind=tag, rep=rep, threads=THREADS, wall_s=round(wall, 2), ctx_before=c0, ctx_after=c1,
             output_text=so.strip()[:300])
    print(json.dumps({k: d[k] for k in ("step","rep","image_tokens","vit_ms","image_prefill_ms","perf_prompt_tokens","perf_prompt_eval_ms","decode_tokens","decode_ms","wall_s")}), flush=True)
    return d

if __name__ == "__main__":
    adb(f"mkdir -p {WD}")
    man = json.loads((EXP / "inputs" / "manifest.json").read_text())
    for m in man:
        subprocess.run([ADB, "-s", SER, "push", str(EXP / "inputs" / f"{m['name']}.png"), WD + "/"], capture_output=True)
    plan = [("aw_home_S", 3), ("aw_home_M", 3), ("aw_home_L", 1), ("desk0_M", 1), ("desk1_M", 1)]
    res = []
    for name, n in plan:
        for r in range(n):
            res.append(run(name, name, "image", r, PROMPT))
    for r in range(3):
        res.append(run("text_only", None, "text", r, PROMPT_TXT))
    (EXP / "results" / f"m3_runs_t{THREADS}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    adb(f"rm -rf {WD}")
    print("DONE")
