"""EXP-003 Mac client: phone as inference server (llama-server on phone, 127.0.0.1:PORT, reached via adb forward).

Per request: real phone screenshot (adb exec-out screencap -p) -> resize on Mac -> OpenAI-compatible
/v1/chat/completions (base64 image) -> record per-segment times + server `timings`.
No input injection, no setting changes. Usage: python client.py [N=3] [PORT=18080]
"""
import base64, io, json, pathlib, subprocess, sys, time, urllib.request
from PIL import Image

ADB = "/Users/hui/Library/Android/sdk/platform-tools/adb"; SER = "10.129.178.171:45021"
EXP = pathlib.Path(__file__).resolve().parents[1]; RES = EXP / "results"; INP = EXP / "inputs"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3; PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
RESIZE = (448, 980)  # multiples of 28 -> 560 image tokens for Qwen2.5-VL
PROMPT = ("You are a GUI agent operating an Android phone. Task: open the Settings app. "
          "Look at the screenshot and output the next action as JSON with fields action, coordinate (x,y in the "
          "screenshot pixels) and a short reason.")

def loadavg():
    return subprocess.run([ADB, "-s", SER, "shell", "cat /proc/loadavg"], capture_output=True, text=True).stdout.split()[:3]

def screencap():
    t = time.perf_counter()
    png = subprocess.run([ADB, "-s", SER, "exec-out", "screencap", "-p"], capture_output=True).stdout
    return png, time.perf_counter() - t

def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter(); body = urllib.request.urlopen(req, timeout=900).read(); dt = time.perf_counter() - t
    return json.loads(body), dt, len(body)

def run(tag, png_bytes, cap_s, cache_prompt, idx):
    t = time.perf_counter()
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB"); full = im.size
    im = im.resize(RESIZE, Image.LANCZOS); buf = io.BytesIO(); im.save(buf, "PNG"); small = buf.getvalue()
    prep_s = time.perf_counter() - t
    (INP / f"{tag}.png").write_bytes(small) if cache_prompt is False else None
    payload = {"model": "qwen2.5-vl-3b", "temperature": 0, "max_tokens": 64, "cache_prompt": cache_prompt,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(small).decode()}},
                   {"type": "text", "text": PROMPT}]}]}
    la0 = loadavg()
    resp, http_s, nbytes_resp = post(payload)
    la1 = loadavg()
    tm = resp.get("timings", {})
    row = dict(tag=tag, idx=idx, cache_prompt=cache_prompt, screencap_s=round(cap_s, 3), screenshot_png_bytes=len(png_bytes), screenshot_wh=list(full),
               resize_encode_s=round(prep_s, 3), sent_png_bytes=len(small), request_json_bytes=len(json.dumps(payload)),
               http_e2e_s=round(http_s, 3), e2e_incl_screencap_s=round(cap_s + prep_s + http_s, 3),
               usage=resp.get("usage"), timings={k: tm.get(k) for k in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")},
               loadavg_before=la0, loadavg_after=la1, finish_reason=resp["choices"][0].get("finish_reason"),
               output=resp["choices"][0]["message"]["content"])
    print(json.dumps({k: row[k] for k in ("tag", "screencap_s", "http_e2e_s", "usage", "timings")}), flush=True)
    return row

if __name__ == "__main__":
    INP.mkdir(exist_ok=True); rows = []
    la_start = loadavg()
    last = None
    for i in range(N):
        png, cs = screencap()
        last = (png, cs)
        rows.append(run(f"shot{i}", png, cs, False, i))
    # extras: same payload as last request with prompt cache enabled (2x) -> shows KV/prompt-cache reuse
    for j in range(2):
        rows.append(run(f"shot{N-1}_cache{j}", last[0], last[1], True, N + j))
    (RES / "requests.json").write_text(json.dumps(dict(loadavg_start=la_start, loadavg_end=loadavg(), resize=RESIZE, prompt=PROMPT, rows=rows), indent=2, ensure_ascii=False))
    print("saved")
