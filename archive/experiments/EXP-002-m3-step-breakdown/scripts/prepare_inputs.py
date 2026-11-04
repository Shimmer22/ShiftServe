"""Prepare real GUI screenshots (resized to Qwen2.5-VL 28px multiples) for EXP-002.

Sources are existing workspace screenshots (referenced, not modified):
  - AndroidWorld emulator home screen (mobile UI, 1080x2400)
  - ScreenSpot-v2 desktop screenshots from the 2026-10 prune-launch experiment
Outputs are small resized PNGs + a manifest with source SHA256 and size.
"""
import hashlib, json, pathlib
from PIL import Image

EXP = pathlib.Path(__file__).resolve().parents[2]
ROOT = EXP.parent
PL = ROOT / "2026-10-qwen38-prune-launch" / "results"
AW = PL / "androidworld-setup-20261024-v1" / "home.png"
SS = PL / "pvst-thinking-screenspot-inputs-20261017-v2" / "originals"
OUT = EXP / "EXP-002-m3-step-breakdown" / "inputs"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

specs = []
for name, w, h in [("aw_home_S", 336, 756), ("aw_home_M", 448, 980), ("aw_home_L", 560, 1232)]:
    specs.append((name, AW, w, h, "mobile(AndroidWorld emulator)"))
desk = sorted(SS.glob("pc_*.png"))[:2]
for i, p in enumerate(desk):
    specs.append((f"desk{i}_M", p, 784, 476, "desktop(ScreenSpot-v2)"))

man = []
OUT.mkdir(parents=True, exist_ok=True)
for name, src, w, h, kind in specs:
    im = Image.open(src).convert("RGB")
    sw, sh = im.size
    im.resize((w, h), Image.LANCZOS).save(OUT / f"{name}.png")
    man.append(dict(name=name, kind=kind, source=str(src.relative_to(ROOT)), source_sha256=sha(src),
                    source_size=[sw, sh], resized=[w, h],
                    expected_image_tokens=(w // 28) * (h // 28),
                    resized_sha256=sha(OUT / f"{name}.png")))
(OUT / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
for m in man: print(m["name"], m["resized"], m["expected_image_tokens"], m["source"])
