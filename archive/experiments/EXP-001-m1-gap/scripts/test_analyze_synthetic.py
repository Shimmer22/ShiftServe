"""Logic check of analyze_m1.analyze on a SYNTHETIC trial (not a measurement; nothing saved to results/)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from analyze_m1 import analyze
ms = 1_000_000; t0 = 0; t1 = 100 * ms
# frames (start,end,hash): change visible between frame 2 (A->B at hash 'b' first seen idx2), animation 'b','c', stable 'd' x3 from idx 4
F = [(110, 250, "a"), (260, 400, "a"), (410, 550, "b"), (560, 700, "c"), (710, 850, "d"), (860, 1000, "d"), (1010, 1150, "d")]
raw = "BASE a\nINJ %d %d\n" % (t0, t1) + "\n".join("F %d %d %d %s" % (i, s * ms, e * ms, h) for i, (s, e, h) in enumerate(F))
r = analyze(dict(cls="t", iter=0, inject="x", raw=raw)); print(r)
assert r["status"] == "ok" and r["lower_ms"] == 560 - 100 and r["upper_ms"] == 850 - 100, r
print("OK")
