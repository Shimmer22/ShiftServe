"""EXP-001 analysis: raw per-trial frame logs -> CSV, per-class stats (mean/median/p90) and distribution plot.

Per trial (all times from one on-device clock, CLOCK_REALTIME ns; t_inj_end = when `input`/`am` returned):
  frames F_i = (capture_start s_i, capture_end e_i, md5_i). Stable = first k with md5_k..md5_{k+N-1} identical
  and md5_k != baseline. The last visible change lies between capture of F_{k-1} and capture of F_k, so
    lower = s_{k-1}-t_end   (change cannot be earlier than F_{k-1} start... conservative bracket)
    upper = e_k  -t_end     (what a screenshot-polling agent would observe: includes capture cost)
    mid   = (lower+upper)/2 (point estimate with screenshot overhead largely removed; +-(upper-lower)/2)
  status: ok | no_change (never differed from baseline) | timeout (never reached N identical frames)
Usage: python analyze_m1.py <raw.json> [--out-prefix results/m1]   (use overhead/noop raw to get capture-cost stats)
"""
import json, sys, csv, pathlib, statistics as st
import numpy as np

N = 3

def parse(raw):
    base = None; inj = None; fr = []
    for ln in raw.splitlines():
        p = ln.split()
        if not p: continue
        if p[0] == "BASE": base = p[1]
        elif p[0] == "INJ": inj = (int(p[1]), int(p[2]))
        elif p[0] == "F" and len(p) >= 5: fr.append((int(p[2]), int(p[3]), p[4]))
    return base, inj, fr

def analyze(tr):
    base, inj, fr = parse(tr["raw"])
    row = dict(cls=tr["cls"], iter=tr["iter"], inject=tr["inject"], status="parse_error", n_frames=len(fr))
    if base is None or inj is None or not fr: return row
    t0, t1 = inj
    row["inject_ms"] = round((t1 - t0) / 1e6, 1)
    caps = [(e - s) / 1e6 for s, e, _ in fr]; row["mean_capture_ms"] = round(float(np.mean(caps)), 1)
    H = [h for _, _, h in fr]
    k = next((i for i in range(len(H) - N + 1) if len(set(H[i:i + N])) == 1 and H[i] != base), None)
    if k is None:
        row["status"] = "no_change" if all(h == base for h in H) else "timeout"; return row
    prev_s = fr[k - 1][0] if k > 0 else t1
    lower = max(0.0, (prev_s - t1) / 1e6); upper = (fr[k][1] - t1) / 1e6
    row.update(status="ok", lower_ms=round(lower, 1), upper_ms=round(upper, 1), mid_ms=round((lower + upper) / 2, 1),
               from_inject_start_mid_ms=round((lower + upper) / 2 + (t1 - t0) / 1e6, 1))
    return row

def stats(v):
    v = np.array(v, float)
    return dict(n=len(v), mean=round(float(v.mean()), 1), median=round(float(np.median(v)), 1), p90=round(float(np.percentile(v, 90)), 1))

if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]); pref = sys.argv[sys.argv.index("--out-prefix") + 1] if "--out-prefix" in sys.argv else str(src.with_suffix(""))
    trials = json.loads(src.read_text())
    rows = [analyze(t) for t in trials]
    keys = ["cls", "iter", "inject", "status", "n_frames", "inject_ms", "mean_capture_ms", "lower_ms", "mid_ms", "upper_ms", "from_inject_start_mid_ms"]
    with open(pref + "_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    classes = list(dict.fromkeys(r["cls"] for r in rows)); summ = {}
    for c in classes:
        rc = [r for r in rows if r["cls"] == c]; ok = [r for r in rc if r["status"] == "ok"]
        s = dict(trials=len(rc), ok=len(ok), status_counts={s: sum(r["status"] == s for r in rc) for s in {r["status"] for r in rc}},
                 capture_cost_ms=stats([r["mean_capture_ms"] for r in rc if "mean_capture_ms" in r]) if any("mean_capture_ms" in r for r in rc) else None)
        if ok:
            s["mid_ms"] = stats([r["mid_ms"] for r in ok]); s["upper_ms"] = stats([r["upper_ms"] for r in ok])
            s["lower_ms"] = stats([r["lower_ms"] for r in ok]); s["inject_ms"] = stats([r["inject_ms"] for r in ok])
        summ[c] = s
    pathlib.Path(pref + "_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    okc = [c for c in classes if summ[c]["ok"]]
    if okc:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        data = [[r["mid_ms"] for r in rows if r["cls"] == c and r["status"] == "ok"] for c in okc]
        ax[0].boxplot(data, labels=okc, showmeans=True)
        for i, d in enumerate(data, 1): ax[0].scatter(np.random.normal(i, .05, len(d)), d, s=8, alpha=.4)
        ax[0].set_ylabel("A2O gap, mid estimate (ms)"); ax[0].tick_params(axis="x", rotation=40); ax[0].set_title("per-class distribution")
        for c, d in zip(okc, data):
            d = np.sort(d); ax[1].step(d, np.arange(1, len(d) + 1) / len(d), where="post", label=f"{c} (n={len(d)})")
        ax[1].set_xlabel("A2O gap (ms)"); ax[1].set_ylabel("ECDF"); ax[1].legend(fontsize=7); ax[1].set_title("ECDF")
        plt.tight_layout(); plt.savefig(pref + "_distribution.png", dpi=140); print("plot ->", pref + "_distribution.png")
    else:
        print("no ok trials; no plot")
