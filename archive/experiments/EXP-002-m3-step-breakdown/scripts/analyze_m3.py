"""EXP-002 analysis: results/m3_runs_t4.json -> CSV + summary + stacked-bar plot. Only measured/derived-from-measured values."""
import json, csv, pathlib, numpy as np
EXP = pathlib.Path(__file__).resolve().parents[1]; R = EXP / "results"
runs = [r for r in json.loads((R / "m3_runs_t4.json").read_text()) if r["kind"] == "image" and r["vit_ms"] is not None]
man = {m["name"]: m for m in json.loads((EXP / "inputs" / "manifest.json").read_text())}
rows = []
for r in runs:
    text_tok = r["perf_prompt_tokens"] - r["image_tokens"]
    resid = r["perf_prompt_eval_ms"] - r["vit_ms"] - r["image_prefill_ms"]   # DERIVED, see README caveat
    rows.append(dict(step=r["step"], rep=r["rep"], kind=man[r["step"]]["kind"], image_wh="x".join(map(str, man[r["step"]]["resized"])),
        threads=r["threads"], image_tokens=r["image_tokens"], text_tokens=text_tok, decode_tokens=r["decode_tokens"],
        vit_ms=r["vit_ms"], image_prefill_ms=r["image_prefill_ms"], perf_prompt_eval_ms=r["perf_prompt_eval_ms"],
        text_prefill_residual_ms=round(resid, 1), decode_ms=r["decode_ms"], decode_ms_per_tok=round(r["decode_ms"] / r["decode_tokens"], 1),
        load_ms=r["load_ms"], wall_s=r["wall_s"], load1_before=r["ctx_before"]["load1_5_15"][0], load1_after=r["ctx_after"]["load1_5_15"][0],
        mem_avail_mb_before=round(r["ctx_before"]["mem_available_kb"] / 1024), batt_temp_deci_c_after=r["ctx_after"]["battery_temp_deci_c"]))
with open(R / "m3_steps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
S = {}
for s in dict.fromkeys(r["step"] for r in rows):
    g = [r for r in rows if r["step"] == s]; m = lambda k: [x[k] for x in g]
    S[s] = dict(n=len(g), image_tokens=g[0]["image_tokens"], text_tokens=g[0]["text_tokens"], decode_tokens=m("decode_tokens"),
        **{k + "_mean_ms": round(float(np.mean(m(k))), 0) for k in ["vit_ms", "image_prefill_ms", "text_prefill_residual_ms", "decode_ms"]},
        vit_min_max=[min(m("vit_ms")), max(m("vit_ms"))], image_prefill_min_max=[min(m("image_prefill_ms")), max(m("image_prefill_ms"))])
(R / "m3_summary.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names = list(S); fig, ax = plt.subplots(figsize=(9, 5)); b = np.zeros(len(names))
for k, lab in [("vit_ms", "ViT / vision encode"), ("image_prefill_ms", "image-token prefill"), ("text_prefill_residual_ms", "text prefill (derived residual)"), ("decode_ms", "decode")]:
    v = np.array([S[n][k + "_mean_ms"] if k != "decode_ms" else S[n]["decode_ms_mean_ms"] for n in names]) / 1000
    ax.bar(names, v, bottom=b, label=lab); b += v
ax.set_xticklabels([f"{n}\n{S[n]['image_tokens']} img tok, n={S[n]['n']}" for n in names], fontsize=8)
ax.set_ylabel("seconds (mean)"); ax.set_title("Qwen2.5-VL-3B Q4_K_M, CPU only, 4 threads, Xiaomi 15 (loaded device)"); ax.legend(); plt.tight_layout()
plt.savefig(R / "m3_stage_breakdown.png", dpi=140)
