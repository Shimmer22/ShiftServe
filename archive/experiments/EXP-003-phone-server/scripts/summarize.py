"""EXP-003: merge results/requests.json with ViT/image-decode times parsed from results/server.log -> results/requests_table.csv."""
import json, re, csv, pathlib
R = pathlib.Path(__file__).resolve().parents[1] / "results"
d = json.loads((R / "requests.json").read_text()); log = (R / "server.log").read_text()
vit = [int(x) for x in re.findall(r"image slice encoded in (\d+) ms", log)]
dec = [int(x) for x in re.findall(r"image decoded \(batch .*?\) in (\d+) ms", log)]
rows = []; k = 0
for r in d["rows"]:
    t = r["timings"]; has_img = t["cache_n"] == 0
    v = vit[k] if has_img else None; di = dec[k] if has_img else None
    k += has_img
    rows.append(dict(tag=r["tag"], cache_prompt=r["cache_prompt"], screencap_s=r["screencap_s"], resize_s=r["resize_encode_s"],
        request_kB=round(r["request_json_bytes"] / 1024), http_e2e_s=r["http_e2e_s"], server_prompt_ms=t["prompt_ms"], prompt_n=t["prompt_n"], cache_n=t["cache_n"],
        vit_ms=v, image_decode_ms=di, predicted_ms=t["predicted_ms"], predicted_n=t["predicted_n"],
        http_minus_server_s=round(r["http_e2e_s"] - (t["prompt_ms"] + t["predicted_ms"]) / 1000, 3), e2e_incl_screencap_s=r["e2e_incl_screencap_s"],
        load1_before=r["loadavg_before"][0], load1_after=r["loadavg_after"][0]))
with open(R / "requests_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
