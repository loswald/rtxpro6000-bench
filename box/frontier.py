#!/usr/bin/env python3
"""Cost / quality frontier for every configuration measured on the 600 W box - each quantisation of a model is
its own point, because it is its own product.

x: dollars per million OUTPUT tokens, self-hosted, at Scan list and 70% utilisation ($4.40 per node-hour), from
   our measured 600 W throughput at the router shape (1,024 in / 128 out; the 8x input tokens ride in the same hour).
   Hollow markers at the same height: the API list price for the same model's output tokens (OpenRouter, 5 Sept 2026).
y: task accuracy over the 403-item suite.
Colour is the precision class (three validated categorical hues); marker shape separates 8-bit from 4-bit
post-training quantisation within the orange class. The step line is the frontier: no other point is both
cheaper and better. Writes report/frontier.svg and prints a markdown table of the points.
"""
import csv, glob, json, os, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_USD_H = 4.40   # Scan list, 70% utilisation (README, Economics)

# label, precision class, eval tag (tree searched in order 600w, 600w2, 400w), throughput tag, shape, C, api $/M out
POINTS = [
    ("GLM-5.3-Flash NVFP4 · TP4",            "ptq4",   "glm53f_base",              "glm53f_tp4ep4_s512",        "router", 1024, 0.25),
    ("GLM-5.3-Flash NVFP4 · TP4 + MTP",      "ptq4",   "glm53f_mtp",               "glm53f_s512_mtp",           "router", 1024, 0.25),
    ("DeepSeek-V4-Flash native · TP4",       "native", "ds4flash_b12x-b12x",       "ds4flash_b12x-b12x",        "router", 256,  0.18),
    # the DP4 + EP layout's own 403-item run is in progress; until it lands the point carries the quality measured
    # on the same weights and MoE kernel (Marlin MXFP4) at TP4, and is drawn with a dashed ring to say so
    ("DeepSeek-V4-Flash native · DP4 + EP*",  "native", "ds4flash_dp4ep4_s512_b12x--|ds4flash_b12x--", "ds4flash_dp4ep4_s512_b12x--", "router", 1024, 0.18),
    ("DeepSeek-V4-Flash native · TP4 + DSpark", "native", "ds4flash_dspark_b12x-b12x", "ds4flash_dspark_b12x-b12x", "router", 256, 0.18),
    ("Qwen3.8-27B BF16",                     "native", "q27_bf16_---",             "q27_bf16_---",              "router", 1024, 3.00),
    ("Qwen3.8-27B QAT NVFP4 (W4A4)",         "qat4",   "q27_nvfp4_quasar_qat_b12x--", "q27_nvfp4_quasar_qat_b12x--", "router", 1024, 3.00),
    ("Qwen3.8-27B FP8",                      "ptq8",   "q27_fp8_b12x--",           "q27_fp8_b12x--",            "router", 1024, 3.00),
    ("Qwen3.8-27B gittensor NVFP4 (W4A4)",   "ptq4",   "q27_nvfp4_b12x--",         "q27_nvfp4_b12x--",          "router", 1024, 3.00),
    ("Qwen3.8-27B RedHat NVFP4 (W4A16)",     "ptq4",   "q27_nvfp4_redhat_---",     "q27_nvfp4_redhat_---",      "router", 1024, 3.00),
    ("Qwen3.8-27B unsloth NVFP4 (W4A16)",    "ptq4",   "q27_nvfp4_unsloth_---",    "q27_nvfp4_unsloth_---",     "router", 1024, 3.00),
    ("Muse-Glimmer-30B BF16",                "native", "muse30_---",               "muse30_---",                "router", 1024, 1.10),
    ("gemma-4-26B-A4B BF16 (thinking, T=0)", "native", "gemma26_---",              "gemma26_---",               "router", 1024, 0.34),
    ("gpt-oss-120b MXFP4 (native)",          "native", "gptoss120_tp4_--flashinfer_cutlass", "full_gptoss",     "router", 2048, 0.17),
    ("gpt-oss-20b MXFP4 (native)",           "native", "gptoss20_--flashinfer_cutlass", "sw_gptoss_ficutlass_mxfp8", "router", 2048, 0.13),
]
CLASS = {"native": ("native precision", "#2a78d6", "circle"), "qat4": ("quantisation-aware 4-bit", "#1baf7a", "circle"),
         "ptq4": ("post-training 4-bit", "#eb6834", "circle"), "ptq8": ("post-training 8-bit (FP8)", "#eb6834", "square")}

def acc_for(tag):
    prov = False
    if "|" in tag:            # "own-tag|fallback-tag": use the run's own result when it exists, else the fallback (provisional)
        own, fb = tag.split("|", 1)
        a, n, _ = acc_for(own)
        if a is not None: return a, n, False
        a, n, _ = acc_for(fb); return a, n, True
    for tree in ("600w", "600w2", "", "5090"):
        p = os.path.join(ROOT, "results", tree, "eval", tag + ".json") if tree else os.path.join(ROOT, "results", "eval", tag + ".json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8")); a = d.get("aggregate", {})
            v = a.get("acc_micro") or a.get("acc")
            if v is not None:
                return float(v), int(a.get("n_scored") or 0), False
    return None, 0, False

def tput_for(tag, label, conc):
    best = 0.0
    for r in csv.DictReader(open(os.path.join(ROOT, "results", "summary_all.tsv"), encoding="utf-8"), delimiter="\t"):
        if r.get("host") != "pro6000-s600w" or r.get("tag") != tag or r.get("label") != label or str(r.get("C")) != str(conc):
            continue
        try: best = max(best, float(r["out_tps"]))
        except (KeyError, ValueError): pass
    return best or None

def main():
    pts = []
    for label, cls, etag, ttag, shape, conc, api in POINTS:
        acc, n, prov = acc_for(etag); tp = tput_for(ttag, shape, conc)
        if acc is None or not tp:
            print(f"  skip {label}: acc={acc} tput={tp}")
            continue
        cost = NODE_USD_H / (tp * 3600 / 1e6)
        pts.append(dict(label=label, cls=cls, acc=acc, n=n, tput=tp, cost=cost, api=api, prov=prov))
    # frontier: sort by cost, keep points whose accuracy exceeds every cheaper point's
    front, best = [], -1
    for p in sorted(pts, key=lambda q: q["cost"]):
        if p["acc"] > best:
            front.append(p); best = p["acc"]
    # ---- SVG ----
    W, H, ml, mr, mt, mb = 920, 560, 70, 30, 40, 70
    xs = [p["cost"] for p in pts] + [p["api"] for p in pts]
    xmin, xmax = 10 ** math.floor(math.log10(min(xs)) - 0.15), 10 ** math.ceil(math.log10(max(xs)) + 0.05)
    ymin, ymax = 0.60, 0.86
    X = lambda c: ml + (math.log10(c) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (W - ml - mr)
    Y = lambda a: mt + (ymax - a) / (ymax - ymin) * (H - mt - mb)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;font-family:system-ui,-apple-system,\'Segoe UI\',sans-serif;background:#fcfcfb" role="img" aria-label="Cost against quality for every configuration measured at 600 W">',
         f'<title>Cost against quality, 4x RTX PRO 6000 at 600 W</title>']
    # grid
    t = xmin
    while t <= xmax * 1.0001:
        for k in (1, 2, 5):
            v = t * k
            if xmin <= v <= xmax:
                o.append(f'<line x1="{X(v):.1f}" y1="{mt}" x2="{X(v):.1f}" y2="{H-mb}" stroke="#e1e0d9" stroke-width="1"/>')
                lab = f"${v:g}" if v >= 1 else f"${v:.2f}".rstrip("0")
                o.append(f'<text x="{X(v):.1f}" y="{H-mb+18}" font-size="12" fill="#898781" text-anchor="middle">{lab}</text>')
        t *= 10
    a = ymin
    while a <= ymax + 1e-9:
        o.append(f'<line x1="{ml}" y1="{Y(a):.1f}" x2="{W-mr}" y2="{Y(a):.1f}" stroke="#e1e0d9" stroke-width="1"/>')
        o.append(f'<text x="{ml-8}" y="{Y(a)+4:.1f}" font-size="12" fill="#898781" text-anchor="end">{a:.2f}</text>')
        a += 0.05
    o.append(f'<line x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}" stroke="#c3c2b7"/>')
    o.append(f'<text x="{(ml+W-mr)/2:.0f}" y="{H-mb+42}" font-size="13" fill="#52514e" text-anchor="middle">$ per million output tokens · filled = self-hosted at Scan list, 70% utilisation, from measured 600 W throughput · hollow = API list price</text>')
    o.append(f'<text transform="translate(18,{(mt+H-mb)/2:.0f}) rotate(-90)" font-size="13" fill="#52514e" text-anchor="middle">task accuracy, 403 items</text>')
    # frontier step line
    if len(front) > 1:
        d = " ".join(f'{"M" if i==0 else "L"}{X(p["cost"]):.1f},{Y(p["acc"]):.1f}' for i, p in enumerate(front))
        o.append(f'<path d="{d}" fill="none" stroke="#898781" stroke-width="2" stroke-dasharray="6 4"/>')
    # points
    for p in pts:
        name, col, shp = CLASS[p["cls"]]
        x, y, xa = X(p["cost"]), Y(p["acc"]), X(p["api"])
        o.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{xa:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="1" stroke-opacity="0.45"/>')
        o.append(f'<circle cx="{xa:.1f}" cy="{y:.1f}" r="5" fill="#fcfcfb" stroke="{col}" stroke-width="2"/>')
        if p["prov"]:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="3 2"/>')
        if shp == "square":
            o.append(f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" fill="{col}" stroke="#fcfcfb" stroke-width="2"><title>{p["label"]}: {p["acc"]:.3f} on {p["n"]} items · {p["tput"]:,.0f} out tok/s · ${p["cost"]:.2f}/M self-hosted · ${p["api"]:.2f}/M API</title></rect>')
        else:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}" stroke="#fcfcfb" stroke-width="2"><title>{p["label"]}: {p["acc"]:.3f} on {p["n"]} items · {p["tput"]:,.0f} out tok/s · ${p["cost"]:.2f}/M self-hosted · ${p["api"]:.2f}/M API</title></circle>')
        o.append(f'<text x="{x+9:.1f}" y="{y-7:.1f}" font-size="11.5" fill="#0b0b0b">{p["label"]}</text>')
    # legend
    lx, ly = ml + 8, mt + 6
    for i, (k, (name, col, shp)) in enumerate(CLASS.items()):
        yy = ly + i * 18
        if shp == "square": o.append(f'<rect x="{lx}" y="{yy-5}" width="10" height="10" fill="{col}"/>')
        else: o.append(f'<circle cx="{lx+5}" cy="{yy}" r="5" fill="{col}"/>')
        o.append(f'<text x="{lx+16}" y="{yy+4}" font-size="12" fill="#52514e">{name}</text>')
    o.append(f'<circle cx="{lx+5}" cy="{ly+4*18}" r="5" fill="#fcfcfb" stroke="#52514e" stroke-width="2"/><text x="{lx+16}" y="{ly+4*18+4}" font-size="12" fill="#52514e">API list price, same model</text>')
    o.append(f'<circle cx="{lx+5}" cy="{ly+6*18}" r="8" fill="none" stroke="#52514e" stroke-width="1.5" stroke-dasharray="3 2"/><text x="{lx+16}" y="{ly+6*18+4}" font-size="12" fill="#52514e">* quality from the same weights and kernels in another layout; own run in progress</text>')
    o.append(f'<line x1="{lx}" y1="{ly+5*18}" x2="{lx+10}" y2="{ly+5*18}" stroke="#898781" stroke-width="2" stroke-dasharray="6 4"/><text x="{lx+16}" y="{ly+5*18+4}" font-size="12" fill="#52514e">frontier: nothing is both cheaper and better</text>')
    o.append("</svg>")
    out = os.path.join(ROOT, "report", "frontier.svg")
    open(out, "w", encoding="utf-8").write("\n".join(o))
    print(f"wrote {out} with {len(pts)} points; frontier: " + " → ".join(p["label"] for p in front))
    print("\n| configuration | accuracy (items) | out tok/s (600 W) | $/M output, self-hosted | $/M output, API | on the frontier |")
    print("|---|---:|---:|---:|---:|:-:|")
    for p in sorted(pts, key=lambda q: q["cost"]):
        print(f"| {p['label']} | {p['acc']:.3f} ({p['n']}{', same kernels at TP4' if p['prov'] else ''}) | {p['tput']:,.0f} | ${p['cost']:.3f} | ${p['api']:.2f} | {'yes' if p in front else ''} |")

if __name__ == "__main__":
    main()
