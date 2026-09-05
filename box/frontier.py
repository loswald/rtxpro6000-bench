#!/usr/bin/env python3
"""Cost / quality frontier for every configuration measured on the 600 W box - each quantisation of a model is
its own point, because it is its own product.

x: dollars per million tokens (input + output) for the AVERAGE of the workloads actually measured for that
   configuration - the router shape (1,024 in / 128 out), the prompt-optimisation shape (3,072-token shared prefix
   + 512 in / 256 out) and the judge shape (4,096 in / 512 out) where they exist. Node cost is $4.40 per node-hour
   (Scan list, 70% utilisation) divided by the requests that hour serves at each shape; the API bill prices the
   same requests at OpenRouter list (5 Sept 2026) with input, cached input (each model's own cache-read price) and output
   priced separately. Both are averaged per request over the same two shapes (router and prompt-optimisation;
   the judge shape exists only for some configurations, so it is left out for comparability), so filled (node)
   and hollow (API) markers are the same workload at the same mix.
y: task accuracy over the 403-item suite.
Colour is the precision class (three validated categorical hues); marker shape separates 8-bit from 4-bit
post-training quantisation within the orange class. The step line is the frontier: no other point is both
cheaper and better. A dashed ring marks a point whose own quality run has not landed yet; it carries the
accuracy measured on the same weights and kernels in another layout ("own-tag|fallback-tag" in POINTS).
Writes report/frontier.svg and prints a markdown table of the points.
"""
import csv, json, os, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_USD_H = 4.40    # Scan list, 70% utilisation (README, Economics)
SHAPE_GEOM = {"router": (1024, 128, 0), "promptopt": (512, 256, 3072), "judge": (4096, 512, 0)}   # in, out, cached prefix
SHAPES_USED = ("router", "promptopt")   # the two shapes every configuration has; judge exists only for some, and a
                                        # point averaged over three shapes would not be comparable with one over two

# API list prices, $ per million tokens (input, output, cached input) - OpenRouter /models on 5 Sept 2026 (default
# routing; a model with no cache-read price bills cached input at the full input price). Qwen3.8-Flash-Next from AA.
API = {"Qwen3.8-27B": (0.42, 3.00, 0.085), "GLM-5.3-Flash": (0.075, 0.25, 0.015), "DeepSeek-V4-Flash": (0.065, 0.18, 0.016),
       "gpt-oss-120b": (0.037, 0.17, 0.037), "gpt-oss-20b": (0.03, 0.13, 0.03), "gemma-4-26B-A4B": (0.07, 0.34, 0.07),
       "Muse-Glimmer-30B": (0.30, 1.10, 0.04), "Qwen3.8-Flash-Next": (0.15, 0.47, 0.15), "MiniMax-M3": (0.30, 1.20, 0.06)}

# label, precision class, eval tag (trees searched: 600w, 600w2, top, 5090), throughput tag, API model
POINTS = [
    ("GLM-5.3-Flash NVFP4 · TP4",              "ptq4",   "glm53f_base",              "glm53f_tp4ep4_s512",        "GLM-5.3-Flash"),
    ("GLM-5.3-Flash NVFP4 · DP2 × TP2 + EP*",  "ptq4",   "glm53f_best|glm53f_base",  "glm53f_dp2tp2ep2_s384",     "GLM-5.3-Flash"),
    ("GLM-5.3-Flash NVFP4 · TP4 + MTP",        "ptq4",   "glm53f_mtp",               "glm53f_s512_mtp",           "GLM-5.3-Flash"),
    ("DeepSeek-V4-Flash native · TP4",         "native", "ds4flash_b12x-b12x",       "ds4flash_b12x-b12x",        "DeepSeek-V4-Flash"),
    ("DeepSeek-V4-Flash native · DP4 + EP*",   "native", "ds4flash_dp4ep4_s512_b12x--|ds4flash_b12x--", "ds4flash_dp4ep4_s512_b12x--", "DeepSeek-V4-Flash"),
    ("DeepSeek-V4-Flash native · TP4 + DSpark", "native", "ds4flash_dspark_b12x-b12x", "ds4flash_dspark_b12x-b12x", "DeepSeek-V4-Flash"),
    ("Qwen3.8-27B BF16",                       "native", "q27_bf16_---",             "q27_bf16_---",              "Qwen3.8-27B"),
    ("Qwen3.8-27B QAT NVFP4 (W4A4)",           "qat4",   "q27_nvfp4_quasar_qat_b12x--", "q27_nvfp4_quasar_qat_b12x--", "Qwen3.8-27B"),
    ("Qwen3.8-27B FP8",                        "ptq8",   "q27_fp8_b12x--",           "q27_fp8_b12x--",            "Qwen3.8-27B"),
    ("Qwen3.8-27B gittensor NVFP4 (W4A4)",     "ptq4",   "q27_nvfp4_b12x--",         "q27_nvfp4_b12x--",          "Qwen3.8-27B"),
    ("Qwen3.8-27B RedHat NVFP4 (W4A16)",       "ptq4",   "q27_nvfp4_redhat_---",     "q27_nvfp4_redhat_---",      "Qwen3.8-27B"),
    ("Qwen3.8-27B unsloth NVFP4 (W4A16)",      "ptq4",   "q27_nvfp4_unsloth_---",    "q27_nvfp4_unsloth_---",     "Qwen3.8-27B"),
    ("Muse-Glimmer-30B BF16",                  "native", "muse30_---",               "muse30_---",                "Muse-Glimmer-30B"),
    ("gemma-4-26B-A4B BF16 (thinking, T=0)",   "native", "gemma26_---",              "gemma26_---",               "gemma-4-26B-A4B"),
    ("gpt-oss-120b MXFP4 (native)",            "native", "gptoss120_tp4_--flashinfer_cutlass", "full_gptoss",     "gpt-oss-120b"),
    ("gpt-oss-20b MXFP4 (native)",             "native", "gptoss20_--flashinfer_cutlass", "sw_gptoss_ficutlass_mxfp8", "gpt-oss-20b"),
    ("Qwen3.8-Flash-Next NVFP4 · TP4",         "ptq4",   "qwen38fn_tp4m_--marlin|qwen38fn_tp4mb_b12x-marlin", "qwen38fn_tp4m_--marlin", "Qwen3.8-Flash-Next"),
    ("Qwen3.8-Flash-Next NVFP4 · TP4, W4A4 linears", "ptq4", "qwen38fn_tp4mb_b12x-marlin|qwen38fn_tp4m_--marlin", "qwen38fn_tp4mb_b12x-marlin", "Qwen3.8-Flash-Next"),
]
# shorter on-chart names for the crowded top of the plane; the table keeps the full labels
SHORT = {"GLM-5.3-Flash NVFP4 · TP4": "GLM-5.3-Flash · TP4", "GLM-5.3-Flash NVFP4 · DP2 × TP2 + EP*": "GLM-5.3-Flash · DP2×TP2+EP*",
         "GLM-5.3-Flash NVFP4 · TP4 + MTP": "GLM-5.3-Flash · TP4+MTP", "DeepSeek-V4-Flash native · TP4": "DeepSeek-V4-Flash · TP4",
         "DeepSeek-V4-Flash native · DP4 + EP*": "DeepSeek-V4-Flash · DP4+EP*", "DeepSeek-V4-Flash native · DP4 + EP": "DeepSeek-V4-Flash · DP4+EP",
         "GLM-5.3-Flash NVFP4 · DP2 × TP2 + EP": "GLM-5.3-Flash · DP2×TP2+EP", "DeepSeek-V4-Flash native · TP4 + DSpark": "DeepSeek-V4-Flash · TP4+DSpark",
         "gemma-4-26B-A4B BF16 (thinking, T=0)": "gemma-4-26B-A4B BF16 (thinking)",
         "Qwen3.8-Flash-Next NVFP4 · TP4": "Qwen3.8-Flash-Next · TP4", "Qwen3.8-Flash-Next NVFP4 · TP4, W4A4 linears": "Qwen3.8-Flash-Next · TP4 W4A4"}
CLASS = {"native": ("native precision", "#2a78d6", "circle"), "qat4": ("quantisation-aware 4-bit", "#1baf7a", "circle"),
         "ptq4": ("post-training 4-bit", "#eb6834", "circle"), "ptq8": ("post-training 8-bit (FP8)", "#eb6834", "square")}

def acc_for(tag):
    """(accuracy, n_items, provisional). 'own|fallback' uses the own run when it exists, else the fallback, flagged."""
    if "|" in tag:
        own, fb = tag.split("|", 1)
        a, n, _ = acc_for(own)
        if a is not None:
            return a, n, False
        a, n, _ = acc_for(fb)
        return a, n, True
    for tree in ("600w", "600w2", "", "5090"):
        p = os.path.join(ROOT, "results", tree, "eval", tag + ".json") if tree else os.path.join(ROOT, "results", "eval", tag + ".json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8")); a = d.get("aggregate", {})
            v = a.get("acc_micro") or a.get("acc")
            if v is not None:
                return float(v), int(a.get("n_scored") or 0), False
    return None, 0, False

_ROWS = None
def rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "summary_all.tsv"), encoding="utf-8"), delimiter="\t")
                 if r.get("host") == "pro6000-s600w"]
    return _ROWS

def shapes_for(tag):
    """Best measured row per shape (router, promptopt, judge) for a throughput tag: {label: out_tps}."""
    best = {}
    for r in rows():
        if r.get("tag") != tag or r.get("label") not in SHAPES_USED:
            continue
        try: v = float(r["out_tps"])
        except (KeyError, ValueError): continue
        if v > best.get(r["label"], 0):
            best[r["label"]] = v
    return best

def cost_avg(tag, model):
    """Average over the measured shapes of node cost and API bill per request; returns $/M tokens for both,
    the shapes used and the average tokens per request (input incl. cached, output)."""
    sh = shapes_for(tag)
    if not sh:
        return None
    p_in, p_out, p_cache = API[model]
    node_req, api_req, tok_req, tin, tout = [], [], [], [], []
    for label, out_tps in sh.items():
        n_in, n_out, n_pre = SHAPE_GEOM[label]
        req_s = out_tps / n_out
        node_req.append(NODE_USD_H / 3600 / req_s)
        api_req.append((n_in * p_in + n_pre * p_cache + n_out * p_out) / 1e6)
        tok_req.append(n_in + n_pre + n_out); tin.append(n_in + n_pre); tout.append(n_out)
    k = len(sh)
    node_m = sum(node_req) / k / (sum(tok_req) / k) * 1e6
    api_m = sum(api_req) / k / (sum(tok_req) / k) * 1e6
    return dict(node=node_m, api=api_m, shapes=sorted(sh, key=lambda l: list(SHAPE_GEOM).index(l)),
                avg_in=sum(tin) / k, avg_out=sum(tout) / k, tput=sh.get("router"))

def place_labels(pts, X, Y, W, H, ml, mr, mt, mb, keep):
    """Greedy label placement: try anchor positions around each marker, take the first whose text box overlaps
    neither an already-placed label, nor any marker, nor a reserved rectangle (the legend)."""
    FS, CW = 11.5, 6.1
    boxes = list(keep)
    for p in pts:
        for cx in (X(p["cost"]), X(p["api"])):
            cy = Y(p["acc"]); boxes.append((cx - 8, cy - 8, cx + 8, cy + 8))
    def hits(b):
        return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in boxes)
    out = []
    for p in sorted(pts, key=lambda q: -q["acc"]):
        x, y = X(p["cost"]), Y(p["acc"]); w = len(SHORT.get(p["label"], p["label"])) * CW
        cands = []
        for dy in (-6, 15, -20, 29, -34, 43, -48, 57, -62, 71, -76, 85):
            cands += [(x + 9, y + dy, "start"), (x - 9, y + dy, "end"), (x + 40, y + dy, "start"), (x - 40, y + dy, "end")]
        chosen = None
        for tx, ty, anchor in cands:
            x0 = tx if anchor == "start" else tx - w
            b = (x0, ty - FS, x0 + w, ty + 3)
            if b[0] < ml - 60 or b[2] > W - 4 or b[1] < mt - 6 or b[3] > H - mb: continue
            if not hits(b):
                chosen = (tx, ty, anchor, b); break
        if chosen is None:
            tx, ty, anchor = cands[0]; chosen = (tx, ty, anchor, (tx, ty - FS, tx + w, ty + 3))
            print(f"  label overlap: {p['label']}")
        boxes.append(chosen[3]); out.append((p, chosen))
    return out

def main():
    pts = []
    for label, cls, etag, ttag, model in POINTS:
        acc, n, prov = acc_for(etag); c = cost_avg(ttag, model)
        if acc is None or c is None:
            print(f"  skip {label}: acc={acc} shapes={None if c is None else c['shapes']}")
            continue
        if not prov and label.endswith("*"):      # the point's own run has landed: drop the provisional mark
            label = label[:-1].rstrip()
        pts.append(dict(label=label, cls=cls, acc=acc, n=n, cost=c["node"], api=c["api"], shapes=c["shapes"],
                        avg_in=c["avg_in"], avg_out=c["avg_out"], tput=c["tput"], prov=prov))
    front, best = [], -1
    for p in sorted(pts, key=lambda q: q["cost"]):
        if p["acc"] > best:
            front.append(p); best = p["acc"]
    # ---- SVG ----
    W, H, ml, mr, mt, mb = 1100, 600, 70, 30, 30, 70
    xs = [p["cost"] for p in pts] + [p["api"] for p in pts]
    xmin, xmax = 10 ** math.floor(math.log10(min(xs)) - 0.15), 10 ** math.ceil(math.log10(max(xs)) + 0.05)
    ymin, ymax = 0.60, 0.86
    X = lambda c: ml + (math.log10(c) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (W - ml - mr)
    Y = lambda a: mt + (ymax - a) / (ymax - ymin) * (H - mt - mb)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;font-family:system-ui,-apple-system,\'Segoe UI\',sans-serif;background:#fcfcfb" role="img" aria-label="Cost against quality for every configuration measured at 600 W">',
         f'<title>Cost against quality, 4x RTX PRO 6000 at 600 W</title>']
    t = xmin
    while t <= xmax * 1.0001:
        for k in (1, 2, 5):
            v = t * k
            if xmin <= v <= xmax:
                o.append(f'<line x1="{X(v):.1f}" y1="{mt}" x2="{X(v):.1f}" y2="{H-mb}" stroke="#e1e0d9" stroke-width="1"/>')
                lab = f"${v:g}" if v >= 1 else (f"${v:.2f}".rstrip("0") if v >= 0.01 else f"${v:.3f}")
                o.append(f'<text x="{X(v):.1f}" y="{H-mb+18}" font-size="12" fill="#898781" text-anchor="middle">{lab}</text>')
        t *= 10
    a = ymin
    while a <= ymax + 1e-9:
        o.append(f'<line x1="{ml}" y1="{Y(a):.1f}" x2="{W-mr}" y2="{Y(a):.1f}" stroke="#e1e0d9" stroke-width="1"/>')
        o.append(f'<text x="{ml-8}" y="{Y(a)+4:.1f}" font-size="12" fill="#898781" text-anchor="end">{a:.2f}</text>')
        a += 0.05
    o.append(f'<line x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}" stroke="#c3c2b7"/>')
    o.append(f'<text x="{(ml+W-mr)/2:.0f}" y="{H-mb+42}" font-size="13" fill="#52514e" text-anchor="middle">$ per million tokens, input + output, averaged over the measured workloads · filled = node at Scan list, 70% utilisation · hollow = API list at the same mix, cached input at its own price</text>')
    o.append(f'<text transform="translate(18,{(mt+H-mb)/2:.0f}) rotate(-90)" font-size="13" fill="#52514e" text-anchor="middle">task accuracy, 403 items</text>')
    L = [(k, CLASS[k]) for k in CLASS] + [("api", ("API list price, same model, same workload mix", "#52514e", "hollow")),
         ("front", ("frontier: nothing is both cheaper and better", "#898781", "dash")),
         ("prov", ("* quality from the same weights and kernels in another layout; own run in progress", "#52514e", "ring"))]
    lw, lh = 480, 18 * len(L) + 12
    lx0, ly0 = W - mr - lw - 6, H - mb - lh - 6
    o.append(f'<rect x="{lx0}" y="{ly0}" width="{lw}" height="{lh}" fill="#fcfcfb" fill-opacity="0.92" stroke="#e1e0d9"/>')
    for i, (k, (name, col, shp)) in enumerate(L):
        yy = ly0 + 14 + i * 18; lx = lx0 + 10
        if shp == "square": o.append(f'<rect x="{lx}" y="{yy-5}" width="10" height="10" fill="{col}"/>')
        elif shp == "hollow": o.append(f'<circle cx="{lx+5}" cy="{yy}" r="5" fill="#fcfcfb" stroke="{col}" stroke-width="2"/>')
        elif shp == "dash": o.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+10}" y2="{yy}" stroke="{col}" stroke-width="2" stroke-dasharray="6 4"/>')
        elif shp == "ring": o.append(f'<circle cx="{lx+5}" cy="{yy}" r="7" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="3 2"/>')
        else: o.append(f'<circle cx="{lx+5}" cy="{yy}" r="5" fill="{col}"/>')
        o.append(f'<text x="{lx+18}" y="{yy+4}" font-size="12" fill="#52514e">{name}</text>')
    if len(front) > 1:
        d = " ".join(f'{"M" if i==0 else "L"}{X(p["cost"]):.1f},{Y(p["acc"]):.1f}' for i, p in enumerate(front))
        o.append(f'<path d="{d}" fill="none" stroke="#898781" stroke-width="2" stroke-dasharray="6 4"/>')
    for p in pts:
        name, col, shp = CLASS[p["cls"]]
        x, y, xa = X(p["cost"]), Y(p["acc"]), X(p["api"])
        tip = f'{p["label"]}: {p["acc"]:.3f} on {p["n"]} items · ${p["cost"]:.3f}/M tokens self-hosted · ${p["api"]:.3f}/M API · shapes {", ".join(p["shapes"])}'
        o.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{xa:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="1" stroke-opacity="0.45"/>')
        o.append(f'<circle cx="{xa:.1f}" cy="{y:.1f}" r="5" fill="#fcfcfb" stroke="{col}" stroke-width="2"><title>{tip}</title></circle>')
        if p["prov"]:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="3 2"/>')
        if shp == "square":
            o.append(f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" fill="{col}" stroke="#fcfcfb" stroke-width="2"><title>{tip}</title></rect>')
        else:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}" stroke="#fcfcfb" stroke-width="2"><title>{tip}</title></circle>')
    for p, (tx, ty, anchor, b) in place_labels(pts, X, Y, W, H, ml, mr, mt, mb, [(lx0, ly0, lx0 + lw, ly0 + lh)]):
        x, y = X(p["cost"]), Y(p["acc"])
        lx_ = b[0] if anchor == "start" else b[2]
        if abs(ty - y) > 12 or abs(lx_ - x) > 14:
            o.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{lx_:.1f}" y2="{(b[1]+b[3])/2:.1f}" stroke="#898781" stroke-width="0.8"/>')
        o.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="11.5" fill="#0b0b0b" text-anchor="{anchor}">{SHORT.get(p["label"], p["label"])}</text>')
    o.append("</svg>")
    out = os.path.join(ROOT, "report", "frontier.svg")
    open(out, "w", encoding="utf-8").write("\n".join(o))
    print(f"wrote {out} with {len(pts)} points; frontier: " + " → ".join(p["label"] for p in front))
    print("\n| configuration | accuracy (items) | workloads averaged | avg tokens / request (in · out) | node $/M tokens | API $/M tokens, same mix | API ÷ node | on the frontier |")
    print("|---|---:|---|---:|---:|---:|---:|:-:|")
    for p in sorted(pts, key=lambda q: q["cost"]):
        note = ", same kernels in another layout" if p["prov"] else ""
        sh = " + ".join(p["shapes"])
        print(f"| {p['label']} | {p['acc']:.3f} ({p['n']}{note}) | {sh} | {p['avg_in']:,.0f} · {p['avg_out']:.0f} | ${p['cost']:.3f} | ${p['api']:.3f} | {p['api']/p['cost']:.1f}× | {'yes' if p in front else ''} |")

if __name__ == "__main__":
    main()
