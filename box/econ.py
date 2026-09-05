#!/usr/bin/env python3
"""Economics of the 4x RTX PRO 6000 node against API providers, from this repository's own measurements.

Cost bases (GBP ex-VAT, the Sqwish GPU Slack Lab decision model, 5 Sept 2026; FX GBP->USD 1.35, the rate the model
implies):
  list        Scan's website price for the four-card box, GBP 1,666.65 a month ex-VAT (GBP 1,999.98 inc-VAT),
              electricity included, 730 h a month, no discount, nothing resold, no relief
  committed   the same with the model's 25% commitment discount: GBP 1,250 per 30 days
  loaded      committed, minus ERIS (14.5% x 186% = 26.97p per qualifying GBP, 80% of the bill qualifying),
              minus idle GPU-hours resold on Vast as interruptible capacity (USD 0.90 per GPU-hour, 74.4% fill,
              25% platform take), plus GBP 15 of stopped-template storage - as a function of how many hours the
              owner keeps for itself

Per-model comparison: our measured 600 W throughput at a shape gives tokens per node-hour; the API bill for the
same token mix at today's OpenRouter list prices is what a provider would charge for that hour of output.
"""
import csv, collections, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FX = 1.35
LIST_GBP_MONTH = 1666.65
HOURS = 730.0
COMMIT_DISC = 0.25
ERIS_RATE = 0.145 * 1.86
ERIS_QUALIFYING = 0.80
VAST_PRICE, VAST_FILL, VAST_TAKE = 0.90, 0.744, 0.25
STORAGE_GBP = 15.0
GPUS = 4

def loaded_per_node_hour(owner_util):
    """GBP per node-hour the owner actually uses, at a given owner utilisation, 30-day period."""
    committed = LIST_GBP_MONTH * (1 - COMMIT_DISC) * (30 * 24 / HOURS)  # scale month -> 30 days
    gpu_hours = GPUS * 30 * 24
    owner_gpu_h = gpu_hours * owner_util
    idle_gpu_h = gpu_hours - owner_gpu_h
    vast_income_gbp = idle_gpu_h * VAST_FILL * VAST_PRICE * (1 - VAST_TAKE) / FX
    eris = committed * ERIS_QUALIFYING * ERIS_RATE
    cost = committed - vast_income_gbp + STORAGE_GBP - eris
    return cost / (owner_gpu_h / GPUS), cost

# API list prices, USD per million tokens (OpenRouter, 5 Sept 2026; Qwen3.8-Flash-Next from Artificial Analysis)
API = {
    "Qwen3.8-27B": (0.42, 3.00), "GLM-5.3-Flash": (0.075, 0.25), "DeepSeek-V4-Flash": (0.065, 0.18),
    "gpt-oss-120b": (0.037, 0.17), "gpt-oss-20b": (0.03, 0.13), "gemma-4-26B-A4B": (0.07, 0.34),
    "Muse-Glimmer-30B": (0.30, 1.10), "MiniMax-M3": (0.30, 1.20), "Qwen3.8-Flash-Next": (0.15, 0.47),
    "Nemotron-3-Super": (0.085, 0.40),
}
# which measured tag/shape stands for each model on the 600 W host (tag prefix, label, concurrency or None=best)
ROWS = [
    ("Qwen3.8-27B", "NVFP4 gittensor, b12x W4A4, 4 replicas", "q27_nvfp4_b12x--", "router"),
    ("Qwen3.8-27B", "FP8, b12x, 4 replicas", "q27_fp8_b12x--", "router"),
    ("Qwen3.8-27B", "NVFP4 QUASAR-QAT, b12x W4A4, 4 replicas", "q27_nvfp4_quasar_qat_b12x--", "router"),
    ("gpt-oss-120b", "MXFP4, 4 replicas", "full_gptoss", "promptopt"),
    ("gpt-oss-20b", "MXFP4, 4 replicas", "sw_gptoss_ficutlass_mxfp8", "router"),
    ("gemma-4-26B-A4B", "BF16, 4 replicas", "gemma26_---", "promptopt"),
    ("Muse-Glimmer-30B", "BF16, 4 replicas", "muse30_---", "router"),
    ("Qwen3.8-Flash-Next", "NVFP4, 2 x TP2", "qwen38fn_---", "router"),
    ("MiniMax-M3", "MXFP4, TP4", "minimaxm3_", "router"),
    ("DeepSeek-V4-Flash", "MXFP4+FP8 native, TP1 x DP4 + EP, 512 seqs per engine", "ds4flash_dp4ep4_s512", "router"),
    ("DeepSeek-V4-Flash", "MXFP4+FP8 native, TP4, b12x experts", "ds4flash_b12x-b12x", "router"),
    ("GLM-5.3-Flash", "NVFP4, TP4 + EP, 512 seqs", "glm53f_tp4ep4_s512", "router"),
]

def load_600w():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "summary_all.tsv"), encoding="utf-8"), delimiter="\t"))
    best = collections.defaultdict(dict)
    for r in rows:
        if r.get("host") != "pro6000-s600w":
            continue
        try:
            o, i = float(r["out_tps"]), float(r["in_tps"])
        except (KeyError, ValueError):
            continue
        k = (r["tag"], r["label"])
        if o > best[k].get("out", 0):
            best[k] = {"out": o, "in": i, "C": r.get("C")}
    return best

def main():
    print("### Cost per node-hour (four cards), GBP ex-VAT and USD\n")
    print("| basis | owner utilisation | GBP / node-hour | USD / node-hour |")
    print("|---|---:|---:|---:|")
    for u in (1.0, 0.7, 0.5):
        v = LIST_GBP_MONTH / HOURS / u
        print(f"| Scan list, nothing resold, no relief | {int(u*100)}% | {v:.2f} | {v*FX:.2f} |")
    for u in (1.0, 0.7):
        v = LIST_GBP_MONTH * (1 - COMMIT_DISC) / HOURS / u
        print(f"| Scan committed (−25%) | {int(u*100)}% | {v:.2f} | {v*FX:.2f} |")
    for u in (1.0, 0.7, 0.5, 0.2):
        v, cost = loaded_per_node_hour(u)
        print(f"| fully loaded: committed − ERIS − Vast resale of idle hours | {int(u*100)}% | {v:.2f} | {v*FX:.2f} |")
    print(f"| renting the same box on Vast today (on-demand, 4 × median $1.55/GPU-h) | — | {4*1.55/FX:.2f} | {4*1.55:.2f} |")

    best = load_600w()
    list_h = LIST_GBP_MONTH / HOURS * FX / 0.7          # USD per active node-hour at 70% utilisation, list
    loaded_h = loaded_per_node_hour(0.7)[0] * FX         # USD per node-hour, fully loaded, 70%
    print("\n### The same hour of output, bought from an API\n")
    print(f"Node cost per active hour: ${list_h:.2f} at Scan list and 70% utilisation, ${loaded_h:.2f} fully loaded. "
          "API prices are OpenRouter list on 5 September 2026 (Qwen3.8-Flash-Next from Artificial Analysis).\n")
    print("| model · configuration (600 W) | shape | tokens / node-hour (in + out, M) | API $/M in · out | API bill for that hour | node cost, list 70% | node cost, loaded | API ÷ node (list) | API ÷ node (loaded) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model, cfg, tagpre, label in ROWS:
        cands = [(k, v) for k, v in best.items() if k[0].startswith(tagpre) and k[1] == label]
        if not cands:
            print(f"| {model} · {cfg} | {label} | pending | {API[model][0]:.3f} · {API[model][1]:.2f} | — | — | — | — | — |")
            continue
        (tag, lab), v = max(cands, key=lambda kv: kv[1]["out"])
        inm, outm = v["in"] * 3600 / 1e6, v["out"] * 3600 / 1e6
        pi, po = API[model]
        bill = inm * pi + outm * po
        print(f"| {model} · {cfg} | {label} C{v['C']} | {inm:,.0f} + {outm:,.0f} | {pi:.3f} · {po:.2f} | ${bill:,.0f} | ${list_h:.2f} | ${loaded_h:.2f} | {bill/list_h:,.0f}× | {bill/loaded_h:,.0f}× |")

if __name__ == "__main__":
    main()
