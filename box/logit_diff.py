#!/usr/bin/env python3
"""Logit-level comparison of two serving configurations (e.g. fp8 KV vs bf16 KV).

This is the sensitive, cascade-free way to measure what a quantization choice does:
for each of N identical contexts we ask both servers for the next-token distribution
and compare them directly, instead of comparing whole generations by string identity.

Reports, over all positions:
  top1_agreement   fraction of positions where both rank the same token first
  top5_overlap     mean |top5(A) & top5(B)| / 5
  mean_kl          mean KL(P_A || P_B) over the shared top-k support (nats)
  p95_kl, max_kl   tail of the divergence
  mean_top1_prob_delta   mean |p_A(top1) - p_B(top1)|

Contexts are built by teacher-forcing: we take a seed prompt, generate a continuation
from server A once, then evaluate the next-token distribution at every prefix of that
continuation on BOTH servers. Identical context on both sides, many positions per prompt.

Usage: logit_diff.py <base_a> <base_b> <alias> <out.json> [positions_per_prompt]
"""
import json
import math
import sys
import urllib.request

A, B, ALIAS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
POS_PER_PROMPT = int(sys.argv[5]) if len(sys.argv) > 5 else 12
TOPK = 20

SEEDS = [
    "The three most important considerations when choosing GPUs for a research lab are",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "Q: A train travels 120 km in 90 minutes. What is its average speed in km/h? A: Let me work through this.",
    "Summarise the trade-offs between tensor parallelism and replica parallelism for LLM inference:",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\nWHERE",
    "The key difference between precision and recall is that",
    "In 2026, the main constraints on running large mixture-of-experts models locally are",
    "Explain why prefix caching helps prompt-optimisation workloads:",
]


def post(base, path, payload, timeout=300):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def continuation(base, prompt, n):
    d = post(base, "/v1/completions", {
        "model": ALIAS, "prompt": prompt, "max_tokens": n,
        "temperature": 0, "seed": 1234})
    return d["choices"][0]["text"]


def next_dist(base, prompt):
    """Next-token distribution at the end of `prompt`, as {token: logprob}."""
    d = post(base, "/v1/completions", {
        "model": ALIAS, "prompt": prompt, "max_tokens": 1,
        "temperature": 0, "logprobs": TOPK, "seed": 1234})
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    top = lp.get("top_logprobs") or []
    if not top:
        return None
    return top[0]


def kl(pa, pb):
    """KL(P_A || P_B) over A's support, with B floored so missing mass is penalised."""
    floor = math.log(1e-8)
    total = 0.0
    for tok, la in pa.items():
        p = math.exp(la)
        lb = pb.get(tok, floor)
        total += p * (la - lb)
    return max(total, 0.0)


rows = []
for seed in SEEDS:
    try:
        cont = continuation(A, seed, POS_PER_PROMPT + 2)
    except Exception as e:
        rows.append({"seed": seed[:50], "error": f"{type(e).__name__}: {str(e)[:90]}"})
        continue
    # evaluate at growing prefixes of the shared continuation
    words = cont.split(" ")
    for k in range(1, min(POS_PER_PROMPT, len(words)) + 1):
        ctx = seed + " ".join(words[:k])
        try:
            da, db = next_dist(A, ctx), next_dist(B, ctx)
        except Exception as e:
            rows.append({"seed": seed[:40], "pos": k, "error": str(e)[:90]})
            continue
        if not da or not db:
            continue
        ta = max(da, key=da.get)
        tb = max(db, key=db.get)
        top5a = set(sorted(da, key=da.get, reverse=True)[:5])
        top5b = set(sorted(db, key=db.get, reverse=True)[:5])
        rows.append({
            "seed": seed[:40], "pos": k,
            "top1_same": ta == tb,
            "top5_overlap": len(top5a & top5b) / 5.0,
            "kl": round(kl(da, db), 6),
            "top1_prob_delta": round(abs(math.exp(da[ta]) - math.exp(db.get(ta, -20.0))), 6),
        })

ok = [r for r in rows if "kl" in r]
if ok:
    kls = sorted(r["kl"] for r in ok)
    summary = {
        "positions": len(ok),
        "top1_agreement": round(sum(r["top1_same"] for r in ok) / len(ok), 4),
        "top5_overlap": round(sum(r["top5_overlap"] for r in ok) / len(ok), 4),
        "mean_kl": round(sum(kls) / len(kls), 6),
        "p95_kl": round(kls[int(0.95 * (len(kls) - 1))], 6),
        "max_kl": round(kls[-1], 6),
        "mean_top1_prob_delta": round(sum(r["top1_prob_delta"] for r in ok) / len(ok), 6),
        "errors": sum(1 for r in rows if "error" in r),
    }
else:
    summary = {"positions": 0, "errors": sum(1 for r in rows if "error" in r)}

json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print("LOGIT-DIFF " + json.dumps(summary))
