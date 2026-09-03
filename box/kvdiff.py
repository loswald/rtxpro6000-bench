#!/usr/bin/env python3
"""Greedy output diff between two servers (fp8 KV vs bf16 KV), identical prompts, temperature 0.
Answers: does the KV cache dtype we enabled for throughput cost us any quality?
Usage: kvdiff.py <base_a> <base_b> <alias> <out.json>
"""
import difflib
import json
import re
import sys
import urllib.request

A, B, ALIAS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

PROMPTS = [
    "Q: A shop sells pens for 1.20 each and notebooks for 3.50. If I buy 4 pens and 3 notebooks, what is the total? Show your working. A:",
    "Q: What is 17*23? Answer with the number only. A:",
    "Q: A train leaves at 14:35 and arrives at 17:10. How long is the journey in minutes? A:",
    "Write a Python function that returns the n-th Fibonacci number iteratively.\n\ndef fib(n):",
    "Write a SQL query returning the top 3 customers by total order value from customers(id,name) and orders(id,customer_id,amount).\n\nSELECT",
    "Explain in two sentences why the sky appears blue.",
    "Rate this answer 1-5 for correctness and justify briefly.\nQuestion: What is the capital of Australia?\nAnswer: Sydney.\nRating:",
    "Translate to French: The quarterly report has been delayed until Thursday afternoon.\nFrench:",
    "Summarise in one sentence: Large language models are trained on vast corpora and can perform many tasks without task-specific training, but they sometimes state false facts confidently.\nSummary:",
    "List three concrete differences between TCP and UDP.\n1.",
    "Rewrite so the outcome is reversed. Original: Because the batch size was larger, training converged faster.\nRewritten:",
    "Extract JSON with keys name, age, city from: Maria Lopez, 34, lives in Lisbon.\n{",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Give a regular expression matching a UK postcode, and explain each part.",
    "Q: A bat and a ball cost 1.10 together. The bat costs 1.00 more than the ball. How much is the ball? Think step by step. A:",
    "You are a judge. Compare answer A (Paris) and answer B (Lyon) for the question: what is the capital of France? Which is correct and why?",
    "Convert 98.6 Fahrenheit to Celsius. Give the formula and the result.",
    "Name the process by which plants make food, and give its balanced chemical equation.",
    "Write a haiku about tensor cores.",
    "Return only valid JSON with keys a, b, c set to 1, 2, 3 respectively.",
    "Q: If 8 machines make 8 widgets in 8 minutes, how long do 40 machines take to make 40 widgets? A:",
    "Explain the difference between precision and recall to a new analyst, with one worked example.",
    "Refactor for readability, keep behaviour identical:\ndef f(x):\n  if x>0:\n    return 1\n  else:\n    if x<0:\n      return -1\n    else:\n      return 0",
    "Name the two hardest parts of running compute for a small AI research lab. Answer in two sentences.",
    "Q: Sort these numbers ascending and give the median: 14, 3, 27, 8, 19. A:",
]


def gen(base, prompt):
    body = json.dumps({
        "model": ALIAS, "prompt": prompt, "max_tokens": 256,
        "temperature": 0, "seed": 1234,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d["choices"][0]["text"]


def corrupt(t):
    flags = []
    if re.search(r"!{6,}", t):
        flags.append("bang-run")
    if re.search(r"(\b\w+\b)(\s+\1){9,}", t):
        flags.append("token-repeat")
    if "�" in t:
        flags.append("replacement-char")
    if not t.strip():
        flags.append("empty")
    return flags


rows, exact, ned_sum = [], 0, 0.0
for p in PROMPTS:
    try:
        a, b = gen(A, p), gen(B, p)
    except Exception as e:
        rows.append({"prompt": p[:60], "error": f"{type(e).__name__}: {str(e)[:100]}"})
        continue
    same = a == b
    exact += same
    ned = 1 - difflib.SequenceMatcher(None, a, b).ratio()
    ned_sum += ned
    rows.append({
        "prompt": p[:60], "exact": same, "ned": round(ned, 4),
        "flags_fp8": corrupt(a), "flags_bf16": corrupt(b),
        "fp8": a[:400], "bf16": b[:400],
    })

n = len([r for r in rows if "ned" in r])
summary = {
    "n": n,
    "exact_match": exact,
    "exact_rate": round(exact / max(n, 1), 3),
    "mean_norm_edit_distance": round(ned_sum / max(n, 1), 4),
    "corrupt_fp8": sum(1 for r in rows if r.get("flags_fp8")),
    "corrupt_bf16": sum(1 for r in rows if r.get("flags_bf16")),
    "errors": sum(1 for r in rows if "error" in r),
}
json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print("KV-DIFF " + json.dumps(summary))
