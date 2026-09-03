#!/usr/bin/env python3
"""Corruption tripwire (not a benchmark): 20 fixed prompts, greedy, against a live server.

usage: quality20.py ALIAS BASE_URL OUT.json [--mode chat|completion] [--max-tokens N]

Why v2: the original used raw /v1/completions on chat/reasoning models and flagged an answer
only if it contained "!!!!" or had fewer than three distinct words. It passed GLM-5.3-Flash
output that read "111 222 333 444 ..." as fine. Now:
  * chat mode uses the model's chat template (the way it is actually served);
  * every answer gets a repetition score (longest repeated 6-gram count, distinct-token
    ratio) and a finish check, so loops are flagged as DEGENERATE;
  * a handful of prompts carry an expected substring, so a confidently wrong answer to
    17*23 is flagged as WRONG rather than passed.
The summary line prints ok / degenerate / wrong / error counts. Anything but ok=20 is a
reason to stop benchmarking and look.
"""
import json, sys, re, urllib.request, collections

alias, base, out = sys.argv[1], sys.argv[2], sys.argv[3]
mode = "chat"
max_tokens = 256
a = sys.argv[4:]
if "--mode" in a:
    mode = a[a.index("--mode") + 1]
if "--max-tokens" in a:
    max_tokens = int(a[a.index("--max-tokens") + 1])

# (prompt, expected substring or None)
PROMPTS = [
    ("What is 17*23? Reply with just the number.", "391"),
    ("If a train travels 60 km in 45 minutes, what is its speed in km/h? Reply with just the number.", "80"),
    ("Write a Python function fib(n) that returns the n-th Fibonacci number iteratively.", "def fib"),
    ("Write a SQL query that returns the top 3 customers by total order value from tables customers(id,name) and orders(id,customer_id,amount).", "ORDER BY"),
    ("Rate the following answer for correctness on a 1-5 scale and explain briefly.\nQuestion: What is the capital of Australia?\nAnswer: Sydney.", "Canberra"),
    ("Translate to French: The meeting has been moved to Thursday afternoon.", "jeudi"),
    ("Summarize in one sentence: Large language models are trained on vast corpora and can perform many tasks without task-specific training, but they sometimes hallucinate facts.", None),
    ("List three differences between TCP and UDP.", "UDP"),
    ("Counterfactual: rewrite the sentence so that the outcome is the opposite. Original: Because the experiment used a larger batch size, training converged faster.", "slower"),
    ("Extract the JSON fields name, age, city from: Maria Lopez, 34, lives in Lisbon. Return only JSON.", "Lisbon"),
    ("Explain why the sky is blue in two sentences.", "scatter"),
    ("What is the derivative of x^3 * sin(x)?", "cos"),
    ("Give a regex that matches a UK postcode.", "["),
    ("A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost? Reply with just the number.", "0.05"),
    ("Complete the proverb and explain it: A stitch in time", "nine"),
    ("You are a judge. Compare answer A: Paris and answer B: Lyon for the question 'What is the capital of France?'. Which is correct and why?", "Paris"),
    ("Write a haiku about GPUs.", None),
    ("Convert 98.6 Fahrenheit to Celsius, show the formula.", "37"),
    ("Name the process by which plants make food and give its chemical equation.", "hotosynthesis"),
    ("Return only valid JSON with keys a,b,c set to 1,2,3.", "\"a\""),
]


def call(prompt):
    if mode == "chat":
        body = {"model": alias, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0, "seed": 1234}
        url = base + "/v1/chat/completions"
    else:
        body = {"model": alias, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "seed": 1234}
        url = base + "/v1/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    c = d["choices"][0]
    if mode == "chat":
        m = c["message"]
        text = m.get("content") or ""
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    else:
        text, reasoning = c["text"], ""
    return text, reasoning, c.get("finish_reason")


def repetition(text):
    toks = re.findall(r"\S+", text)
    if len(toks) < 12:
        return 0, 1.0
    grams = collections.Counter(tuple(toks[i:i + 6]) for i in range(len(toks) - 5))
    top = max(grams.values())
    distinct = len(set(toks)) / len(toks)
    return top, distinct


res, counts = [], collections.Counter()
# the prompts are independent, so run them concurrently: 20 x 1024 tokens single-stream
# on a reasoning model would otherwise take most of ten minutes
from concurrent.futures import ThreadPoolExecutor


def _safe(prompt):
    try:
        return call(prompt)
    except Exception as e:
        return e


with ThreadPoolExecutor(max_workers=10) as ex:
    outcomes = list(ex.map(lambda pe: _safe(pe[0]), PROMPTS))

for (prompt, expect), outcome in zip(PROMPTS, outcomes):
    if isinstance(outcome, Exception):
        res.append({"prompt": prompt, "text": "", "finish": "error", "verdict": "error", "err": str(outcome)[:200]})
        counts["error"] += 1
        continue
    text, reasoning, fin = outcome
    top, distinct = repetition(text if text.strip() else reasoning)
    verdict = "ok"
    # a loop: the same 6-gram at least 4 times, or a collapsed vocabulary, while hitting the cap
    if fin == "length" and (top >= 4 or distinct < 0.35):
        verdict = "degenerate"
    elif not text.strip() and fin == "length":
        verdict = "degenerate"      # burned the whole budget thinking, produced nothing
    elif expect and expect.lower() not in (text + " " + reasoning).lower():
        verdict = "wrong"
    counts[verdict] += 1
    res.append({"prompt": prompt, "text": text, "reasoning_chars": len(reasoning), "finish": fin,
                "top6gram": top, "distinct": round(distinct, 3), "verdict": verdict})

json.dump(res, open(out, "w"), indent=1)
print(f"quality20[{mode}]: ok={counts['ok']} degenerate={counts['degenerate']} "
      f"wrong={counts['wrong']} error={counts['error']} -> {out}")
