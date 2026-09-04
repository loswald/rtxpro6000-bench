#!/usr/bin/env python3
"""Is speculative decoding actually lossless? Compare completions token-for-token, greedily.

Speculative decoding proposes tokens with a cheap head and verifies them against the full model, so with
greedy sampling the output must be BIT-IDENTICAL to the same model without speculation. Anything else is a
bug in the speculator, not a quality trade-off, and no amount of task accuracy will tell you which you are
looking at - accuracy differences at temperature are confounded by sampling noise and by truncation.

This is what prompted it: GLM-5.3-Flash scored 0.800 without its MTP head and 0.740 with it, a gap that
would be impossible if speculation were exact. The whole difference tracked truncation - 28 of 80 maths
items hit the token ceiling with speculation against 10 without - so the model was rambling further, which
a distribution-preserving speculator cannot cause.

Usage:  specdiff.py capture <base_url> <model> <out.json>     # once per arm, greedy, fixed prompts
        specdiff.py compare <a.json> <b.json>                 # exit 0 if identical, 1 if not
"""
import json
import sys
import urllib.request

# Deliberately mixed: arithmetic that runs long (where the gap showed), code, prose, and a long-chain
# problem. Greedy, so any divergence is the speculator.
PROMPTS = [
    "Compute 17 * 23 and reply with only the number.",
    "What is 2^17? Reply with only the number.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h? Show your reasoning.",
    "Sum every integer from 1 to 200 inclusive. Show your working, then give the total.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Explain in two sentences why prefix caching helps shared-prefix workloads.",
    "List the first 12 prime numbers, comma separated.",
    "Solve for x: 3x + 7 = 25. Show each step.",
    "What is the derivative of x^3 * sin(x)? Show the product rule applied.",
    "Write one SQL query returning the top 3 customers by total order value.",
    "Factor 2x^2 + 7x + 3 completely, showing the steps.",
    "In one paragraph, describe what tensor parallelism costs that replica parallelism does not.",
]


def post(base, payload, timeout=1200):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def capture(base, model, out):
    rows = []
    for i, p in enumerate(PROMPTS):
        d = post(base, {"model": model, "messages": [{"role": "user", "content": p}],
                        # greedy: no temperature, no top_p, no seed dependence
                        "temperature": 0.0, "top_p": 1.0, "max_tokens": 4096})
        ch = d["choices"][0]
        msg = ch.get("message", {})
        rows.append({
            "i": i, "prompt": p,
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"),
            "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
        })
        print("  [%2d] %-6s %5s tok  %s" % (i, ch.get("finish_reason"),
                                            (d.get("usage") or {}).get("completion_tokens"),
                                            (msg.get("content") or "")[:60].replace("\n", " ")))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print("  wrote %s" % out)


def compare(pa, pb):
    a = json.load(open(pa, encoding="utf-8"))
    b = json.load(open(pb, encoding="utf-8"))
    same = 0
    for x, y in zip(a, b):
        # compare the whole generation, thinking included: the speculator must not change that either
        ta = (x["reasoning"] or "") + "\x00" + (x["content"] or "")
        tb = (y["reasoning"] or "") + "\x00" + (y["content"] or "")
        if ta == tb:
            same += 1
            continue
        # find where they diverge, in characters
        n = min(len(ta), len(tb))
        k = next((j for j in range(n) if ta[j] != tb[j]), n)
        print("  [%2d] DIVERGES at char %d of %d/%d  (%s tok vs %s tok)"
              % (x["i"], k, len(ta), len(tb), x["completion_tokens"], y["completion_tokens"]))
        print("       a: ...%r" % ta[max(0, k - 40):k + 60])
        print("       b: ...%r" % tb[max(0, k - 40):k + 60])
    print("  %d/%d completions identical" % (same, len(a)))
    if same == len(a):
        print("  VERDICT: speculation is exact at greedy sampling - any eval gap is sampling noise or truncation")
        return 0
    print("  VERDICT: speculation CHANGES the output at greedy sampling - the speculator is wrong, not the model")
    return 1


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
