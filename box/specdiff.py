#!/usr/bin/env python3
"""Is speculative decoding lossless? Compare completions greedily - AGAINST A CONTROL.

Speculation proposes tokens with a cheap head and verifies each against the full model, so in theory greedy
output is unchanged by it. GLM-5.3-Flash scored 0.800 without its MTP head and 0.740 with it, which that
theory says is impossible, so this tool was written to compare the two greedily.

It found 11 of 12 completions diverging and I nearly published that as an engine defect. Then the control
ran - the SAME server captured twice - and matched on only 4 of 12. This stack is not deterministic at
greedy sampling at all: prefix caching serves an identical prompt from cached KV rather than recomputing
it, MoE routing and FlashInfer reductions use atomics with no fixed order, and continuous batching changes
which batch a request lands in. Any one of those flips an argmax on a near-tie, and the texts diverge from
there. So sequence comparison cannot attribute anything until the control reproduces.

The lesson generalises past this tool: a difference is only evidence when the control says the measurement
could have shown no difference.

Usage:  specdiff.py capture <base_url> <model> <out.json>          greedy, fixed prompts
        specdiff.py compare <a.json> <b.json>                      exit 0 if identical
        specdiff.py judge <base.json> <base2.json> <spec.json>     control-gated verdict
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
    return 0 if same == len(a) else 1


def verdict(control_same, control_n, treat_same, treat_n):
    """Read the treatment ONLY against its control.

    The first version of this tool printed "the speculator is wrong" the moment base and speculating
    outputs differed. Then the control was run - the same server compared with itself, greedy - and it
    matched on only 4 of 12. The base model does not reproduce itself, so divergence proves nothing about
    speculation, and that verdict would have gone into a public repository as an engine defect.

    Why a greedy server is non-deterministic here: prefix caching means an identical prompt is served from
    cached KV on the second pass rather than recomputed, which is a different numerical path; MoE routing
    and FlashInfer reductions use atomics whose order is not fixed; and continuous batching changes the
    batch a request lands in. Any of those flips an argmax on a near-tie, after which the texts diverge
    completely. Disable prefix caching and serialise the requests before this test means anything.
    """
    if control_same < control_n:
        print("  CONTROL FAILED: the base model matched itself on only %d/%d - this stack is not "
              "deterministic at greedy sampling, so this test cannot attribute anything to speculation."
              % (control_same, control_n))
        print("  Use the logit-level pass instead: it compares next-token DISTRIBUTIONS against a "
              "control pair, which does not require bit-exact reproduction.")
        return 2
    if treat_same == treat_n:
        print("  Speculation is exact: control reproduces, and speculation matches it. Any eval gap is "
              "sampling noise or truncation.")
        return 0
    print("  Speculation CHANGES the output while the control reproduces exactly (%d/%d) - the speculator "
          "is at fault, not the model." % (control_same, control_n))
    return 1


def _count_identical(pa, pb):
    a = json.load(open(pa, encoding="utf-8"))
    b = json.load(open(pb, encoding="utf-8"))
    same = sum(1 for x, y in zip(a, b)
               if ((x["reasoning"] or "") + "\x00" + (x["content"] or ""))
               == ((y["reasoning"] or "") + "\x00" + (y["content"] or "")))
    return same, len(a)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "capture":
        capture(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    elif mode == "judge":
        # judge <base.json> <base_repeat.json> <spec.json>
        cs, cn = _count_identical(sys.argv[2], sys.argv[3])
        print("  control  (base vs base):  %d/%d identical" % (cs, cn))
        ts, tn = _count_identical(sys.argv[2], sys.argv[4])
        print("  treatment(base vs spec):  %d/%d identical" % (ts, tn))
        sys.exit(verdict(cs, cn, ts, tn))
    else:
        print("usage: specdiff.py capture|compare|judge ...")
        sys.exit(2)
