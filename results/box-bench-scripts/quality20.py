import json, sys, urllib.request
alias, base, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
 "Q: What is 17*23? A:", "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h? A:",
 "Write a Python function that returns the n-th Fibonacci number iteratively.\n\ndef fib(n):",
 "Write a SQL query that returns the top 3 customers by total order value from tables customers(id,name) and orders(id,customer_id,amount).\n\nSELECT",
 "Rate the following answer for correctness on a 1-5 scale and explain briefly.\nQuestion: What is the capital of Australia?\nAnswer: Sydney.\nRating:",
 "Translate to French: The meeting has been moved to Thursday afternoon.\nFrench:",
 "Summarize in one sentence: Large language models are trained on vast corpora and can perform many tasks without task-specific training, but they sometimes hallucinate facts.\nSummary:",
 "List three differences between TCP and UDP.\n1.",
 "Counterfactual: rewrite the sentence so that the outcome is the opposite. Original: Because the experiment used a larger batch size, training converged faster.\nRewritten:",
 "Extract the JSON fields name, age, city from: Maria Lopez, 34, lives in Lisbon.\n{\"name\":",
 "Explain why the sky is blue in two sentences.", "What is the derivative of x^3 * sin(x)?",
 "Give a regex that matches a UK postcode.", "Q: A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost? A:",
 "Complete the proverb and explain it: A stitch in time",
 "You are a judge. Compare answer A: Paris and answer B: Lyon for the question What is the capital of France?. Which is correct and why?",
 "Write a haiku about GPUs.", "Convert 98.6 Fahrenheit to Celsius, show the formula.",
 "Name the process by which plants make food and give its chemical equation.",
 "Return only valid JSON with keys a,b,c set to 1,2,3.",
]
res = []
for p in prompts:
    body = json.dumps({"model": alias, "prompt": p, "max_tokens": 128, "temperature": 0, "seed": 1234}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
        txt = d["choices"][0]["text"]; fin = d["choices"][0].get("finish_reason")
    except Exception as e:
        txt = "ERROR: " + str(e)[:200]; fin = "error"
    res.append({"prompt": p, "text": txt, "finish": fin})
json.dump(res, open(out, "w"), indent=1)
bad = [r for r in res if r["finish"] == "error" or "!!!!" in r["text"] or len(set(r["text"].split())) < 3]
print("quality20:", len(res), "answers,", len(bad), "suspicious ->", out)
