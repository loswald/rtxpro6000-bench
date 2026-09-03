#!/usr/bin/env python3
"""
gates/kv_diff.py -- FP8-KV vs bf16-KV output diff gate (sm_120 corruption check).

Sends a FIXED set of 50 prompts (20 judge-style rubric prompts, 20 code prompts,
10 long-ish multi-step reasoning prompts) to an OpenAI-compatible server with
temperature=0, top_p=1, a fixed seed (1234) and max_tokens=512, and compares two
captures of the SAME cell launched with different --kv-cache-dtype (fp8 vs auto/bf16).

Endpoint (default `completions` = POST /v1/completions with the raw prompt): no chat
template is rendered, so Qwen3.x / DeepSeek-V4 never enter thinking mode and the
comparison is a pure decode-path diff.  `--endpoint chat` (POST /v1/chat/completions)
sends chat_template_kwargs {"enable_thinking": false, "thinking": false} -- the
Qwen3.x and DeepSeek-V4 template switches respectively (recipes.vllm.ai, 2026-09-02);
Jinja ignores the one the template does not know -- unless --allow-thinking.
Request fields verified against vLLM main protocol.py (2026-09-02): CompletionRequest
{model, prompt, temperature, top_p, seed, max_tokens, stream}; ChatCompletionRequest
{messages, chat_template_kwargs, max_completion_tokens, seed, ...}; the chat response
carries `reasoning` (older servers: `reasoning_content`).

Reports
  * exact-match rate (whitespace-stripped)
  * mean normalised edit distance (token-level Levenshtein / max token count)
  * mean character similarity (difflib ratio, 1.0 == identical)
  * corruption flags per output: runs of '!' (>= 6), a token repeated >= 10x, an
    n-gram (n=2..6) repeated >= 6x -- the known sm_120 signature of a broken FP8-KV /
    attention kernel path -- plus replacement chars (>= 3), control chars (>= 3) and
    empty outputs.  Any flagged output on either side fails the gate.
  * acs_suspected / pessimistic_tp (TP>1 cells only) from results/hw/decisions.env
    via gates/hwdecisions.py, plus launch.json excerpts of both sides when given.

Modes (Python 3 stdlib only)
  capture       --url http://127.0.0.1:8000 --label fp8 --out results/<cell>/kv_capture_fp8.json
  compare       --a <capture_a.json> --b <capture_b.json> --cell <cell> [--tp N --dp N --replicas N]
                [--launch-a <launch.json> --launch-b <launch.json>] --out results/<cell>/kv_diff.json
  live          --a-url ... --b-url ... --cell <cell> --out ...   (both servers up at once)
  dump-prompts  --out prompts.jsonl

Servers of one cell normally cannot coexist on the same GPUs, hence
capture -> relaunch -> capture -> compare (gates/run_kv_diff.sh drives bench/launch.sh).
Batch composition changes floating-point reduction order, so ALWAYS capture both
sides with the same --concurrency; two captures of the same server give the
noise floor (run_kv_diff.sh SELF_CHECK=1 -> kv_diff_selfcheck.json).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hwdecisions import hw_decisions, pessimistic_flags  # noqa: E402

MAX_TOKENS_DEFAULT = 512
SEED_DEFAULT = 1234
CONCURRENCY_DEFAULT = 4
# Qwen3.x templates read `enable_thinking`, DeepSeek-V4 templates read `thinking`; unknown kwargs are ignored.
NO_THINKING_KWARGS = {"enable_thinking": False, "thinking": False}
LAUNCH_KEYS = ("status", "seconds_to_ready", "engine", "engine_version", "model", "kv_cache_dtype", "tp", "dp", "replicas",
               "gpu_ids", "max_num_batched_tokens", "max_num_seqs", "custom_allreduce", "p2p_disabled", "run_tag",
               "kv_cache_line", "max_concurrency_line", "error_excerpt")

# --------------------------------------------------------------------------- #
# Fixed prompt set                                                            #
# --------------------------------------------------------------------------- #

JUDGE_TEMPLATE = """You are an impartial grader. Score the candidate answer against each rubric item with 0 (fails), 1 (partially meets) or 2 (fully meets).

Question:
{question}

Rubric:
{rubric}

Candidate answer:
\"\"\"{candidate}\"\"\"

Return ONLY a JSON object of the form
{{"scores": [<int per rubric item>], "total": <int>, "verdict": "PASS" or "FAIL", "justification": "<one or two sentences>"}}
where verdict is PASS if total >= 4, otherwise FAIL."""

# (question, [rubric items], candidate answer) -- a deliberate mix of right, wrong and partial candidates
JUDGE_ITEMS = [
    ("Explain why the sky appears blue during the day.",
     ["Mentions Rayleigh scattering or wavelength-dependent scattering by air molecules",
      "States that shorter (blue) wavelengths are scattered more strongly than longer ones",
      "Does not attribute the colour to reflection from the ocean"],
     "The sky is blue because it reflects the colour of the oceans. Water is blue, so the light bouncing off the sea colours the atmosphere above it."),
    ("What is the worst-case time complexity of binary search on a sorted array of n elements?",
     ["States O(log n)", "Explains that each step halves the remaining interval", "Notes the precondition that the array is sorted"],
     "Binary search runs in O(n) in the worst case because in the worst case you may have to examine every element before finding the key."),
    ("In one sentence, describe what a SQL LEFT JOIN returns.",
     ["All rows from the left table are kept", "Columns from the right table are NULL when there is no match", "The answer is exactly one sentence"],
     "A LEFT JOIN returns every row from the left table together with the matching rows from the right table, filling the right-hand columns with NULL where no match exists."),
    ("Convert 72 degrees Fahrenheit to Celsius, showing the formula.",
     ["Uses C = (F - 32) x 5/9", "Arrives at approximately 22.2 C", "Shows at least one intermediate step"],
     "C = (72 - 32) * 5/9 = 40 * 5/9 = 200/9 = 22.2 C"),
    ("Summarise the purpose of a mutex in two sentences.",
     ["States that a mutex provides mutual exclusion (one holder at a time)", "Mentions protecting shared state / a critical section", "Exactly two sentences"],
     "A mutex is a lock that lets many threads enter a critical section at the same time so throughput improves. It is mainly used to make I/O faster."),
    ("What does HTTP status 429 mean and which response header commonly accompanies it?",
     ["429 means Too Many Requests / rate limited", "Mentions the Retry-After header", "Does not confuse it with a 5xx server error"],
     "429 means the server is temporarily down for maintenance. Clients should read the Location header to find out where to retry."),
    ("Give the derivative of f(x) = 3x^4 - 2x + 7.",
     ["f'(x) = 12x^3 - 2", "The constant term disappears", "No extraneous terms remain"],
     "f'(x) = 12x^3 - 2 + 7"),
    ("Name the four DNA bases and state how they pair.",
     ["Lists adenine, thymine, cytosine, guanine", "Pairs A with T", "Pairs C with G"],
     "The bases are adenine, thymine, cytosine and guanine. Adenine pairs with guanine and cytosine pairs with thymine."),
    ("Explain what a LoRA adapter is in machine learning.",
     ["Frozen base weights with trainable low-rank matrices", "Mentions the low-rank factorisation (e.g. W + BA or rank r)", "Mentions a practical benefit (fewer trainable params, swappable/mergeable adapters)"],
     "LoRA freezes the pretrained weights and injects a pair of small trainable low-rank matrices A and B into selected linear layers, so the effective update is W + BA with rank r. Only A and B are trained, which cuts trainable parameters and optimizer memory dramatically, and the adapters can be merged into the base weights or swapped at inference time."),
    ("Describe the difference between TCP and UDP in at most three bullet points.",
     ["TCP is connection-oriented and reliable/ordered", "UDP is connectionless with no delivery guarantee", "At most three bullets"],
     "- TCP establishes a connection and guarantees ordered, reliable delivery with retransmission and flow control.\n- UDP is connectionless and sends datagrams with no delivery or ordering guarantee.\n- TCP adds latency and header overhead; UDP is lighter and preferred for real-time media and DNS."),
    ("A recipe needs 3 eggs for 12 muffins. How many eggs are needed for 30 muffins?",
     ["Sets up the proportion (3/12 per muffin or 1 egg per 4 muffins)", "Computes 7.5 eggs", "Rounds up to 8 whole eggs or explains the fractional result"],
     "3 eggs per 12 muffins is 0.25 eggs per muffin, so 30 muffins need 7.5 eggs; in practice you would use 8 eggs."),
    ("What is idempotency in the context of REST APIs?",
     ["Repeating the same request has the same effect as making it once", "Gives an example method (GET, PUT, DELETE) or contrasts with POST", "Does not describe caching"],
     "An idempotent request is one that the server caches so that it returns faster the second time it is made."),
    ("Translate 'Where is the train station?' into French.",
     ["Uses 'gare' for train station", "Correct interrogative structure (Où est ... ?)", "No spelling errors"],
     "Où est la gare ?"),
    ("Explain what the Python GIL is and one consequence of it.",
     ["Global Interpreter Lock: only one thread executes Python bytecode at a time", "Consequence: CPU-bound threads do not run in parallel", "Mentions a workaround such as multiprocessing or C extensions releasing the GIL"],
     "The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. As a result, CPU-bound multithreaded code does not speed up on multiple cores; people use multiprocessing or native extensions that release the GIL instead."),
    ("What is the capital of Australia?",
     ["Answers Canberra", "Does not answer Sydney or Melbourne", "Concise"],
     "The capital of Australia is Sydney."),
    ("State the Pythagorean theorem and give one numerical example.",
     ["a^2 + b^2 = c^2 for a right triangle", "Identifies c as the hypotenuse", "Gives a correct example such as 3-4-5"],
     "In a right-angled triangle the square of the hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2. Example: 3^2 + 4^2 = 9 + 16 = 25 = 5^2."),
    ("Explain speculative decoding in LLM inference in two sentences.",
     ["A cheap draft model (or n-gram/EAGLE head) proposes several tokens", "The target model verifies them in one forward pass, accepting a prefix", "Output distribution is unchanged / lossless"],
     "A small draft model proposes a block of candidate tokens, and the large target model scores all of them in a single forward pass, accepting the longest prefix that matches what it would have sampled. Because rejected tokens are resampled from the target distribution, the output distribution is unchanged while several tokens are produced per target step."),
    ("Why should passwords not be stored in plaintext, and what is the recommended alternative?",
     ["Plaintext exposes every password if the database leaks", "Recommends salted hashing", "Names a slow password hash (bcrypt, scrypt, Argon2, PBKDF2)"],
     "Plaintext passwords are dangerous because a database leak exposes them all. The fix is to encrypt them with AES so they can be decrypted when the user logs in."),
    ("Compute 17 * 23.",
     ["Answer is 391", "Shows work or states the answer directly", "No incorrect numbers"],
     "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391"),
    ("Explain the difference between latency and throughput for an inference server.",
     ["Latency is per-request time (e.g. TTFT, end-to-end)", "Throughput is aggregate work per unit time (tokens/s or requests/s)", "Notes the trade-off: batching raises throughput but can raise latency"],
     "Latency is how long one request waits, for example time to first token or total end-to-end time, whereas throughput is how much aggregate work the server completes per second, such as total output tokens per second across all requests. Larger batches usually increase throughput while increasing per-request latency."),
]

CODE_PROMPTS = [
    "Write a Python function `is_balanced(s: str) -> bool` that returns True when the brackets (), [], {} in `s` are balanced and properly nested, ignoring all other characters. Include three `assert` test cases after the function.",
    "Implement an `LRUCache` class in Python with `get(key)` and `put(key, value)` in O(1) time, with the capacity given to the constructor. Do not use functools.lru_cache or any third-party library. Show a short usage example.",
    "Write a bash script that takes a directory as `$1`, finds the 10 largest regular files beneath it (recursively) and prints their human-readable size and path, largest first. It must handle file names containing spaces.",
    "The following Python function is meant to return the arithmetic mean of a non-empty list. Find and fix the bug, then explain the fix in one sentence.\n\n```python\ndef mean(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs) - 1\n```",
    "Table `orders(id, customer_id, amount, created_at)`. Write a SQL query (PostgreSQL) that returns each customer's total `amount` over the last 30 days, only for customers whose total exceeds 1000, sorted by total descending.",
    "Write a Python generator function `fib_below(n)` that yields the Fibonacci numbers strictly less than `n`, starting 0, 1, 1, 2, ... Then show how to collect the values below 100 into a list.",
    "Implement binary search in C with the exact signature `int bsearch_int(const int *a, int n, int key)` that returns the index of `key` in the sorted array `a` or -1 if absent. Avoid integer overflow when computing the midpoint.",
    "Write a JavaScript function `debounce(fn, ms)` that returns a debounced version of `fn` (only the last call within `ms` milliseconds runs). Show a usage example attaching it to a window resize event.",
    "Write a Python function that takes a list of strings, parses the ones that are valid ISO-8601 datetimes (e.g. `2026-09-02T14:05:00Z`), skips invalid ones, and returns the parsed `datetime` objects sorted ascending. Use only the standard library.",
    "Write a Python regular expression that matches an IPv4 address strictly (each octet 0-255, no leading zeros such as 01). Compile it with `re.fullmatch` semantics and show four test cases: two that match and two that must not.",
    "Write a Rust function `fn reverse_words(s: &str) -> String` that reverses the order of whitespace-separated words, collapsing repeated whitespace into a single space. Include a `#[test]`.",
    "Write a Go HTTP handler for `GET /health` that responds with JSON `{\"status\":\"ok\",\"time\":\"<RFC3339 timestamp>\"}` and status 200, and returns 405 for other methods. Show the `main` that registers it on port 8080.",
    "Write a Dockerfile for a minimal Python 3.12 FastAPI application in `app/main.py` served by uvicorn on port 8080, running as a non-root user, with dependencies installed from `requirements.txt` in a cached layer.",
    "Implement Dijkstra's shortest-path algorithm in Python over an adjacency dict of the form `{node: [(neighbour, weight), ...]}`. Return a dict of shortest distances from the source. Use `heapq`.",
    "Write a Python script that reads a JSONL file whose path is given in `sys.argv[1]`, counts the distinct values of the field `category`, and prints a two-column table (category, count) sorted by count descending. Lines that fail to parse should be counted and reported on stderr.",
    "Write a Python `@dataclass` named `Request` with fields `id: str`, `prompt: str`, `max_tokens: int = 128`, `temperature: float = 0.0`, and a method `to_openai(self, model: str) -> dict` that returns the JSON payload for POST /v1/completions.",
    "Write pytest unit tests for a function `slugify(title: str) -> str` (lower-case, words joined by single hyphens, punctuation removed, accents stripped). Cover: multiple spaces, punctuation, a unicode title with accents, and the empty string.",
    "Write a Python function `merge_intervals(intervals)` that takes a list of `[start, end]` pairs and returns the merged, sorted list of non-overlapping intervals. Explain the time complexity in one line.",
    "Write a CUDA kernel in C++ that adds two float arrays element-wise into a third, plus the host code that allocates device memory, copies inputs, launches the kernel with a correct grid/block computation for `n` elements, copies the result back and frees memory.",
    "This Python code is run with 8 threads and the final `counter` is wrong. Explain the race condition and give a corrected version.\n\n```python\ncounter = 0\n\ndef worker():\n    global counter\n    for _ in range(100000):\n        counter += 1\n```",
]

REASONING_PROMPTS = [
    "You are scheduling six benchmark jobs on four identical GPUs. Each job needs exactly one GPU for its whole duration and a GPU runs one job at a time. Durations in minutes: A=40, B=25, C=60, D=15, E=35, F=50. Constraints: D must finish before B starts; C and F cannot run at the same time because they share a dataset lock; E may not start before minute 20. Find a schedule with the minimum makespan. Give the start time and GPU for every job, check each constraint explicitly, state the makespan, and explain in two sentences why no shorter schedule is possible.",
    "An inference server sustains 12,000 output tokens/s of decode throughput at saturation and 90,000 tokens/s of prefill throughput. A batch of 2,048 requests arrives at once; each has 4,096 input tokens and generates exactly 512 output tokens. Assume prefill and decode do not overlap and the server stays saturated. (a) How long does all prefill take? (b) How long does all decode take? (c) What is the wall-clock time and the average requests/s? (d) If an FP8 KV cache lets the server hold 1.6x more concurrent sequences and decode throughput consequently scales by 1.3x, recompute (c). Show every step with units.",
    "On an island, knights always tell the truth and knaves always lie. You meet three inhabitants: Ada, Ben and Cy. Ada says: 'Ben is a knave.' Ben says: 'Ada and Cy are the same type.' Cy says: 'I am a knight.' Determine the type of each person. Work through the cases systematically, show which assignment is consistent with all three statements, and explain why every other assignment leads to a contradiction.",
    "A lab rents a 4-GPU machine at $3.20 per hour for a 48-hour benchmark. Model downloads total 950 GB and the provider charges $0.02 per GB of ingress beyond the first 500 GB. Storage is billed at $0.10 per GB-month for a 1.5 TB volume, prorated by the hour over a 30-day (720-hour) month. Results egress is 12 GB at $0.09 per GB. The lab wastes 7 of the 48 hours re-running failed cells. (a) Compute each cost component. (b) Give the total. (c) What fraction of the total was wasted, counting only the wasted rental hours? Round money to cents and show the arithmetic.",
    "Three researchers (Priya, Marco, Lena) each own a different GPU (RTX PRO 6000, H100, MI300X) and live in a different city (Cambridge, Zurich, Toronto). Clues: (1) The person in Zurich does not own the H100. (2) Marco lives in Toronto. (3) Priya owns the RTX PRO 6000. (4) The MI300X owner does not live in Cambridge. Determine who owns which GPU and lives where. Show the elimination steps in a small table, then restate the final assignment in three sentences.",
    "You have 9 coins that look identical; exactly one is heavier than the rest. Using a two-pan balance with no reference weights, what is the minimum number of weighings that guarantees you find the heavy coin? Describe the strategy weighing by weighing, explain with an information-theoretic (three-outcome) argument why fewer weighings cannot suffice, and generalise: what is the maximum number of coins you can handle in k weighings?",
    "Two fair six-sided dice are rolled. (a) What is the probability that the sum is a prime number? Enumerate the 36 outcomes grouped by sum. (b) What is the probability that the sum is prime given that at least one die shows a 3? (c) Are the events 'sum is prime' and 'at least one die shows a 3' independent? Justify with the computed numbers.",
    "Order these tasks so that every dependency is satisfied: provision (no deps); hardware_truth (needs provision); download_models (needs provision); collect_env (needs hardware_truth); launch_cell (needs download_models and collect_env); sweep (needs launch_cell); gsm8k_gate (needs launch_cell); kv_diff (needs launch_cell); cotenancy (needs sweep and gsm8k_gate); summarise (needs sweep, kv_diff and cotenancy). Give one valid ordering, list which tasks can run in parallel at each stage, and compute the critical-path length if every task takes 1 unit except download_models (6 units) and sweep (4 units).",
    "A tank of capacity 600 litres is empty at t=0. An inlet fills it at 12 L/min. A drain removes 5 L/min but is closed for the first 10 minutes. At t=25 min a leak starts removing a further 2 L/min. At t=40 min the inlet rate is increased to 20 L/min. Compute the volume at t=10, t=25 and t=40, and determine the exact time at which the tank becomes full. Show the piecewise computation and check that the volume never goes negative.",
    "Estimate how many tokens per day a team of nine engineers sends to an LLM coding agent if each engineer works 6 focused hours, issues one agent turn every 4 minutes on average, and each turn carries 32,000 input tokens of context and 1,500 output tokens. Then estimate the GPU-hours per day needed if a single 4-GPU node sustains 15,000 total tokens/s for this workload. State every assumption, show the unit conversions, and finish with a one-sentence sanity check on the result.",
]


def build_prompts() -> list[dict]:
    prompts = []
    for i, (q, rubric, cand) in enumerate(JUDGE_ITEMS):
        rub = "\n".join(f"{j + 1}. {r}" for j, r in enumerate(rubric))
        prompts.append({"id": f"judge_{i:02d}", "category": "judge",
                        "prompt": JUDGE_TEMPLATE.format(question=q, rubric=rub, candidate=cand)})
    for i, p in enumerate(CODE_PROMPTS):
        prompts.append({"id": f"code_{i:02d}", "category": "code", "prompt": p})
    for i, p in enumerate(REASONING_PROMPTS):
        prompts.append({"id": f"reason_{i:02d}", "category": "reasoning", "prompt": p})
    assert len(prompts) == 50, len(prompts)
    return prompts


def load_prompts(path: str | None) -> list[dict]:
    if not path:
        return build_prompts()
    prompts = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise SystemExit(f"{path}:{n + 1}: each line needs a 'prompt' field")
            obj.setdefault("id", f"custom_{n:03d}")
            obj.setdefault("category", "custom")
            prompts.append(obj)
    return prompts


def prompt_set_hash(prompts: list[dict]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p["id"].encode())
        h.update(b"\0")
        h.update(p["prompt"].encode())
        h.update(b"\0")
        h.update((p.get("system") or "").encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# HTTP                                                                        #
# --------------------------------------------------------------------------- #

def http_json(url: str, payload: dict | None = None, timeout: float = 600, method: str | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_meta(base: str) -> dict:
    meta = {"base_url": base}
    try:
        meta["models"] = [m["id"] for m in http_json(base + "/v1/models", timeout=30)["data"]]
    except Exception as e:  # noqa: BLE001
        meta["models_error"] = repr(e)
    try:
        meta["version"] = http_json(base + "/version", timeout=10)
    except Exception:  # noqa: BLE001
        pass
    return meta


def one_request(base: str, model: str, item: dict, args) -> dict:
    t0 = time.time()
    if args.endpoint == "chat":
        messages = []
        if item.get("system"):
            messages.append({"role": "system", "content": item["system"]})
        messages.append({"role": "user", "content": item["prompt"]})
        # max_tokens is deprecated on the chat endpoint in favour of max_completion_tokens (vLLM main).
        payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0,
                   "seed": args.seed, "max_completion_tokens": args.max_tokens, "stream": False}
        if not args.allow_thinking:
            payload["chat_template_kwargs"] = dict(NO_THINKING_KWARGS)
        url = base + "/v1/chat/completions"
    else:
        # Raw prompt, no chat template -> no thinking block can be triggered; pure decode-path diff.
        payload = {"model": model, "prompt": item["prompt"], "temperature": 0.0, "top_p": 1.0,
                   "seed": args.seed, "max_tokens": args.max_tokens, "stream": False}
        url = base + "/v1/completions"

    last_err = None
    for attempt in range(args.retries + 1):
        try:
            resp = http_json(url, payload, timeout=args.timeout)
            ch = resp["choices"][0]
            if args.endpoint == "chat":
                msg = ch.get("message") or {}
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            else:
                text = ch.get("text") or ""
                reasoning = None
            usage = resp.get("usage") or {}
            return {"id": item["id"], "category": item["category"], "text": text, "reasoning": reasoning,
                    "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "latency_s": round(time.time() - t0, 3), "error": None}
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            last_err = f"HTTP {e.code}: {body}"
            if e.code == 400 and "chat_template_kwargs" in payload:
                payload.pop("chat_template_kwargs", None)  # template rejected the kwarg; retry without it
                continue
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(2.0 * (attempt + 1))
    return {"id": item["id"], "category": item["category"], "text": "", "reasoning": None,
            "finish_reason": None, "completion_tokens": None, "prompt_tokens": None,
            "latency_s": round(time.time() - t0, 3), "error": last_err}


def capture(base: str, label: str, args, prompts: list[dict]) -> dict:
    base = base.rstrip("/")
    meta = server_meta(base)
    model = args.model or (meta.get("models") or [None])[0]
    if not model:
        raise SystemExit(f"could not determine model id from {base}/v1/models; pass --model")
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()
    outputs: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(one_request, base, model, p, args): p["id"] for p in prompts}
        done = 0
        for fut in cf.as_completed(futs):
            r = fut.result()
            outputs[r["id"]] = r
            done += 1
            if done % 10 == 0 or done == len(prompts):
                print(f"[kv_diff capture {label}] {done}/{len(prompts)}", file=sys.stderr)
    errors = [o for o in outputs.values() if o["error"]]
    return {
        "kind": "kv_capture",
        "label": label,
        "cell": args.cell,
        "model": model,
        "server": meta,
        "settings": {"endpoint": args.endpoint, "url_path": "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions",
                     "temperature": 0.0, "top_p": 1.0, "seed": args.seed,
                     "max_tokens": args.max_tokens, "concurrency": args.concurrency,
                     "thinking_disabled": not args.allow_thinking,
                     "chat_template_kwargs": (dict(NO_THINKING_KWARGS) if (args.endpoint == "chat" and not args.allow_thinking) else None),
                     "note": "completions endpoint = raw prompt, no chat template, so no thinking mode" if args.endpoint != "chat" else None},
        "prompt_set_hash": prompt_set_hash(prompts),
        "n_prompts": len(prompts),
        "n_errors": len(errors),
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(time.time() - t0, 2),
        "outputs": {pid: outputs[pid] for pid in sorted(outputs)},
    }


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

BANG_RUN = re.compile(r"!{6,}")
REPLACEMENT_CHAR = "�"


def _is_punct(tok: str) -> bool:
    return not any(ch.isalnum() for ch in tok)


def corruption_flags(text: str, min_token_repeat: int = 10, min_ngram_repeat: int = 6) -> list[str]:
    """Known sm_120 corruption signatures: long runs of '!' and degenerate repetition."""
    flags: list[str] = []
    if not text.strip():
        return ["empty_output"]
    if BANG_RUN.search(text):
        flags.append("bang_run")
    if text.count(REPLACEMENT_CHAR) >= 3:
        flags.append("replacement_chars")
    ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if ctrl >= 3:
        flags.append("control_chars")

    toks = text.split()
    # consecutive identical (non-punctuation) tokens
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= min_token_repeat and not _is_punct(a):
            flags.append("repeated_token")
            break
    # consecutive repeated n-grams (n = 2..6), skipping punctuation-only grams (markdown tables etc.)
    if "repeated_token" not in flags:
        found = None
        for n in range(2, 7):
            if len(toks) < n * min_ngram_repeat:
                break
            for i in range(0, len(toks) - n * min_ngram_repeat + 1):
                gram = toks[i:i + n]
                if all(_is_punct(t) for t in gram):
                    continue
                if all(toks[i + k * n: i + (k + 1) * n] == gram for k in range(1, min_ngram_repeat)):
                    found = f"repeated_{n}gram"
                    break
            if found:
                flags.append(found)
                break
    return flags


def levenshtein(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_edit_distance_tokens(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    m = max(len(ta), len(tb))
    return 0.0 if m == 0 else levenshtein(ta, tb) / m


def first_divergence(a: str, b: str):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def launch_excerpt(path):
    """Compact view of a bench/launch.sh launch.json (engine version, kv dtype, TP, KV capacity, errors)."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": repr(e)}
    out = {k: d.get(k) for k in LAUNCH_KEYS if k in d}
    out["path"] = path
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def _median(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    m = len(xs) // 2
    return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2, 4)


def compare(cap_a: dict, cap_b: dict, cell: str, max_ned: float, noise_floor: dict | None = None,
            tp=None, dp=None, replicas=None, launch_a: str | None = None, launch_b: str | None = None) -> dict:
    oa, ob = cap_a["outputs"], cap_b["outputs"]
    ids = [i for i in oa if i in ob]
    if cap_a.get("prompt_set_hash") != cap_b.get("prompt_set_hash"):
        print("[kv_diff] WARNING: prompt-set hashes differ between captures", file=sys.stderr)

    pairs = []
    for pid in ids:
        A, B = oa[pid], ob[pid]
        ta, tb = (A.get("text") or "").strip(), (B.get("text") or "").strip()
        ned = norm_edit_distance_tokens(ta, tb)
        ratio = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
        fa, fb = corruption_flags(ta), corruption_flags(tb)
        div = first_divergence(ta, tb)
        pairs.append({
            "id": pid, "category": A.get("category"),
            "exact_match": ta == tb,
            "norm_edit_distance": round(ned, 4),
            "char_similarity": round(ratio, 4),
            "first_divergence_char": div,
            "len_a_chars": len(ta), "len_b_chars": len(tb),
            "tokens_a": A.get("completion_tokens"), "tokens_b": B.get("completion_tokens"),
            "finish_a": A.get("finish_reason"), "finish_b": B.get("finish_reason"),
            "flags_a": fa, "flags_b": fb,
            "error_a": A.get("error"), "error_b": B.get("error"),
            "snippet_a": ta[:160], "snippet_b": tb[:160],
        })

    def side_stats(cap: dict, key: str) -> dict:
        outs = [cap["outputs"][i] for i in ids]
        flagged = {p["id"]: p[key] for p in pairs if p[key]}
        finish = {}
        for o in outs:
            finish[str(o.get("finish_reason"))] = finish.get(str(o.get("finish_reason")), 0) + 1
        return {
            "label": cap.get("label"), "model": cap.get("model"),
            "engine": (cap.get("server") or {}).get("version"),
            "base_url": (cap.get("server") or {}).get("base_url"),
            "settings": cap.get("settings"),
            "n_errors": sum(1 for o in outs if o.get("error")),
            "corrupt_count": len(flagged),
            "corrupt_ids": flagged,
            "finish_reasons": finish,
            "mean_completion_tokens": _mean([o.get("completion_tokens") for o in outs]),
            "mean_latency_s": _mean([o.get("latency_s") for o in outs]),
            "capture_duration_s": cap.get("duration_s"),
            "started_utc": cap.get("started_utc"),
        }

    cats = sorted({p["category"] for p in pairs})
    by_cat = {}
    for c in cats:
        ps = [p for p in pairs if p["category"] == c]
        by_cat[c] = {"n": len(ps),
                     "exact_match_rate": round(sum(p["exact_match"] for p in ps) / len(ps), 4),
                     "mean_norm_edit_distance": _mean([p["norm_edit_distance"] for p in ps]),
                     "mean_char_similarity": _mean([p["char_similarity"] for p in ps]),
                     "corrupt_a": sum(1 for p in ps if p["flags_a"]),
                     "corrupt_b": sum(1 for p in ps if p["flags_b"])}

    n = len(pairs)
    exact_rate = round(sum(p["exact_match"] for p in pairs) / n, 4) if n else None
    mean_ned = _mean([p["norm_edit_distance"] for p in pairs])
    a_stats, b_stats = side_stats(cap_a, "flags_a"), side_stats(cap_b, "flags_b")
    errors = a_stats["n_errors"] + b_stats["n_errors"]

    reasons = []
    if a_stats["corrupt_count"]:
        reasons.append(f"{a_stats['corrupt_count']} corrupted outputs on side A ({a_stats['label']})")
    if b_stats["corrupt_count"]:
        reasons.append(f"{b_stats['corrupt_count']} corrupted outputs on side B ({b_stats['label']})")
    if mean_ned is not None and mean_ned > max_ned:
        reasons.append(f"mean normalised edit distance {mean_ned} > {max_ned}")
    if errors:
        reasons.append(f"{errors} request errors")
    if n == 0:
        reasons.append("no comparable prompt ids")
    passed = not reasons

    nf = None
    if noise_floor:
        nf = {"exact_match_rate": noise_floor.get("exact_match_rate"),
              "mean_norm_edit_distance": noise_floor.get("mean_norm_edit_distance"),
              "source": noise_floor.get("source")}
        if isinstance(nf["mean_norm_edit_distance"], (int, float)) and isinstance(mean_ned, (int, float)):
            nf["excess_over_noise_floor"] = round(mean_ned - nf["mean_norm_edit_distance"], 4)

    worst = sorted(pairs, key=lambda p: (-p["norm_edit_distance"], p["id"]))[:5]
    la, lb = launch_excerpt(launch_a), launch_excerpt(launch_b)
    if tp in (None, ""):   # fall back to the launcher's record of the cell layout
        for lx in (la, lb):
            if lx and lx.get("tp") not in (None, ""):
                tp, dp, replicas = lx.get("tp"), lx.get("dp"), lx.get("replicas")
                break
    dec = hw_decisions()
    flags = pessimistic_flags(dec, tp, dp, replicas)

    return {
        "gate": "kv_diff",
        "cell": cell,
        "pass": passed,
        "status": "compared",
        "fail_reasons": reasons,
        "kv_a": cap_a.get("label"),
        "kv_b": cap_b.get("label"),
        "n_prompts": n,
        "exact_match_rate": exact_rate,
        "mean_norm_edit_distance": mean_ned,
        "median_norm_edit_distance": _median([p["norm_edit_distance"] for p in pairs]),
        "mean_char_similarity": _mean([p["char_similarity"] for p in pairs]),
        "mean_first_divergence_char": _mean([p["first_divergence_char"] for p in pairs if p["first_divergence_char"] is not None]),
        "thresholds": {"max_mean_norm_edit_distance": max_ned, "corrupt_outputs_allowed": 0},
        "noise_floor": nf,
        "side_a": a_stats,
        "side_b": b_stats,
        "by_category": by_cat,
        "worst_pairs": worst,
        "pairs": pairs,
        "prompt_set_hash": cap_a.get("prompt_set_hash"),
        "launch_a": la,
        "launch_b": lb,
        **flags,
        "hw_decisions": {k: v for k, v in dec.items() if not k.startswith("_")},
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": [
            "exact match between FP8-KV and bf16-KV is NOT expected to be 100%; the gate fails on corruption signatures, request errors, or mean normalised edit distance above the threshold",
            "compare against the noise floor (two captures of the same server) before reading small distances as a precision effect",
            "batch composition changes reduction order; both captures must use the same --concurrency",
            "pessimistic_tp marks TP>1 cells on this ACS-suspected host (throughput lower bound); it does not affect correctness verdicts",
        ],
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[kv_diff] wrote {path}", file=sys.stderr)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="served model id / alias (default: first from /v1/models)")
    p.add_argument("--endpoint", choices=["chat", "completions"], default="completions",
                   help="completions = /v1/completions with the raw prompt (default, no thinking possible); chat = /v1/chat/completions")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY_DEFAULT)
    p.add_argument("--timeout", type=float, default=900.0, help="per-request timeout (s)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--allow-thinking", action="store_true",
                   help="chat only: do not send chat_template_kwargs {enable_thinking: false, thinking: false}")
    p.add_argument("--prompts", default=None, help="optional JSONL of {id,category,prompt[,system]} to replace the built-in 50")
    p.add_argument("--cell", default=os.environ.get("CELL", "unknown"))


def _add_layout_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tp", default=None, help="tensor-parallel size of the cell (pessimistic_tp is set for TP>1 only)")
    p.add_argument("--dp", default=None)
    p.add_argument("--replicas", default=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("capture", help="query one server, save outputs")
    pc.add_argument("--url", required=True)
    pc.add_argument("--label", required=True, help="e.g. fp8, fp8_ds_mla, bf16")
    pc.add_argument("--out", required=True)
    _add_gen_args(pc)

    pm = sub.add_parser("compare", help="compare two captures")
    pm.add_argument("--a", required=True, help="capture JSON of side A (typically FP8 KV)")
    pm.add_argument("--b", required=True, help="capture JSON of side B (typically bf16 KV)")
    pm.add_argument("--cell", default=os.environ.get("CELL", "unknown"))
    pm.add_argument("--out", required=True)
    pm.add_argument("--max-ned", type=float, default=float(os.environ.get("MAX_NED", "0.30")))
    pm.add_argument("--noise-floor", default=None, help="kv_diff JSON from a same-server self-check")
    pm.add_argument("--no-pairs", action="store_true", help="omit per-pair detail from the output")
    pm.add_argument("--launch-a", default=None, help="launch.json written by bench/launch.sh for side A")
    pm.add_argument("--launch-b", default=None, help="launch.json written by bench/launch.sh for side B")
    _add_layout_args(pm)

    pl = sub.add_parser("live", help="capture both servers (both up) and compare")
    pl.add_argument("--a-url", required=True)
    pl.add_argument("--b-url", required=True)
    pl.add_argument("--a-label", default="fp8")
    pl.add_argument("--b-label", default="bf16")
    pl.add_argument("--out", required=True)
    pl.add_argument("--max-ned", type=float, default=float(os.environ.get("MAX_NED", "0.30")))
    _add_gen_args(pl)
    _add_layout_args(pl)

    pd = sub.add_parser("dump-prompts", help="write the built-in prompt set as JSONL")
    pd.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.mode == "dump-prompts":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for p in build_prompts():
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[kv_diff] wrote 50 prompts to {args.out}", file=sys.stderr)
        return 0

    if args.mode == "capture":
        cap = capture(args.url, args.label, args, load_prompts(args.prompts))
        _write(args.out, cap)
        print(json.dumps({"label": cap["label"], "model": cap["model"], "n_errors": cap["n_errors"],
                          "duration_s": cap["duration_s"]}))
        return 0 if cap["n_errors"] == 0 else 2

    if args.mode == "compare":
        nf = None
        if args.noise_floor:
            nfj = _load(args.noise_floor)
            nf = {"exact_match_rate": nfj.get("exact_match_rate"),
                  "mean_norm_edit_distance": nfj.get("mean_norm_edit_distance"), "source": args.noise_floor}
        rep = compare(_load(args.a), _load(args.b), args.cell, args.max_ned, nf,
                      tp=args.tp, dp=args.dp, replicas=args.replicas, launch_a=args.launch_a, launch_b=args.launch_b)
        if args.no_pairs:
            rep.pop("pairs", None)
        _write(args.out, rep)
        print(json.dumps({k: rep[k] for k in ("cell", "pass", "fail_reasons", "exact_match_rate",
                                                "mean_norm_edit_distance", "mean_char_similarity", "pessimistic_tp")}, indent=2))
        return 0 if rep["pass"] else 1

    if args.mode == "live":
        prompts = load_prompts(args.prompts)
        cap_a = capture(args.a_url, args.a_label, args, prompts)
        cap_b = capture(args.b_url, args.b_label, args, prompts)
        base = os.path.splitext(args.out)[0]
        _write(f"{base}_capture_{args.a_label}.json", cap_a)
        _write(f"{base}_capture_{args.b_label}.json", cap_b)
        rep = compare(cap_a, cap_b, args.cell, args.max_ned, tp=args.tp, dp=args.dp, replicas=args.replicas)
        _write(args.out, rep)
        print(json.dumps({k: rep[k] for k in ("cell", "pass", "fail_reasons", "exact_match_rate",
                                                "mean_norm_edit_distance", "pessimistic_tp")}, indent=2))
        return 0 if rep["pass"] else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
