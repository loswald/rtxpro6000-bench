# evalsuite — quality eval suite for the RTX PRO 6000 serving benchmark

Small, concentrated, high-information item sets from public benchmarks, run against the node's
OpenAI-compatible vLLM replicas and scored programmatically. Six capability axes —
reasoning/maths, code, agentic tool-calling, long-context retrieval, knowledge, instruction following —
in ≤ 15 minutes per model at concurrency 64, ≤ ~400 items, everything reproducible from a seed.

Python 3.12, standard library + `aiohttp` (present in the node's vLLM environment). No gated datasets,
no API keys, no LLM judges. Verified end-to-end against a local mock server, so the pipeline can be
exercised without the node.

This directory currently contains the **core**: the runner, the data builder, the shared client and
scoring helpers, the family plugin interface, the mock server and a synthetic `selftest` family. The six
real families (`math`, `code`, `tools`, `longctx`, `knowledge`, `ifeval`) are plugin modules that drop into
`families/` and follow `families/_base.py`; the design spec (item selection, prompts, scorers, statistics)
is the reference for their implementation.

## Layout

```
evalsuite/
  common.py            shared core: seeded RNG, Wilson CI / McNemar, response normalisation
                       (<think>/reasoning stripping, harmony leaks, degenerate-loop detection), \boxed{} and
                       letter extraction, fetch() with .part files + sha256, ChatClient (aiohttp, round-robin
                       sticky routing, request semaphore, retries, token accounting, /tokenize, probing)
  run_eval.py          the runner: CLI, plan, time-budget scheduler, reports (JSON / items.jsonl / TSV)
  prepare_data.py      builds data/items/<family>.jsonl for every family and writes data/manifest.json
  mock_server.py       aiohttp OpenAI-compatible mock (oracle/noisy/canned/echo/garbage, think variants, faults)
  families/
    __init__.py        registry: discover(), load(name), default_families(), resolve()
    _base.py           THE FAMILY INTERFACE (documented template + defaults copied onto family modules)
    selftest.py        synthetic arithmetic + MCQ family (no downloads) for pipeline verification / node smoke
  data/                created by prepare_data.py: raw/ (downloads), items/ (built sets), manifest.json
  README.md
```

`run_eval.py` and `prepare_data.py` are invoked as `python3 evalsuite/<script>.py …` from the directory that
contains `evalsuite/` (`/workspace/bench` on the node, where `box/` is synced). Each entry script puts its own
directory on `sys.path`, so `import common` / `import families` work from anywhere.

## Quick start (no node needed)

```bash
cd box                                                            # or /workspace/bench on the node
python3 evalsuite/prepare_data.py --only selftest                 # builds data/items/selftest.jsonl + manifest.json
python3 evalsuite/mock_server.py --port 9000 --think inline &     # oracle answers wrapped in a decoy <think> block
python3 evalsuite/mock_server.py --port 9001 --think field &
python3 evalsuite/run_eval.py --tag mock_oracle --base-urls http://127.0.0.1:9000,http://127.0.0.1:9001 \
    --model m --families selftest --out results/eval_mock --concurrency 16 --retry-backoff 0.2,0.5,1
# -> selftest acc=1.000, exit 0.  Then try the failure modes:
python3 evalsuite/mock_server.py --port 9002 --mode canned &      # acc 0, everything 'unparsed'
python3 evalsuite/mock_server.py --port 9003 --mode noisy --accuracy 0.7 &   # Wilson CI should cover 0.7
python3 evalsuite/mock_server.py --port 9004 --fail-rate 0.3 &   # retries > 0, n_error == 0
python3 evalsuite/mock_server.py --port 9005 --think unclosed &  # trunc_rate 1.0, status 'truncated'
```

## Node run recipe

```bash
cd /workspace/bench
python3 evalsuite/prepare_data.py                                 # once; caches sources under evalsuite/data/raw
python3 evalsuite/run_eval.py --dry-run --tag x --base-urls http://127.0.0.1:8000 --model m
python3 evalsuite/run_eval.py --tag qwen38_27b_fp8 --model m --out /workspace/results/eval --reasoning \
    --base-urls http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003
python3 evalsuite/run_eval.py --tag qwen38_27b_nvfp4_b12x … --reasoning     # same items, same seed -> pairable
python3 evalsuite/run_eval.py --profile smoke --tag smoke_$(date +%s) --base-urls http://127.0.0.1:8000 --model m \
    --families selftest                                          # one-minute liveness check of a new stack
```

Never call `pkill`/`pgrep -f` from anything on the node; the suite only ever manages its own processes.

## `run_eval.py`

```
python3 evalsuite/run_eval.py --tag TAG --base-urls URL[,URL...] --model NAME [--out results/eval]
    [--families math,code,tools,longctx,knowledge,ifeval] [--limit N] [--concurrency 64]
    [--max-tokens 2048] [--max-tokens-family tools=1024,longctx=512] [--reasoning]
    [--temperature T] [--top-p P] [--seed 20260903] [--chat-template-kwargs JSON] [--extra-body JSON]
    [--time-budget 900] [--grace 120] [--request-timeout 600] [--retries 3] [--retry-backoff 2,8,20]
    [--family-opt tools.mode=native --family-opt knowledge.shots=0 --family-opt longctx.fixed=true]
    [--profile default|full|smoke] [--data-dir evalsuite/data] [--gpus 4] [--notes TEXT]
    [--save-responses tail|full|none] [--dry-run] [--resume] [--list-families] [--no-manifest-check]
```

* **Sampling.** Greedy by default (T=0, top_p=1, seed 20260903). `--reasoning` switches to T=0.6/top_p=0.95,
  `--max-tokens 4096` and the families' reasoning caps (greedy decoding loops on this stack's reasoning
  models). Explicit `--temperature/--top-p/--max-tokens` always win. Loops are flagged `degenerate`
  (same 6-gram ≥ 4× or distinct-token ratio < 0.35 in the last 2000 chars while hitting the cap — the
  `box/quality20.py` rule) and counted wrong.
* **Response normalisation** (`common.normalize_response`). Reads `message.reasoning` (current vLLM) and
  `reasoning_content` (older builds); strips closed `<think>…</think>` and sibling tags, treats an unclosed
  `<think>` as swallowing the rest (`unclosed_think`), handles harmony leaks (`<|channel|>analysis…`,
  `harmony_leak`). Empty visible text + `finish_reason=length` → `truncated`; empty visible text with
  reasoning present → the reasoning tail is scored and `answer_from_reasoning` is counted (a misconfigured
  reasoning parser shows up as a non-zero count). Scorers only ever see the visible text.
* **Concurrency and routing.** `asyncio.Semaphore(--concurrency)` over requests (what "concurrency 64"
  means to the server); item *i* is pinned to `urls[i % n]` for prefix-cache locality, a retry moves to the
  next URL. Retries on connection errors, timeouts, HTTP 408/409/429/5xx and malformed 200 bodies with
  back-off 2/8/20 s (+U(0,1)); other 4xx are terminal (`error_kind: context_length` when the body mentions
  the context window). `finish_reason=length` is never retried.
* **Dispatch order.** Within a family, sub-families are interleaved round-robin over their seeded `order`;
  families are interleaved round-robin in priority order (tools, code, math, longctx, knowledge, ifeval).
  `--limit N` keeps the first N per family in that order, so smaller runs are nested subsets.
* **Time budget.** `deadline = start + --time-budget`. An item is launched only if `now + p90(item wall time
  of its family) ≤ deadline` (priors before three observations: tools 90 s, code 120 s, math 120 s,
  longctx 45 s, knowledge 40 s, ifeval 20 s, or the family's `ITEM_TIME_FALLBACK_S`, capped at a quarter of
  the budget so a prior can never keep a family from being sampled at all). A blocked family stays
  blocked, so its attempted set is a prefix of the order. In-flight items get `--grace` seconds after the
  deadline, then are cancelled (`status: cancelled`); never-launched items are `skipped`. `truncated: true`
  whenever a planned item was not attempted.
* **Profiles.** `default`; `full` (larger item sets, built with `prepare_data.py --profile full`); `smoke`
  (`--limit 3 --time-budget 300`).
* **Exit codes.** 0 complete and valid · 1 truncated, or invalid (`n_error/n_attempted > 0.05`) · 2 items
  missing or manifest hash mismatch · 3 no live endpoint.

### Outputs

* `<out>/<tag>.json` — per family and per sub-family: `n_planned, n_attempted, n_scored, n_error,
  n_cancelled, n_skipped, n_correct, acc (= n_correct/n_scored), ci95 (Wilson), acc_strict
  (= n_correct/n_attempted), status_counts, mean/p50 output tokens, mean prompt tokens, trunc_rate,
  degenerate_rate, answer_from_reasoning, p50/p90 item seconds, wall_s, gpu_minutes, info_per_gpu_min,
  failed_ids` (every scored-and-wrong id), `failures` (details of every non-correct item), `extra` (from the
  family's `aggregate()`, e.g. `acc_official` for BFCL multi-turn). Plus `aggregate` (micro/macro accuracy,
  totals, requests, retries, per-URL tallies), `config`, `endpoint`, `data_manifest_sha256`, `notes`,
  `truncated`, `valid`, `invalid_reasons`.
* `<out>/<tag>.items.jsonl` — one line per attempted item, appended as items finish (`--resume` skips ids
  already present): `id, family, sub, status, correct, score, expected, extracted, finish_reason,
  prompt_tokens, completion_tokens, requests, retries, latency_s, item_s, base_url, error, flags, detail,
  content, reasoning` (`--save-responses tail` keeps the last 4000/1000 chars).
* `<out>/eval_summary.tsv` — append-only, long format: one row per family, per sub-family and one
  `aggregate` row per run; columns `tag model date family sub n_planned n_attempted n_scored n_correct acc
  ci_lo ci_hi acc_strict acc_official mean_out_tok mean_prompt_tok trunc_rate wall_s gpu_min truncated valid
  profile reasoning max_tokens temperature tools_mode lcb_window manifest_sha256 notes`.
* `<out>/<tag>.run.json` (arguments, host, endpoint probe) and `<out>/<tag>.log`.

Statuses: `correct wrong unparsed truncated empty error cancelled skipped`. `acc` uses scored items only
(attempted − error − cancelled); `acc_strict` charges errors against the model.

## `prepare_data.py`

```
python3 evalsuite/prepare_data.py [--data-dir evalsuite/data] [--seed 20260903] [--profile default|full]
    [--only math,code,…] [--refresh] [--allow-short] [--pin] [--opt [FAMILY.]KEY=VALUE …]
    [--offline-fixtures DIR] [--list]
```

Calls every family's `prepare()`; the family fetches its sources with `common.fetch()` into
`data/raw/<source>/` (cached, `.part` files, sha256 recorded, `HF_TOKEN`/`HF_ENDPOINT` honoured) and writes
`data/items/<family>.jsonl` with sorted keys in seeded order, so rebuilds are byte-identical.
`data/manifest.json` records per family the file, `n`, sha256 and ids, the sub-family counts, the resolved
sources and `manifest_sha256`; `run_eval.py` verifies the item files against it before a run and stores the
digest in its output. A family raises `common.ShortPool` when a pool is smaller than requested (exit 1
unless `--allow-short`).

## The family interface (`families/_base.py`)

A family is one module `families/<name>.py`. Required: `NAME`, `prepare(...)`, `score(item, text[, meta])`.
Optional (defaults in `_base.py`): `SUBFAMILIES`, `PRIORITY`, `HIDDEN`, `DEFAULT_MAX_TOKENS`
(`{"default": 1024, "reasoning": 2048}` style caps), `ITEM_TIME_FALLBACK_S`, `NOTES`,
`load_items(limit, seed, data_dir=None)` (returns the items **in run order**; the default reads
`data/items/<NAME>.jsonl`, interleaves sub-families by `order`, applies the limit), `build_messages(item,
ctx)` (the PROMPT builder; default `item["messages"]`), `mock_response(item)` (what the mock should answer),
`aggregate(records)` (extra family statistics), and two coroutines for agentic families:
`run_item(item, ctx) -> common.ItemOutcome` (replaces the single-request path — BFCL multi-turn drives its
own step loop through `ctx.client.chat(..., route_key=item["_index"])`) and `prepare_run(items, ctx)`
(once before dispatch — long-context `/tokenize` calibration). `ctx` is a `common.RunContext` (config,
client, `max_tokens`, sampling, `opts` from `--family-opt`, `deadline`).

Scorers receive only the visible text and return `common.Verdict(correct, status, score, extracted,
expected, detail, flags)`. Helpers: `common.extract_final_answer` (`\boxed{}` with brace matching →
"answer is" phrase → optional last integer), `common.extract_letter` (bounded letter cascade),
`common.seeded_rng(seed, family, subfamily)` for all randomness.

## `mock_server.py`

```
python3 evalsuite/mock_server.py --port 9000 [--model m] [--items-dir evalsuite/data/items] [--lookup answers.json]
    --mode oracle|noisy|canned|echo|garbage [--accuracy 0.7] [--think none|inline|field|unclosed]
    [--reasoning-field reasoning|reasoning_content] [--fail-rate P --fail-mode first|always] [--latency-ms N]
    [--jitter-ms N] [--slow-every N --slow-ms 20000] [--truncate-rate P] [--seed 7]
```

Faults: `--fail-rate p` returns HTTP 503 with probability *p*, by default only on the first sighting of a
prompt (a runner with retries must therefore finish with `n_error == 0`), or on every request with
`--fail-mode always`; `--slow-every N` makes every N-th request sleep `--slow-ms` (request timeouts);
`--truncate-rate p` cuts answers with `finish_reason=length`; `--latency-ms` shapes response times for
time-budget tests.

Answers come from a lookup keyed on a marker in the prompt: every built item is indexed by the sha1 of its
normalised last user message and by a marker (`item["mock_marker"]` or the first 240 normalised characters of
its last user message) searched as a substring of the rendered prompt, so prompt wrapping by a family's
`build_messages()` does not break the match. The response is `item["mock"]`, else the family's
`mock_response(item)`, else `"… \boxed{<answer>}."`; `--lookup` adds arbitrary marker → response pairs
(strings or `{"content", "reasoning", "finish_reason", "tool_calls"}`); anything unmatched is echoed.
Tool-result turns are answered with `Done.`. Endpoints: `/health`, `/v1/models`, `/version`, `/tokenize`,
`/stats` (`requests`, `max_in_flight`, `errors_injected`, match counts), `/v1/chat/completions`.

## Statistics

95 % Wilson intervals on `n_scored` (`common.wilson`; n = 0 → [0, 1]). At n ≤ 40 a single family's CI is
±14–17 pp, so single families only rank models for large gaps; the 400-item aggregate (±4.4 pp at p ≈ 0.7)
and paired per-item comparison (exact McNemar, `common.mcnemar_exact`) are the instruments for serving
decisions (NVFP4 vs FP8, kernel choice, KV dtype). Pairing needs identical items, prompts, `max_tokens`,
seed and sampling settings — the manifest digest and `config` block in each result make that checkable.

## Known limits of the core

* Only `selftest` ships here; the six real families are separate plugin modules (see the design spec).
* `compare.py` (paired McNemar/Newcombe over two runs, TSV pivot) is not part of the core.
* The sandbox for code scoring belongs to the `code` family, not the core.
* The mock's token counts are estimates (`len(text)//4 + 6·messages`; completion = words × 1.3).
