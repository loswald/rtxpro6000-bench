# Blackwell inference benchmarks: 4× RTX PRO 6000 and 8× RTX 5090

Measured serving throughput and quality for current open-weight models on **sm_120** — the architecture
shared by the RTX PRO 6000 Blackwell (96 GB) and the RTX 5090 (32 GB). Run by [Sqwish Labs](https://sqwish.ai)
in September 2026 to decide a two-year hardware commitment, and published because almost nothing about this
architecture is written down: the datacentre kernels (sm_100) do not run here, and the defaults are wrong in
ways that cost between 30% and 300%.

Everything here is reproducible from the scripts in `box/`. Every number is steady state with 6–8× as many
prompts as concurrent slots, a unique seed per run, and controls launched minutes apart in the same session.

**Workload shapes** (chosen for agent harnesses, not for leaderboards): `router` 1,024 in / 128 out ·
`promptopt` 3,072-token shared prefix + 512 in / 256 out · `judge` 4,096 in / 512 out · `short` 256 / 64 ·
`rollout` 8,192 / 2,048. Intelligence figures in brackets are the Artificial Analysis Intelligence Index
v4.1.1.

---

## The ten rules

1. **NVFP4 beats FP8 by 64%, but only if you force the kernel.** vLLM's auto-selection picks a *W4A16
   dequant* kernel for a W4A4 checkpoint and lands 47% **below** FP8. `--kernel-config.linear_backend b12x`
   or `flashinfer_b12x` are the only true W4A4 paths on this silicon. Always read the `Using … Kernel` line
   the engine prints, and force the backend.
2. **Use ModelOpt NVFP4 checkpoints, not compressed-tensors ones.** RedHat and unsloth "NVFP4" builds are
   mixed precision: attention is FP8 with *dynamic* activation scales, which the b12x kernel refuses, so they
   fall back to 2,048 out tok/s where the ModelOpt build does 5,161.
3. **Never hard-code a mixture-of-experts backend — sweep it.** `flashinfer_cutlass`, correct for gpt-oss, is
   rejected outright by any four-bit expert checkpoint with group-16 scales. Eleven arms died on this in one
   night. When the engine rejects a backend it prints the set it accepts; `triton` is not in it for NVFP4.
4. **Four replicas beat one tensor-parallel server, by several times**, whenever the model fits one card.
   There is no NVLink here; replicas exchange nothing.
5. **`--no-enable-flashinfer-autotune` on every tensor-parallel launch.** If rank 0 has a cached autotune
   result and the others do not, the ranks desynchronise in a collective and the server hangs forever.
6. **Per-architecture flags that are not optional.** Qwen3.8-Flash-Next rejects an fp8 KV cache outright
   (`Qwen4Exp QSA requires a BF16 main KV cache`). MiniMax-M3 and Inkling need `--enforce-eager` on sm_120 —
   a `KeyError: '/psm_…'` is a worker *segfault* seen from the parent, not a config error. MiniMax also needs
   `--block-size 128`; GLM-5.3-Flash needs `--block-size 1024` for the DeepGEMM indexer.
7. **Check `power.limit` against `power.max_limit` before trusting any host.** The same cards at 400 W
   instead of 600 W lose 23% on NVFP4 and 34% on FP8.
8. **On consumer cards, the container's CUDA must match the host driver.** The cu130 image ships a
   compatibility shim that only works on datacentre GPUs; on an RTX 5090 host with a 575 driver every launch
   dies with CUDA error 804.
9. **Serve each model the way its vendor says to, or you are benchmarking your own harness.** A missing
   `--reasoning-parser` makes the scorer grade the model's chain-of-thought; the wrong sampling costs more
   than any kernel here; and `enable_thinking` defaults differ per model, so a naive table compares some
   models thinking and others not. Worth **+0.17 aggregate** on Qwen3.8-27B — larger than every kernel
   finding in this document combined. See `box/lists/profiles.tsv`.
10. **A model that does not fit one card is a different product.** Replicas exchange nothing; tensor
    parallelism over PCIe without NVLink costs 8× throughput and turns a 0.7 s time-to-first-token into
    324 s. Decide the memory ceiling before the card count.

---

## Headline throughput

### 4× RTX PRO 6000 Blackwell, Server Edition, 600 W

Qwen3.8-27B, four replicas, one per card — the kernel comparison the whole campaign turns on. Same
checkpoint, same session, only the backend changes:

| kernel the engine used | router C256 | router C1024 | promptopt C1024 | judge C512 | tripwire |
|---|---:|---:|---:|---:|---|
| **B12xNvFp4 W4A4** (`b12x`) | 4,288 | **5,161** | 4,804 | 4,400 | 19/20 |
| **FlashInferB12xNvFp4 W4A4** (`flashinfer_b12x`) | 4,304 | **5,182** | 4,842 | 4,448 | 20/20 |
| B12xFp8BlockScaledMM (FP8 checkpoint) | 2,642 | 3,148 | 2,992 | 2,799 | 19/20 |
| FlashInferCuteDslNvFp4 **W4A16** (auto) | 1,608 | 1,671 | 1,637 | 1,621 | 19/20 |

Output tokens/s per node. The NVFP4 checkpoint is 17.9 GB against 29 GB for FP8, and *faster* — but the
auto-selected kernel is slower than FP8 and slower than it looks, because it barely moves from C256 to
C1024 while FP8 climbs 19%.

Other models on the same box:

| model (AA index) | configuration | shape | out tok/s | in tok/s | TTFT p50 |
|---|---|---|---:|---:|---:|
| gpt-oss-120b (24) | 4 replicas, FlashInfer CUTLASS + mxfp8 | promptopt C2048 | **19,689** | 275,653 | 1.5 s |
| | | router C2048 | 9,948 | 79,581 | 1.0 s |
| | | short C2048 | 16,017 | 64,066 | 1.3 s |
| | | judge C1024 | 7,468 | 59,745 | 1.4 s |
| gpt-oss-20b (24) | 4 replicas, mxfp8 activations | router C2048 | 13,752 | 110,017 | 0.7 s |
| | | judge C1024 | 9,152 | 73,217 | 1.0 s |
| | 4 replicas, Marlin (the default) | router C2048 | 9,956 | 79,649 | 1.0 s |
| DeepSeek-V4-Flash (52) | TP1 × DP4 + expert parallel | promptopt C512 | 3,683 | 51,562 | 6.6 s |
| | Marlin + EP | promptopt C512 | 3,002 | 42,032 | 3.1 s |
| **GLM-5.3-Flash (57)** | TP4, ported vendor vLLM | promptopt C256 | 1,002 | 14,030 | 2.7 s |
| | | router C256 | 920 | 7,362 | 2.2 s |
| | | judge C128 | 771 | 6,165 | 2.1 s |

GLM-5.3-Flash is the highest-intelligence open model that fits in 384 GB at all — three points below the
60-point ceiling only 750B–2.8T models reach. Getting it to produce *correct* output on this card took a
day; see "GLM-5.3-Flash" below.

### 4× RTX PRO 6000 Blackwell, Workstation Edition, 400 W cap

A second host, deliberately measured against the first. Absolute numbers here are ~25% lower for the power
cap alone, so read them relatively; the controls are `c6_*` in `results/`.

| model (AA) | shape | out tok/s | in tok/s | TTFT p50 | tripwire |
|---|---|---:|---:|---:|---|
| gpt-oss-20b (24), 4 replicas | promptopt C256 | **13,442** | 188,192 | 0.6 s | 20/20 |
| | router C256 | 10,291 | 82,332 | 0.7 s | |
| Nemotron-3.5-Lightning-30B (24) | router C256 | 8,037 | 64,296 | 1.2 s | 18/20 |
| | judge C128 | 6,317 | 50,533 | 1.0 s | |
| Qwen3.6-35B-A3B (32) | promptopt C256 | 7,680 | 107,518 | 1.2 s | 16/20 |
| | router C256 | 6,903 | 55,220 | 0.8 s | |
| gemma-4-26B-A4B (26) | promptopt C256 | 7,705 | 107,870 | 1.1 s | 19/20 |
| gemma-4-26B-A4B **+ official MTP** | promptopt C256 | **9,560** | 133,834 | **0.22 s** | 20/20 |
| | judge C128 | 4,720 | 37,764 | 0.47 s | |
| Laguna-S-2.1 (agentic coder) | promptopt C256 | 4,527 | 63,382 | 1.6 s | 20/20 |
| | router C256 | 2,850 | 22,803 | 1.2 s | |
| Laguna-S-2.1 **+ DFlash drafter** | judge C128 | **2,455** | 19,643 | **0.63 s** | 20/20 |
| Nemotron-3-Super-120B (26) | router C256 | 2,845 | 22,761 | 1.9 s | 20/20 |
| Qwen3.8-27B **+ DFlash2 drafter** | judge C128 | 2,682 | 21,458 | 1.0 s | 20/20 |
| Ornith-1.5-397B | router C256 | 877 | 7,017 | 2.4 s | 20/20 |
| Hy3 (42) | promptopt C256 | 1,821 | 25,488 | 3.6 s | 20/20 |

**Speculative decoding buys latency, not throughput.** Every drafter measured loses output tokens per second
at saturation and cuts time-to-first-token by 2–5×: Nemotron-3.5 with NVIDIA's DSpark drafter goes from
1,161 ms to 244 ms TTFT while output drops from 8,037 to 4,305 tok/s. For an interactive agent that is the
right trade; for batch rollouts it is not. The two exceptions are gemma-4-26B's official MTP head and
Laguna's DFlash, which win on *both* axes.

### 8× RTX 5090 (32 GB), 400 W — the same monthly price

Scan sells an 8× RTX 5090 machine for **the same £1,999.98/month** as the 4× RTX PRO 6000, so this is a
straight read of throughput per pound: twice the tensor cores, a third of the memory each, 256 GB aggregate
against 384 GB.

| model | best kernel pair found | router C1024 | promptopt C1024 | judge C512 |
|---|---|---:|---:|---:|
| gpt-oss-20b | mxfp8 activations | **25,634** | **39,529** | 16,790 |
| gemma-4-26B-A4B NVFP4 | engine auto | 15,674 | 27,696 | 13,478 |
| Nemotron-3.5-Lightning-30B NVFP4 | `b12x` + Marlin experts | **16,154** | 10,773 | **13,838** |
| Qwen3.6-35B-A3B NVFP4 | `b12x` + Marlin experts | 13,654 | 15,352 | 10,212 |
| Muse-Glimmer-30B NVFP4 | `b12x` + FlashInfer CUTLASS | 9,782 | 23,833 | 8,056 |
| Qwen3.8-27B NVFP4 | `b12x` | 6,558 | 4,889 | 5,237 |

Head to head on the identical checkpoint, kernel and shapes:

| Qwen3.8-27B NVFP4, `b12x`, one replica per card | 4× PRO 6000, 600 W | 8× RTX 5090 | Δ |
|---|---:|---:|---:|
| router C256 | 4,288 | 6,445 | **+50%** |
| router C1024 | 5,161 | 6,558 | **+27%** |
| judge C512 | 4,400 | 5,237 | **+19%** |
| promptopt C1024 | 4,804 | 4,889 | +2% |
| KV cache per server | 1,500,000 tokens | 181,000 | −88% |

The consumer node wins everywhere and wins hugely on short-prompt routing; the advantage collapses to
nothing on shared-prefix traffic, where 32 GB per card leaves an eighth of the cache and the scheduler
queues instead of batching. Caveats worth carrying: this rented host is dual-socket (Scan's is not, so its
tensor-parallel tiers should be better than measured here), and its cards are also capped at 400 W.

**And then the memory ceiling arrives.** Everything above is a model that fits on one card. A model that
does not is forced into tensor parallelism across cards with no NVLink, where the 96 GB box still runs
independent replicas that exchange nothing. gpt-oss-120b is the one model measured both ways:

| gpt-oss-120b, same MoE kernel | 4× PRO 6000, four TP1 replicas | 8× RTX 5090, two TP4 groups |
|---|---:|---:|
| router, out tok/s | **13,752** (C2048) | 1,640 (C1024) |
| prompt-optimisation | **19,689** | 3,352 |
| judge | **9,152** | 1,578 |
| TTFT p50 | **0.7 s** | **324 s** |

Eight times the throughput and three orders of magnitude on time-to-first-token. That is not a tuning gap,
it is the configuration: 61 GB fits a 96 GB card and does not fit a 32 GB one. Two further tensor-parallel
arms failed there outright — Ling-3.0-flash needs 102,400 bytes of shared memory against the 101,376 the
architecture allows, and Qwen3.8-Flash-Next rejects an fp8 KV cache. **So the consumer box is the better
buy for high-volume work on models under ~30 GB, and the wrong buy for anything frontier-class**, which on
this roster means DeepSeek-V4-Flash (167 GB) and GLM-5.3-Flash (198 GB). Both are queued at TP8 there to
put a number on it rather than an inference.

---

## GLM-5.3-Flash: getting the index-57 model to work

Neither vLLM 0.28.1 nor SGLang 0.5.18 knows the `glm5_next` architecture; both vendors ship it as a
per-model Docker image. `box/pull_image.py` lifts those images' Python trees over the registry API without
Docker.

* **The vLLM route works.** Its only sm_120 sparse-attention backend is hard-wired to DeepSeek's cache
  layout (`pe_dim` 64; GLM has 0). Its Hopper NoPE backend supports that but was gated to major 9 and built
  on FlashAttention-3. `box/vllm_sm120_nope.py` widens the gate to major 12 and rebuilds the wrapper on
  FlashInfer's FA2 path. With `--block-size 1024` (DeepGEMM's paged-MQA indexer accepts arch 12 only at
  particular block sizes) and autotune off, it serves TP4 with a 2.14M-token KV cache and passes 18/20 on
  the chat tripwire.
* **The SGLang route does not.** Nine blockers cleared, server healthy in 135 s, prefill exact after a
  dense-prefill patch — and decode still drifts into loops. Ten discriminators cleared CUDA graphs, MTP, the
  MoE runner, TileLang tiles, PDL, the indexer, conv and SSM dtype. The TileLang sparse *decode* kernel is
  what is left. Documented in `notes/` rather than swept under it.

Credit: the DGX Spark (GB10, sm_121) community had already hit most of these and published fixes.

---

## Quality

Throughput numbers are worthless without a quality gate, and string comparison of greedy generations is not
one — a single differing token cascades, and it cannot tell better from worse. Three layers here:

**1. A corruption tripwire** (`box/quality20.py`), run on every configuration before it is benchmarked: 20
fixed prompts through the model's own chat template, a repetition detector (repeated 6-grams, distinct-token
ratio) and expected substrings. Reported as `ok/degenerate/wrong` in every table above. The first version of
this passed GLM output reading "111 222 333 444"; do not trust a tripwire that only checks for `!!!!`.

**2. Logit-level divergence** (`box/logit_diff.py`), for questions a task benchmark cannot resolve: identical
contexts on two servers, twenty log-probabilities per position, and a control pair that fixes the noise
floor. For fp8 KV cache on gpt-oss:

| metric | control: fp8 vs fp8 | treatment: fp8 vs bf16 | excess |
|---|---:|---:|---:|
| top-1 token agreement | 0.988 | 0.925 | −0.062 |
| top-5 overlap | 0.988 | 0.965 | −0.023 |
| mean KL divergence | 0.0053 | 0.0589 | **+0.054** |

**3. A task-accuracy suite** (`box/evalsuite/`), 435 items across six capability families — contest
mathematics, code executed against its tests in a sandbox, tool calling matched against the Berkeley
function-calling answer sets, synthetic long-context retrieval calibrated to the served window, ten-option
knowledge questions and short-answer factuality, and instruction following with two dozen checkers
re-implemented from the reference. Ungated public sources, programmatic scoring, no model judging another
model, Wilson intervals, and the same items and seed across configurations so arms are directly pairable.

### How to serve a model without accidentally measuring your own harness

The first round of quality numbers here was wrong, and the way it was wrong is the most useful thing in
this repository. Serving every model with one house default produced a table in which every model that
*reasons* scored below every model that does not — which reads like a finding and is in fact three bugs.

**The same weights, the same hardware, the same 403 items. Only the serving changed:**

| Qwen3.8-27B NVFP4 | house default | vendor recipe | |
|---|---:|---:|---|
| **overall** | 0.558 | **0.732** | |
| instruction following | 0.317 | **0.833** | its chain-of-thought was the graded answer |
| code | 0.413 | 0.733 | |
| maths | 0.525 | 0.625 | |
| tools | 0.829 | 0.900 | |
| long context | 0.917 | 0.979 | |
| truncation rate | 0.248 | 0.139 | |

Three independent errors, each worth more than any kernel choice in this repository:

1. **No reasoning parser.** vLLM registers one per family — `qwen3`, `glm45`, `muse_glimmer`, `nemotron_v3`,
   `ling3`, `minimax_m3`, `inkling`, `hy_v3`, `step3p5`, `poolside_v1`, `mimo`, `deepseek_v4`, `gemma4`,
   `openai_gptoss` — and we passed none. Without one the chain-of-thought is returned as the answer.
   It hides well: Qwen3-family templates put the opening `<think>` in the *prompt*, so the model emits only
   a closing `</think>` and the output reads as ordinary prose. 306 of 403 responses, and 21 of 60
   instruction items began "We need answer user's request…".
2. **A token cap.** A truncated answer was scored *wrong*. GLM-5.3-Flash lost 51% of the maths items that
   way. Cap generation just under the context window and let the *time* budget be the only limit — running
   out of time marks an item skipped, which is excluded from the accuracy.
3. **Sampling, and whether the model is even thinking.** Not one vendor recommends greedy or T=0.6. Qwen
   wants T=1.0/top_p=0.95/top_k=20/min_p=0 in thinking mode, gemma-4 T=0.0/top_k=64, Ling-3.0 T=0.85,
   Hy3 T=0.9/top_p=1.0. `top_k` and `min_p` have no slot in the OpenAI schema and must go through
   `extra_body` or they silently do not apply. And thinking is not on by default everywhere: **Hy3 defaults
   to `no_think` and gemma-4 to `enable_thinking: false`**, while Qwen, Inkling, MiniMax, Nemotron, Ling and
   Ornith default on — so a naive table compares models in two different modes.

`box/lists/profiles.tsv` carries the researched recipe per model (parser, tool parser, template kwargs,
sampling), applied by the harness to both the server and the eval client. Every pre-fix run is kept under
`results/eval/capped/`, `no_parser/` and `pre_profiles/` so the size of each artefact stays measurable.
The full re-run is in flight; the corrected table will replace this section rather than sit beside it.

### Does four-bit cost quality? On this model, no — and it is 3× cheaper to run

The first, buggy round suggested NVFP4 lost about five points to FP8. Served correctly, on the same 403
items and seed, that gap disappears:

| Qwen3.8-27B, vendor recipe, 8× RTX 5090 | overall | maths | code | tools | long ctx | knowledge | instructions | GPU-min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **NVFP4**, `b12x`, one replica per card | **0.732** | 0.625 | 0.733 | 0.900 | 0.979 | 0.429 | 0.833 | **78** |
| **FP8**, `b12x`, TP2 × 4 | 0.740 | 0.700 | 0.733 | 0.814 | 0.979 | 0.443 | 0.867 | 260 |

0.008 apart, against a 95% interval of roughly ±0.045 — statistically indistinguishable, with each format
ahead on three of six families. And the four-bit configuration did it in **a third of the GPU-minutes**,
because it fits one card and the FP8 build does not. On intelligence per pound that is not a close call.

Two honest limits on this pair: the layouts differ (the FP8 checkpoint cannot fit a 32 GB card, which is
itself part of the answer), and this is one model family. The professional box is running the full ladder —
native BF16, FP8, the community PTQ build, a quantisation-aware-trained build and a second independent PTQ
build, all at TP1 with the same items and seed — plus logit-level divergence of every rung against the BF16
parent. That separates "four-bit is lossy" from "that particular upload is bad", which is the question
worth answering before standardising on a checkpoint.

### Which four-bit releases are lossless

Worth separating before reading any quantisation comparison. These are **natively quantised** — the
low-precision weights are the trained artefact, so there is nothing to recover: **gpt-oss** (MXFP4 experts),
**DeepSeek-V4-Flash** (MXFP4 experts + FP8 attention), **MiMo-V2.5** (FP8), **both Nemotron 3 models**
(pre-trained with an NVFP4 recipe), and **zai-org's own GLM-5.3-Flash release, which is native FP8** — which
means a community NVFP4 build of it is a re-quantisation of an already-quantised model, not of a BF16
parent. Conversely Google publishes no official 4-bit `gemma-4-26B-A4B` and advises against serving one.

Two findings from building it that generalise:

* **Give reasoning models room.** The first run scored 181 of 403 items as wrong because they hit the token
  cap mid-thought (52 of 80 in maths). These are hybrid reasoners that think in the visible channel with no
  tags to strip. The truncated run is kept in `results/eval/truncated_2048/` as the evidence.
* **Adversarial review of a scorer is not optional.** Every family was attacked by an independent reviewer
  after it was written, and every one had false-positive paths: the maths scorer accepted decimal
  approximations of irrational closed forms and read "3 or 5" as 35; short-answer scoring accepted any answer
  whose tokens were a superset of the gold; the multiple-choice extractor manufactured an answer from a
  trailing capital letter about one time in ten.

---

## Layout

```
box/          the campaign scripts, pulled back from the nodes verbatim
  ksweep.sh       kernel/backend sweep: walks candidate pairs per model, keeps every pair that serves,
                  derives memory and sequence budgets from the card, resumable
  kldiff.sh       logit-level pass: two servers side by side, control pair first
  evalsuite/      the 435-item task-accuracy suite (runner, families, mock server, statistics)
  quality20.py    the corruption tripwire
  logit_diff.py   next-token distribution comparison
  pull_image.py   lifts a vendor Docker image's Python tree over the registry API, no Docker needed
  vllm_sm120_nope.py, dsa_sm120.py, pdl_patch.py, kda_patch.py   the sm_120 ports
  lists/          per-host sweep lists (model, TP, candidate kernel pairs, extra flags)
results/      per-run probe JSON, per-tag summaries, summary_all.tsv (327 rows), kernel lines per server
  5090/           the 8× RTX 5090 tree, kept separate
report/       the full write-up, including economics and the open questions
notes/        failure analyses that did not fit anywhere else
```

`results/summary_all.tsv` is one row per (configuration, shape, concurrency) with req/s, in/out/total tok/s
and TTFT/TPOT/ITL/E2E at p50 and p99. `results/kernels_by_server.tsv` records the kernel line each server
actually selected, which is the only way to know what was measured.

## Reproducing

```bash
# on a fresh sm_120 box with the vLLM cu130 image
uv pip install --system --pre "vllm[b12x]" flashinfer-python==0.6.18 flashinfer-jit-cache==0.6.18 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match --torch-backend cu130
# the matching cubin lives only on FlashInfer's root index; the image ships a mismatched one
uv pip install --system --no-deps flashinfer-cubin==0.6.18 --index-url https://flashinfer.ai/whl/

bash box/ksweep.sh box/lists/<your-list>.txt              # throughput + tripwire
MODE=eval bash box/ksweep.sh box/lists/<your-list>.txt     # task accuracy on the same servers
bash box/kldiff.sh                                         # logit-level divergence
```

Treat all of this as evidence about the state of sm_120 software in September 2026, which changes weekly.
The measurements are honest about their limits: the first host had PCIe ACS enabled, so its tensor-parallel
numbers are pessimistic lower bounds; the second is power-capped; the 5090 host is dual-socket. Where a
configuration failed, the reason is recorded rather than the row omitted.
