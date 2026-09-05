# Throughput evidence audit — 5 September 2026

Scope: DeepSeek-V4-Flash, GLM-5.3-Flash, and Qwen3.8-Flash-Next. No new GPU measurements were run by this audit. A subsequently authorized read-only inspection found much newer work on an existing machine controlled by another person. The current findings below supersede the original local snapshot recorded later in this document. Recorded throughput is diagnostic until output quality is independently scored on the same configuration.

## Current evidence: live host 1, captured 5 September at 20:26 UTC

The isolated snapshot is `results/live_20260905_host1/`. `manifest.json` records 187 bounded result/script files (8.58 MB) with source hashes and timestamps. `suite_manifest.json` records the existing six-family evaluation suite and fixed datasets (9.26 MB), the GLM patch and package sources, model configuration/index and download-revision evidence. No existing service, process, cache, file or setting was changed remotely.

The existing host was actively evaluating `glm53f_best`; the captured `glm53f_best.json` is **partial**, despite its separate `valid: true` field. It cannot establish completed quality. Chain logs attribute the current scheduled work to Nish's programme and explicitly reserve time until 22:00 UTC; this machine is not free for another overlapping test.

| Current model / configuration | Retained throughput | Quality evidence and decision |
|---|---|---|
| GLM-5.3-Flash **NVFP4**, DP2 × TP2 + EP, 384 sequences per rank | Router **1299.60** out tok/s; shared-prefix **2104.86**, both C1024, 4096/4096 completed | **Reject as a quality-preserving winner:** chat smoke has 11/20 `ok`, 8 degenerate and 1 wrong. One arithmetic answer repeats `bul` hundreds of times. The existing speed-only selector chose it anyway; a full evaluation is running. |
| GLM NVFP4, DP4 + EP, TP1, 192 sequences per rank | Router **1073** out tok/s; shared-prefix **880**, both C1024, 4096/4096 completed | All 20 smoke verdicts `ok`. This is the next configuration to evaluate, not a demonstrated capability-equivalent winner. |
| GLM NVFP4, TP4 CUTLASS, 512 sequences | Router **830** out tok/s; shared-prefix **916**, C1024, 4096/4096 completed | All 20 smoke verdicts `ok`; slower alternative. |
| DeepSeek-V4-Flash, DP4 + EP, TP1, b12x linear / automatic MoE | Router **1288.79** out tok/s at C256, 2048/2048; judge **1593.44** at C512, 4096/4096 in 1316.12 s | New sustained judge evidence exists. No full quality run for this exact DP4 tag was captured; compare to a matching baseline before promotion. |
| Qwen3.8-Flash-Next **NVFP4**, two TP2 replicas | No successful throughput measurement | Four recent auto/CUTLASS launch attempts failed with missing `ngram_embedding.weight_scale` in `Qwen4ExpNGramEmbedding`. This is a loader/quantization compatibility issue, not demonstrated poor throughput. |

Sources are the corresponding tag directories under the snapshot's `results/probe/`, `results/glm_perf2.log`, `results/keval_qwen38fn.log`, `results/ksweep_qwen38fn.log`, and `results/ksweep_ds4.log`. The NVFP4 checkpoints are different precision variants from the earlier proposed official FP8 cells; preserve checkpoint identity in every comparison.

Additional GLM TP4/TP4+EP/MTP C1024 rows have only 976 completed requests and thousands of failures. Their existence in a TSV is insufficient for selection. The copied `pick_best.py` only ranks `out_tps` in TSVs; it never inspects failures or smoke verdicts. The new local `bench/select_candidate.py` fixes this selection step for the **next evaluation**: it reads raw results, rejects incomplete requests/outputs and missing or bad smoke verdicts, and by default requires measurement integrity metadata. Explicit `--legacy-screen` permits archived provenance-unknown candidates for a fresh rerun only. `analysis/glm_candidate_screen.json` selects `glm53f_dp4ep4_s192` and records why the faster corrupt arm is excluded. It never certifies deployment or quality preservation.

The full existing GLM base quality result is **320/403 correct (79.4%)**, zero errors/skips, 32 token-truncated answers counted wrong, with fixed manifest `efb50b88d5b0aaae7f92d527bf05c761c0da6c9c153af5bde45e1add2ed3735b`. It covers tools, code, math, long context, knowledge and instruction following. It was resumed: the final aggregate contains 403 items but the latest run's request/time counters cover only the 36 resumed requests. Use it for item-level quality comparison; do not infer a serving-rate gain from its mixed timing fields. Reuse the captured suite and fixed inputs for a new pair instead of silently replacing those six capabilities with a smaller suite.

The working GLM runtime is vendor vLLM **0.1.dev20051+g487ecf187**, extracted from `vllm/vllm-openai:glm53-flash-x86_64-cu130`, with only its `vllm` package on `PYTHONPATH=/workspace/glmvllm`. It uses the host's torch 2.13.0+cu130, Transformers 5.15.1, FlashInfer 0.6.18, compressed-tensors 0.17.0 and Triton 3.7.1. `vllm_sm120_nope.py` adapts the vendor's SM90 sparse MLA backend to sm_120 using FlashInfer FA2 and adds it to backend dispatch. The actual model is `RedHatAI/GLM-5.3-Flash-NVFP4`; its retained config download metadata records revision `36c184c6cda000a481711306df5adde42f63321a`. The copied baseline and runtime files make an isolated reproduction reviewable; the image digest and complete installed-tree equivalence still need pinning on the new machine.

The DP4 GLM smoke-passing launch is fully reconstructible from `bench/glm_perf2.sh`: TP1 / DP4 / expert parallelism, sequence cap 192, batch budget 16384, context 40960, memory utilization 0.90, KV `auto`, block size 1024, attention `FLASHINFER_MLA_SPARSE_SM90`, speculation off, prefix cache on, custom all-reduce off, FlashInfer autotune off, alias `m`, port 8000. Its performance launcher lacks reasoning/tool parsers; for the paired capability run use the captured `glm_eval.sh` settings (`glm45`, `glm47`, auto tool choice, T0.95, top_p0.95, min_p0, reasoning effort max) on **both** baseline and candidate, retaining the same caps and fixed suite.

## Archived local snapshot: 2–3 September, superseded where noted above

### What the original local copy contained

- **DeepSeek-V4-Flash-0731:** 18 retained benchmark JSON files, including one duplicate. Later long runs establish a promising DP4 + EP4 / TP1 baseline. Earlier probes often overwrote their own raw files.
- **Qwen3.8-Flash-Next-FP8:** no retained throughput, startup, or quality measurements. Its two cells are proposed configurations, not an economic result. Qwen3.8-27B measurements concern a different model and cannot substitute.
- **GLM-5.3-Flash:** no retained throughput or attempted-load result. `results/REPORT.md` says it was not attempted. `cells/glm53flash_fp8_tp4_loadtest.env` describes a missing architecture in the then-installed main build and an expected failure; that is not a measured local launch failure. Establish current architecture and sm_120 kernel support before a large download.

## Strongest retained DeepSeek evidence

All numbers below are aggregate for the four-GPU machine. The long runs use random-token `/v1/completions` requests, fixed output lengths, and infinite request rate. They are single observations, not repeated confidence intervals or capability scores.

| Configuration | Shape | Concurrency | Completed / requested | Output tok/s | Total tok/s | Seconds | Mean / p99 TTFT |
|---|---|---:|---:|---:|---:|---:|---:|
| TP4 + EP, Marlin | router 1024 → 128 | 256 | 2048 / 2048 | 1084.35 | 9759.11 | 241.75 | 3.50 / 22.37 s |
| TP4 + EP, Marlin | router 1024 → 128 | 512 | 2048 / 2048 | 1206.81 | 10861.30 | 217.22 | 7.98 / 45.97 s |
| DP4 + EP4, TP1 | router 1024 → 128 | 256 | 1536 / 1536 | 1326.29 | 11936.63 | 148.24 | 6.61 / 17.44 s |
| DP4 + EP4, TP1 | router 1024 → 128 | 512 | 3072 / 3072 | **1474.77** | 13272.95 | 266.63 | 8.33 / 32.26 s |
| TP4 + EP, Marlin | shared prefix 3072 + unique 512 → 256 | 512 | 2048 / 2048 | 3002.26 | 45033.85 | 174.63 | 5.52 / 25.17 s |
| DP4 + EP4, TP1 | shared prefix 3072 + unique 512 → 256 | 512 | 3072 / 3072 | **3683.00** | 55244.95 | 213.53 | 6.97 / 20.06 s |

Source JSONs: `results/probe/ds_marlin_ep/ds_marlin_ep__{router__c256,router__c512,promptopt__c512}__p8000.json` and `results/probe/vllm_dp4/vllm_dp4__{router__c256,router__c512,promptopt__c512}__p8000.json`; client parameters and seeds are retained in the adjacent `{shape}_c{concurrency}.log` files. `results/ds4_proper.log` identifies the layouts. The DP4 observations use six times concurrency worth of requests; the TP4 observations use four or eight times concurrency. The +22.2% router and +22.7% shared-prefix throughput are promising comparisons, with differing run lengths and incomplete exact late-launch provenance.

The shared-prefix result counts cached input tokens in total throughput. Its 55,245 total tok/s is not fresh-prefill capacity and must use cached-input provider pricing where applicable. These legacy results lack the complete metadata and explicit cache-reset verification required by the new integrity gate; they identify rerun candidates and are not retrospectively certified headline or cost results.

There is **no valid retained sustained judge or long-rollout saturation result**. The best clean retained finite-batch rollout is TP4 b12x, `results/probe/n_ds4_tp4_b12x/n_ds4_tp4_b12x____c.json`: 1168.10 output tok/s, 8192 → 2048, C64, 64/64 completed in 112.21 s. That has only one concurrency wave. Its TSV reports short 1827 output tok/s at C256 and judge 1029 at C128, but those earlier raw results were overwritten; treat them as provisional. The intact Gen5 TP4+EP judge JSON is only C16, 64 requests, 521.86 output tok/s.

## Exclusions and measurement defects

1. **Partial runs were reported as throughput.** TP4 router C1024 completed 976/2048 with 1072 failures. Its log says `Too many open files`; its 1182 tok/s cannot establish saturation. TP4 judge C256 completed 304/2048 with 1744 failures, invalid streams and internal-server errors; even its successful output averages about 426 tokens instead of 512. Exclude both from ranking and cost.
2. **Fixed-seed cache contamination.** `results/box-bench-scripts/probe.sh` uses seed 1234 for all shapes and concurrency levels while enabling prefix caching. Later higher-concurrency calls reuse earlier prompts. The early Marlin/EP router 1544, judge 2550 and rollout 1523 output tok/s therefore cannot represent fresh-input results. The later `probe_v2.sh` varies seeds by shape and concurrency; this reduces contamination but does not prove a fresh cache on reruns of the same point.
3. **Raw files overwritten.** Both legacy probe scripts compute `f="${TAG}__${shape}__c${C}.json"` in the same `local` declaration that initializes `shape` and `C`. Shell expansion sees those names before assignment, producing `<tag>____c.json`. Successive points overwrite it. Early tables survive only as rounded TSV/log lines. One router JSON was also copied unchanged into two directories, so it is not an independent replicate.
4. **No saturation proof from a burst.** Most early high-concurrency points use requests equal to concurrency. Infinite arrival rate alone is not evidence of sustained saturation. The later long runs are better, but there are no repeated runs or plateau after the successful C512 points; the C1024 client failure leaves the next rung unresolved.
5. **Quality guard is not quality evidence.** `quality20.py` only flags request errors, `!!!!`, or fewer than three distinct words. It uses raw completions with a 128-token cap and no chat template. The DP4 output includes repeated examples, unfinished HTML and instruction-format errors despite “0 suspicious”. No capability-preservation claim follows from this check. Late winning DP4 files have no linked quality run.
6. **Hardware provenance changes.** The README's old ~21 GB/s same-switch / ~38 cross-switch / ~19 ring figures do not describe all later tests. `results/hwtruth/gen5_hwtruth.log` and `nccl01/02/4.json` show ~38.7 same-switch / ~26.44 cross-switch / ~27.65 ring at 512 MiB on the later hardware. Do not carry one box's pairing or ACS inference to another. The causal claim “MoE GEMM is not the bottleneck” is also stronger than these unprofiled, partly contaminated comparisons support.

## Reproducible starting configuration and remaining provenance

The archived `results/box-bench-scripts/launch_ds4flash_dp4ep4.sh` is a complete **earlier** DP4 configuration: physical GPUs 0,1,2,3, one API endpoint at port 8000, TP1, DP4, expert parallelism, model `deepseek-ai/DeepSeek-V4-Flash-0731`, plain fp8 KV, block size 256, b12x linear backend, automatic MoE backend (server log confirms Marlin), max model length 40960, max sequences 256, max batched tokens 8192, memory utilization 0.92, FULL_AND_PIECEWISE CUDA graphs, prefix caching, and the sm_120 o_proj fallback enabled. Speculation is off. Earlier smoke log records 203,189 KV tokens per DP rank.

The winning later `vllm_dp4` run is identified as DP4 + EP / TP1 in `results/ds4_proper.log`, with 192,946 KV tokens and 2.69 GiB CUDA graphs. Its client logs target `http://127.0.0.1:8000`, alias `ds4`. Its exact server command was not archived here. Promote the demonstrated topology and known runnable earlier recipe, but rerun and capture the exact new launch instead of claiming a bit-for-bit reproduction of the later winner.

## Next experiments, in order

1. Persist unique raw files, input identities, all expected replica shards, exact launch/version/patch hashes, cache reset verification, client return codes and full completion/output counts. Failed and provenance-unknown runs remain diagnostic only.
2. Run the DP4+EP/TP1 baseline against TP4+EP on the same host, same prompts, same decoding settings, cache policy and request counts. Router and judge: C128/256/512; add C768/1024 only after fixing the client descriptor limit. Use at least six to eight waves and repeated runs. Add long rollouts at a feasible KV budget.
3. Tune DP4 batch budget 4096/8192/16384, then sequence cap and safe KV budget, one change at a time. Measure preemption, cache hit rate, request failures and GPU memory alongside throughput. Profile attention, o_proj fallback, expert dispatch/combination and collective time before investing in a kernel rewrite. TP1 attention removes attention tensor-parallel collectives, while EP communication remains.
4. Score real chat/API outputs across reasoning/math, executable coding, instruction/JSON following, tool use and long-context retrieval against an unmodified reference. Gate every speed change by paired task accuracy and failure rates; retain reasoning budgets and output caps. Score useful answers per second, not merely forced-length random tokens.
5. DSpark k7 on random prompts loses at every archived matched point; accepted length is ~1.77–1.87. It is unsuitable as the current saturation default. If revisited, test shorter draft lengths on real target prompts, with acceptance and quality checked, before investing further.
6. Qwen Flash-Next: verify the selected engine actually implements the PLE table offload option, measure per-worker host/GPU memory, then compare TP4 to two TP2 groups selected from fresh hardware measurements. Current offload-memory claims are untested.
7. GLM Flash: use a build that recognizes its architecture and implements sparse MLA on sm_120; archive a small successful quality check before a sweep. SGLang DeepSeek startup failed during initialization after compilation (`results/sgl_ds4.log`); this is no throughput comparison and should first be resolved in an isolated environment.
