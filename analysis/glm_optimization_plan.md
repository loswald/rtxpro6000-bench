# GLM-5.3-Flash: correctness diagnosis and next throughput experiment

Source audit, 2026-09-05. All live evidence below was copied read-only from the existing host. No running host was changed. The proposed kernel change has **not** been benchmarked or quality-qualified here.

## Decision

Use the existing **TP1 + DP4 + EP4, max sequences 192** run as the GLM candidate for a fresh full evaluation. Reject the nominally faster **TP2 + DP2 + EP4, max sequences 384** run: its 20-item smoke test reports eight degenerate outputs and one wrong answer. Several nominally passing outputs also contain repetitive text, so the smoke verdicts may undercount the problem.

The strongest new implementation opportunity is native SM120 NoPE sparse attention, now merged in FlashInfer. It removes the current generic FA2 attention route and its per-step host planning requirement. It preserves the checkpoint weights but introduces FP8 KV and query arithmetic; maintaining output quality therefore needs a paired capability evaluation, not an assumption of numerical identity.

The immediate experiment that preserves current precision is a matched reproduction of good/bad layouts with graph capture disabled, plus an exact sparse-length diagnostic. This can isolate a correctness failure before tuning the faster layout.

## What the captured runs establish

| Run | Actual parallel groups | Local attention heads | Router output tok/s | Prefix workload output tok/s | Smoke verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `glm53f_dp4ep4_s192` | TP1, DP4, EP4 | 64 | 1073.34 | 880.00 | 20/20 marked OK |
| `glm53f_dp2tp2ep2_s384` | TP2, DP2, EP4 | 32 | 1299.60 | 2104.86 | 11 OK, 8 degenerate, 1 wrong |
| `glm53f_s512_moecutlass` | TP4, DP1, no EP | 16 | 830.34 | 915.72 | 20/20 marked OK |

The DP2 run name says `ep2`, but its four worker ranks form EP4. Both DP runs selected `FLASHINFER_CUTLASS` NVFP4 experts and `MoEPrepareAndFinalizeNaiveDPEPModular`. Both used the patched `FLASHINFER_MLA_SPARSE_SM90` backend, BF16 model/query arithmetic, `kv_cache_dtype=auto` (BF16 on this path), async scheduling, prefix caching, chunked prefill, 40960 context, 16384 batched tokens, disabled custom all-reduce, and no speculation. The layouts and sequence limits both changed, so this is not a controlled kernel comparison.

Both logs warn that there is no dense-MHA prefill backend for this model; prefill also runs through sparse top-k MQA. The requested attention block size was 1024, but hybrid state allocation changed the **actual** block size to 5120 tokens for TP1 and 3072 for TP2. Mamba page padding was 20.75% and 44.91%, respectively. Performance and memory analysis must use these effective settings.

Raw evidence is under `results/live_20260905_host1/results/{smoke,probe}`. `analysis/glm_candidate_screen.json` records the quality-filtered candidate selection. These historical runs lack all newly required provenance/cache controls; they identify a candidate, not a promotion-ready cost claim. The complete old GLM base suite is 320/403 with 32 truncations counted wrong. Its resumed run timings are not a valid throughput measurement, and it is not an exact full-suite result for the DP4 candidate.

## Reproduction pins

- Model: `RedHatAI/GLM-5.3-Flash-NVFP4`, downloaded revision `36c184c6cda000a481711306df5adde42f63321a`.
- Vendor model code: `v0.1.dev20051+g487ecf187`, extracted from `vllm/vllm-openai:glm53-flash-x86_64-cu130` and imported through a separate `PYTHONPATH`.
- The public reproducer identifies image digest `sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703`. Verify the downloaded manifest against that digest before claiming the exact image was reproduced. [vLLM issue #53963](https://github.com/vllm-project/vllm/issues/53963)
- Actual captured host packages include Torch `2.13.0+cu130`, Transformers `5.15.1`, FlashInfer `0.6.18`, compressed-tensors `0.17.0`, and Triton `3.7.1`. The image alone does not reproduce the runtime because only its vLLM package was imported into that environment.
- Captured patched backend SHA256: `7a19dafb16f1a2f9ac58992ce78e4d27b8f52edf08059c387d4f32d70d0edab3`.

## Concrete source hazards, with limits on the conclusion

The captured backend is `results/live_20260905_host1/glmimg/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py`.

1. **A planned-length mismatch silently changes attention.** `forward_mqa` obtains exact `valid_counts` from sparse-index conversion but ignores them. It clamps negative indices to physical KV slot zero, while the FA2 wrapper uses host-planned row lengths. If a planned length exceeds the actual count, slot zero contributes extra terms to the softmax. This is memory-safe addressing with incorrect attention semantics. The existence of the code hazard is proven; the captured results contain no trace proving that this mismatch caused the DP2 corruption.
2. **One global wrapper assumes identical geometry and no overlapping plans.** `_SM90_STATE` and its private workspace are process-wide and initialized by the first attention implementation. Reuse does not verify heads, device, dtype, scale, top-k width, or token capacity. Each DP worker has a separate process, so this is not by itself evidence of cross-DP state corruption. It becomes a problem if differing attention groups, target/draft paths, or overlapping metadata schedules share one instance.
3. **The host plan is a real performance target.** Every metadata build creates per-row lengths, populates full 16384-row CPU staging arrays, and replans FA2. Because async scheduling is enabled in both logs, the captured code reads exact device positions back to CPU, adding a synchronization. It is therefore incorrect to blame an optimistic CPU sequence-length upper bound without checking which branch actually ran. Profile the synchronization, planning, index conversion, sparse prefill, MoE, and collectives separately before assigning a percentage bottleneck.

TP4 without EP passed the smoke screen, TP1 with EP passed it, and TP2 with EP failed it. This makes the TP/EP combination, per-rank state, graph lifetime, and scheduling interactions plausible leads. It does not prove a double all-reduce bug, a universal NVFP4 accuracy defect, or a particular FA2 head-count bug.

## Experiments in order

### A. Preserve precision and locate the corruption

On the independent test host, reproduce the two layouts with pinned packages and the same model files. First run the 20 prompts sequentially at temperature zero with request-to-DP-rank attribution; repeat the failing prompts enough times to distinguish one bad worker from a content-dependent fault. Save all text, token IDs when available, finish reasons, and runtime logs. Then compare:

| Change from the same failing TP2/DP2/EP4 launch | Diagnostic meaning |
| --- | --- |
| Disable graph capture with `--enforce-eager` | A correction points toward capture, replay, buffer lifetime, or padded metadata |
| Disable async scheduling using the pinned build's supported setting | A correction points toward metadata synchronization or overlapping execution |
| Disable EP while retaining TP2/DP2, if memory permits | A correction narrows the problem to expert dispatch/reduction or its interactions |
| Use TP1/DP4/EP4 at matched workload and scheduler capacity | Confirms the quality candidate on the new host and avoids treating old smoke data as a baseline |

In eager diagnostic runs, compare actual `valid_counts` against the most recent host-planned lengths **before** the negative-index clamp, every layer. Log rank, layer, active row count, physical block size, local head count, first mismatching row, planned length, actual length, and query position. Check that the wrapper's geometry matches the layer. Use short contexts and boundaries around 2048, all four kpool remainders, actual physical page boundaries, ragged batches, and chunked prefill. Run the test after initialization so dummy profiling rows are identified separately. A nonzero mismatch should stop that experiment; it must not be converted into a serving workaround that masks the error.

Compare FA2 against a direct sparse-attention reference at heads 16, 32, and 64 using identical BF16 queries, cache values, actual sparse slots, and scale. Include graph replay with changed indices and row lengths separately, because an eager match does not validate replay. This test can distinguish metadata errors from numerical/backend errors.

### B. Native SM120 sparse attention with unchanged weights

FlashInfer PR #4802 merged on 2026-09-03 at commit `453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc`. It includes `GLM53_NOPE`, a persistent `flashinfer.mla.SparseMLASm120Wrapper`, TP1/64-head support, and graph-safe scratch ownership. Decode covers the relevant head counts; prefill accepts the correct padded width of 2176. Upstream reports 658 GPU tests on RTX PRO 6000, but the published speed tables are not an end-to-end GLM comparison against this FA2 fallback. No GLM throughput gain is claimed here. [FlashInfer PR #4802](https://github.com/flashinfer-ai/flashinfer/pull/4802)

The native format uses 512 FP8 latent values, four arbitrary FP32 scales, and reserved bytes to retain a 656-byte row ABI. It has no positional-key contribution. Its numerical tests compare against dequantized KV, which tests the kernel but does not establish parity with the original BF16 cache. [Native NoPE PR #4791](https://github.com/flashinfer-ai/flashinfer/pull/4791), [pinned reference tests](https://github.com/flashinfer-ai/flashinfer/blob/453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc/tests/attention/test_sparse_mla_sm120.py)

Required adapter contract:

- Preserve the actual model configuration: absorbed query width 512, RoPE width zero, `index_topk=2048`, `index_kpool=4`, always-selected incomplete tail, padded index width 2176, and the layer's existing attention scale.
- Instantiate the public wrapper with `kv_scale_format="arbitrary_fp32"`, `d_v=512`, and appropriate token/head capacities. At query width 512, `auto` selects the DeepSeek V4 family, which has different layout semantics. [Pinned runner implementation](https://github.com/flashinfer-ai/flashinfer/blob/453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc/flashinfer/mla/_sparse_mla_sm120.py)
- Implement and verify GLM's FP8 KV writer with four arbitrary scales and 656-byte contiguous rows. Convert request-relative indices to actual physical token slots. Present packed storage as contiguous 64-token pages, independent of vLLM's larger hybrid logical block size; assert that the storage strides make this view valid. Never pass the captured 3072/5120 page size directly to a kernel compiled for 64.
- Retain and pass the exact valid-count information with the native masking convention. Keep all runner scratch and buffers alive for every captured graph; warm each captured shape before capture. Do not use a transient runner inside `forward_mqa`.
- Run upstream GLM NoPE decode, prefill, and swapAB tests on the new build, then integration tests with real kpool/tail/physical-slot mappings, followed by full model evaluation. Benchmark CPU-inclusive serving, not only GPU kernel events.

The older vLLM PR #53969 demonstrates a zero-RoPE compatibility route but describes a model altered to `index_topk=2044` to fit width 2048. That is not this checkpoint's attention configuration and should not be copied into a same-quality optimization. Native support now admits width 2176. [vLLM PR #53969](https://github.com/vllm-project/vllm/pull/53969)

This is an integration experiment, not an instruction to replace FlashInfer in the active shared host. Build it in a separate pinned environment and keep the existing baseline reproducible.

### C. Promote only after the paired gate

Run the complete fixed capability suite for the reproduced baseline and the candidate with identical prompts, templates, sampling, limits, scoring, and model revision. Count truncations, malformed outputs, exceptions, and skipped items as failures according to the gate. Require aggregate and per-family quality acceptance; inspect paired regressions, not just two rounded percentages. Measure at least three complete serving repetitions per accepted workload with controlled cache state, complete request counts, output-token accounting, runtime fingerprints, and an observed memory/latency record. Only an accepted quality result and integrity-valid throughput result can enter the economics comparison.

If the native FP8 attention arm loses quality, the same-precision route remains available: repair any demonstrated FA2 metadata/lifetime fault, profile and remove unnecessary host planning/synchronization, and evaluate a BF16 NoPE kernel implementation independently. Do not hide a quality loss by reducing top-k, context, output budgets, or evaluation coverage.
