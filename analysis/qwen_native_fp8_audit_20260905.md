# Native Qwen FP8 source and startup audit — 5 September 2026

The native `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint at revision `236dfdf285828023ca3bcd3f37366c58a3469b13` loads and compiles on four RTX PRO 6000 GPUs with pinned vLLM `0.28.1rc1.dev446+g798544433`, Torch `2.13.0+cu130`, FlashInfer `0.6.18` and Triton `3.7.1`. The stock TP4+EP4 run loaded **44.35 GiB per GPU**, compiled in **22.44 seconds**, and reached serving readiness. This establishes engine feasibility; it does not establish preserved model quality or a throughput advantage over another checkpoint.

## PLE is sharded, and the large startup allocation is a compiler input

The native checkpoint headers contain 128 FP8 PLE shards with a combined shape of **320,001,536 × 160** and 51,200,245,760 table bytes, plus a two-byte checkpoint scale. Runtime `PLEVocabParallelEmbedding` assigns each TP rank only its overlapping rows; the global scale is preserved. At TP4 the table is **80,000,384 × 160 = 11.92 GiB per GPU**. Earlier documentation describing a full table replicated on each TP rank was incorrect. The inspected implementation has no `VLLM_PLE_CPU_OFFLOAD` path. These findings come from the [installed source and checkpoint-header capture](qwen_native_fp8_source_audit_20260905.json).

The generated compile-time autotuning block calls `generate_example_value((80000384, 160), ..., torch.float8_e4m3fn)`. Installed Torch implements that through `rand_strided`, which generates a full FP16 random buffer before converting it to FP8. That single example input can therefore temporarily hold **23.84 GiB FP16 plus 11.92 GiB FP8**, in addition to the loaded model. This is an example allocation, rather than a copy of the checkpoint's values. See the [generated code capture](qwen_native_ple_generated_20260905.json) and [allocator source](qwen_native_autotune_allocators_20260905.json).

vLLM's memory profiler wraps the first `profile_run`, including compilation. The stock log records 45.57 GiB consumed by weights and other persistent allocations, a reported 36.77 GiB peak-activation term, and 1.10 GiB actual graph-pool memory. The resulting cache budget is only **3.13 GiB / 216,691 tokens**. The profiler's stored peak term already includes its graph estimate; these diagnostic lines must not be added together as independent measurements. [Profiler and log evidence](qwen_native_compile_peak_20260905.json).

The [opaque PLE patch](../patches/qwen_ple_opaque_lookup.md) targets this compiler input without changing the original lookup, scale or precision. Its effect on cache capacity and answers must be measured in an otherwise identical TP4 arm. [The opt-in cell](../cells/qwen38flashnext_fp8_tp4_ep_opaque.env) selects a fresh compile-cache root. The patch does not apply itself through the cell.

## Backend and state controls

The native FP8 block-128 MoE selects **Triton**, confirmed both by its guards and the live log. FlashInfer TRTLLM FP8 MoE requires SM100-family hardware; FlashInfer CUTLASS's block-FP8 scheme requires SM90. `b12x` is not a supported native-FP8 MoE selector. QSA uses its dedicated Triton backend and requires BF16 KV. GDN auto selects Triton/FLA on SM120. The anchor explicitly retains BF16 model/KV/convolution state and FP32 GDN recurrent state, with no speculation. [Installed source capture](qwen_native_fp8_source_audit_20260905.json).

The installed reset-prefix-cache endpoint awaits the engine's reset and returns `{"success": bool}`; false means blocks remain held. The strict cache controller can require `success: true` without accepting an empty HTTP 200 as proof. [Endpoint source capture](qwen_g798_reset_prefix_sources_20260905.json).

The separate [NVFP4 result review](qwen_other_result_review_20260905.md) records a genuine 1,442 output-token/s measurement. These experiments differ in quantization, expert layout, batch depth and memory budget and are not a matched quality/speed comparison.
