# Review of the other Qwen result — 5 September 2026

Commit `4270d214a47b900f89a86e93716166aac804bfa7` reports a **real completed throughput measurement**, verified against the managed host's saved artifacts at **21:03:38 UTC**. It is a useful candidate result. Its model quality has not yet been established by the full evaluation, and it should not be dismissed merely because a native FP8 alternative is now running.

## Verified measurement and configuration

`/workspace/results/probe/qwen38fn_tp4m_--marlin/qwen38fn_tp4m_--marlin__router__c1024__p8000.json` records **8,192 completed requests, zero failed**, 727.062 seconds, and **1,442.210 output tokens/s**. The synthetic workload uses 1,024 input tokens and 128 forced output tokens per request, concurrency 1,024, the completions endpoint, and ignored EOS. Mean TTFT is 69.652 seconds; p99 is 80.066 seconds. This is throughput under substantial queueing, without a goodput/latency acceptance target. [Captured result scalars and hashes](qwen_other_result_detail_20260905.json).

| Setting | Other managed-host result | Native stock anchor |
|---|---|---|
| Checkpoint | RadixArk Qwen3.8-Flash-Next-NVFP4, revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | Official Qwen FP8, revision `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| Routed experts | NVFP4, Marlin | FP8 block-128, Triton |
| Layout | TP4, DP1, EP disabled | TP4, DP1, EP4 |
| Maximum sequences / batch tokens | 512 / 8,192 | 256 / 8,192 |
| Memory utilization | 0.92 | 0.90 |
| Model / KV / GDN state | BF16 / BF16 / FP32 | BF16 / BF16 / FP32 |
| Vision tower | Retained | Language-model-only |
| PLE | FP8 shards and global scale via existing loader-selection fix | Native recognized FP8 loader |
| Loaded weights per GPU | 35.55 GiB | 44.35 GiB |
| Initial cache capacity | 13.97 GiB / 967,845 tokens | 3.13 GiB / 216,691 tokens |
| Speculation | Disabled | Disabled |

The other run's `--kv-cache-dtype auto` resolves to BF16. Its model config declares `mamba_ssm_dtype=float32`; the pinned `Qwen4ExpForConditionalGenerationConfig` uses the Qwen3.5 cache normalization that copies that declaration into the recurrent-state cache when the cache setting is auto. The other run therefore does not use a reduced GDN state precision. [Launch, checkpoint and state-source capture](qwen_other_precision_20260905.json).

The native run's low initial cache capacity has a concrete compiler-allocation explanation; see the [native startup audit](qwen_native_fp8_audit_20260905.md). Neither the different resident memory nor the two cache capacities establishes which checkpoint delivers better useful throughput at matched quality.

## Quality status at inspection

`/workspace/results/probe/qwen38fn_tp4m_--marlin_quality20.json` contains **19 completed/accepted answers and one length-limited, degenerate answer**. For “Give a regex that matches a UK postcode,” final content is empty, reasoning length is 5,518 characters, the largest repeated six-gram count is 31, and the distinct-token ratio is 0.176. The smoke runner used a 2,048-token cap and greedy sampling. This is an unresolved failure of that diagnostic protocol. It does not isolate quantization as its cause, and it cannot establish a regression against native FP8 without a matched baseline.

No Qwen full-403 evaluation artifact was present under the managed host's result directories at inspection. The prompt-optimization throughput run was still in progress; the W4A4-linear variant also had no completed result. The new frontier entries in the commit identify intended result names, not completed quality measurements. [Result listing and progress capture](qwen_other_result_audit_20260905.json), [smoke answers and full-evaluation search](qwen_other_result_detail_20260905.json).

A useful follow-up is a matched, complete-answer quality comparison using the same checkpoint arm and declared reasoning budget, followed by a workload and concurrency comparison. The native opaque-PLE capacity check is independent of that question. GLM is the user's current priority; these findings do not justify delaying it for another broad Qwen sweep.
