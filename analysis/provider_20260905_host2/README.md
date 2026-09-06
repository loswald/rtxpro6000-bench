# Existing provider baseline audit

Captured read-only from host2 on 5 September 2026, around 21:15 UTC. Only the named evaluation artifacts were read; no API key was accessed, no paid request was sent and no remote process was changed. `manifest.json` records each captured file's hash, size and modification time. Every local copy was verified against the bytes read on the host. The capture is not atomic across files; this matters for the still-running DeepSeek evaluation.

## GLM baseline is complete

`or_glm` requested `z-ai/glm-5.3-flash` through `https://openrouter.ai/api`. It completed all **403 cases**, scoring **332 correct (82.38%)**, with **zero request errors, retries, cancellations or skipped cases**. There were **25 truncated cases** and one result marked degenerate. The dataset manifest is `efb50b88d5b0aaae7f92d527bf05c761c0da6c9c153af5bde45e1add2ed3735b`, matching the local suite.

It used temperature **0.95**, top-p **0.95**, `min_p=0`, seed **20260903**, concurrency **24**, a 2,700-second wall budget and 3,600-second request timeout. Completion caps were math **32,768**, code/knowledge **20,480**, instruction following **16,384**, tools **8,192**, and long context **6,144**. These caps and sampling match the latest local 600 W GLM runs; local concurrency was 96. All runs completed, so their different wall budgets did not leave missing cases.

The provider run did **not** pin a specific OpenRouter backend. Its saved records contain neither the actual routed provider nor returned checkpoint revision/precision, so the family slug does not prove a specific FP8 or FP4 deployment was used. It also omitted an explicit API `reasoning_effort=max`: the launcher removed the vLLM chat-template option, while the local runs explicitly set it. The official GLM default is max, but actual provider behavior is not attested in these artifacts. Responses were saved in tail mode, and full request bodies/raw provider response metadata were not retained.

## What the existing results show

| Family | Cases | Provider | Local TP4 base | Local TP4 MTP | Local DP2 × TP2 |
|---|---:|---:|---:|---:|---:|
| Math | 80 | 65 | 59 | 65 | 45 |
| Code | 75 | 61 | 57 | 58 | 48 |
| Knowledge | 70 | 46 | 42 | 43 | 38 |
| Instruction following | 60 | 50 | 52 | 53 | 34 |
| Tools, JSON prompt mode | 70 | 63 | 64 | 63 | 50 |
| Long context, differing prompts | 48 | 47 | 46 | 44 | 44 |
| All cases | 403 | **332** | **320** | **326** | **259** |

The provider could not use `/tokenize`, so the long-context family used fixed fallback calibration: assumed model context 32,768, 3.7 characters/token and a 25,088-token cap on the nominal 32K prompt budget. Local calibration used a 40,960 context and approximately 4.6 characters/token, preserving the 32,768 prompt budget. The actual mean long-context prompt length was **13,030 tokens provider versus 18,619 local**. These 48 requests therefore do not form an exact comparison.

For the remaining **355 cases**, all observed per-case prompt-token counts match between provider and each local run, and their case IDs, completion caps and sampling match. This improves comparability, although identical token counts are not proof of byte-identical request bodies or identical hidden provider settings.

| Local candidate, excluding long context | Provider correct /355 | Local correct /355 | Provider-only correct | Local-only correct | Exact paired McNemar p |
|---|---:|---:|---:|---:|---:|
| TP4 base | 285 | 274 | 24 | 13 | 0.0989 |
| TP4 MTP | 285 | 282 | 19 | 16 | 0.7359 |
| DP2 × TP2 | 285 | 215 | 77 | 7 | 5.14×10⁻¹⁶ |

The large DP2 regression is also present entirely within local evidence: TP4 base scored 320/403 versus DP2's 259/403, with **68 TP4-only correct cases versus seven DP2-only**, paired p≈1.17×10⁻¹³. This supports rejecting the DP2 arm without buying another provider baseline. It does not identify the exact faulty kernel. The much smaller TP4/MTP gaps do not establish general NVFP4 degradation, and nonsignificant paired results do not prove quality equivalence.

The TP4 base artifact resumed **367 previously completed cases** after an earlier run failed, then reran 36 cases. Treat it as a combined result rather than a fresh uninterrupted pass. MTP and DP2 were not resumed. Do not reuse the API summary's generated token rate or its synthetic four-GPU accounting as measured provider capacity, actual GPU efficiency or useful-work goodput.

## DeepSeek capture is partial

`or_ds` correctly requested `deepseek/deepseek-v4-flash-0731`, with temperature 1, top-p 0.95 and the same completion caps. At capture, the latest saved summary had **127 scored of 403**, while the subsequently read item ledger contained **133 records**. The summary explicitly says `partial=true`; its `ended_at` is a checkpoint time, not proof the run ended. It is not eligible as a complete baseline. A later read-only capture can retain its final results separately after completion.

`audit.json` contains the complete paired case-ID lists, family counts, configuration differences, local source hashes and captured-provider metadata. Existing provider data already supplies a useful GLM diagnostic anchor. A strict preservation claim still needs a matched run with frozen long-context requests, pinned provider identity/effort, and the isolated code runner.
