# GLM-5.3-Flash: capability and break-even audit

5 September 2026. At the verified Scan list price, the complete historical GLM DP4 run would need **1,459 output tokens/s on router traffic** or **2,138 output tokens/s on shared-prefix traffic** to match today's promotional API bill at 70% realized demand. Reaching those targets requires **36% and 143% more throughput** than its measured 1,073 and 880 output tokens/s. These are engineering targets conditional on preserving quality and meeting the chosen latency requirement; the existing runs have not passed those gates.

The corrupt `glm53f_dp2tp2ep2_s384` arm is excluded. The `glm53f_tp4ep4_s512` arm is also excluded: each raw router/promptopt result contains **976 completed and 3,120 failed requests out of 4,096**. A summary table that retains their token rates is not sufficient cost evidence.

## Price basis and model identity

Scan's four-card product **LN160437** lists **£1,999.98 including VAT**, with four 96 GB RTX PRO 6000 cards, 512 GB RAM, an EPYC 9354P, and a **30-day subscription**. The product page is in “Inc VAT” mode. Full VAT recovery makes the arithmetic £1,666.65 ex VAT; eligibility remains a business assumption. Its explicit 720-hour duration replaces the README's 730-hour month. At the repository's **assumed** £1 = $1.35, this is **$3.12497 per calendar hour**, or **$4.46424 per active hour at 70% demand**. No commitment discount, ERIS relief or resale income is assumed. Additional allocated operational costs would increase the targets. [Scan product](https://www.scan.co.uk/products/3xs-sc-pb4-32t-1-month-4x-96gb-nvidia-rtx-pro-6000-512gb-ddr5-ecc-amd-epyc-9354p)

Z.ai identifies the endpoint as **GLM-5.3-Flash**. Current USD prices per million tokens are **$0.075 uncached input / $0.015 cached input / $0.25 output**. The corresponding list prices are **$0.15 / $0.03 / $0.50**. The 50% promotion ends at **16:00 UTC on 9 September 2026**; do not extend it across a two-year commitment. Cached-input storage is currently marked temporarily free. [Z.ai pricing](https://docs.z.ai/guides/overview/pricing)

The captured OpenRouter endpoint response labels the model `z-ai/glm-5.3-flash-20260826`. It reports FP8 for GMICloud, Novita and Z.AI at the promotional rates, and FP4 for DeepInfra at the same rates. Those provider-reported labels do not establish the exact weight revision, local NVFP4 equivalence, identical reasoning behavior or matched task accuracy. Pin the provider for any paid capability comparison. The local historical launcher uses `/workspace/models/GLM-5.3-Flash-NVFP4`; matching the family name alone is insufficient. [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints)

The official GLM model card says `reasoning_effort` defaults to `max` and should remain there for benchmark reproduction. Lowering thinking effort or completion budgets changes the capability comparison. The local 403-case suite is a bounded regression suite, with materially smaller task/context budgets than several vendor benchmarks; it does not reproduce all vendor claims. [Official GLM model card](https://huggingface.co/zai-org/GLM-5.3-Flash)

## Targets at Scan list and 70% realized demand

All targets assume the generated output is usable and satisfies the latency requirement (`g=1`). For a retained fraction of 0.8, multiply the targets by 1.25. Realized demand is the paid-month fraction of benchmark capacity used for billable work, not the GPU utilization counter.

| Traffic and provider billing | API band | Required output tok/s | Historical DP4 tok/s | Throughput multiplier required | Demand needed at historical speed |
|---|---|---:|---:|---:|---:|
| Router: 1,024 uncached input, 128 output | Promotion | 1,459 | 1,073 | 1.36× | 95.2% |
| Router: same tokens | List | 729 | 1,073 | 0.68× | 47.6% |
| Shared prefix: 3,072 cached + 512 uncached input, 256 output | Promotion | 2,138 | 880 | 2.43× | 170.1%, impossible |
| Shared prefix: same tokens | List | 1,069 | 880 | 1.21× | 85.0% |
| Same long input, **no** provider cache hit | Promotion | 954 | 880 | 1.08× | 75.9% |
| Same long input, **no** provider cache hit | List | 477 | 880 | 0.54× | 37.9% |

The 3,072-token prefix is **85.7% of input tokens**, not a 90% cache discount. GLM's cached rate is one fifth of its uncached rate. This makes the shared-prefix API bill **$0.00014848 per request** during the promotion, versus **$0.00033280** if all 3,584 input tokens are billed uncached. Historical local cost at 70% demand is **$0.00036073 per request**. Router costs are **$0.00010880 API** versus **$0.00014792 local**. Cache residency, reuse eligibility and the fraction actually billed as hits must come from the real traffic/API records; a repeated local prefix does not prove the provider will bill every prefix token as cached.

For an explicitly assumed equal **number of requests** across router and shared-prefix traffic, the historical local cost is about **$0.00025432 per request**, against **$0.00012864 promotional API** or **$0.00025728 list API**. Uniformly increasing both measured throughput rates would need roughly **1.98×** to match the promotion; at list prices that selected mix is approximately at arithmetic parity. Different task mixes give different answers. These are costs of the same tokens at the same selected mix, not evidence that longer reasoning output is more valuable.

## What the measurements do and do not qualify

The retained reference is `glm53f_dp4ep4_s192`, TP1 × DP4 with expert parallelism, at concurrency 1,024. Both raw files completed **4,096/4,096 requests with zero HTTP failures**. Router throughput is **1,073.0807 output tok/s**; shared-prefix throughput is **880.0454**. Its 20-case tripwire returned “ok,” but no complete paired capability result exists for this layout in the evidence used here. Another layout's 403-case score cannot qualify it.

This is also a high-latency saturation reference: router median/p99 time to first token is **104.3/118.8 seconds**, and shared-prefix median/p99 is **253.1/283.8 seconds**. The raw `request_goodput` field is null. It does not establish an interactive serving result or SLO goodput. Fresh matched tests must retain actual task completions, quality, prompt/output/reasoning lengths, cache statistics and request latency rather than treating all generated tokens as successful work.

The cost calculation is `API = output_rate + (input/output) × [(1-hit_fraction) × input_rate + hit_fraction × cached_rate]`, per million output tokens including their associated input. Required output throughput is `1,000,000 × calendar_hour_cost / (3,600 × demand_fraction × retained_fraction × API)`. Divide the provider bill and local bill by the same accepted task count when output lengths differ.

The next GLM result needs to establish a correct baseline and then measure an improvement against **1.36× router / 2.43× shared-prefix promotional targets**, or the lower list-price targets if those are the relevant future bill. Public prices alone provide no evidence of provider costs or margins; the targeted speedups follow directly from the measured workload and billing arithmetic.

## Reproduction and evidence

Run `python bench/glm_economics_audit.py` from AIRR to regenerate `report/priority/glm_economics_20260905.json`. It reads raw complete results through `bench/economics.py`, retains source hashes, and marks every scenario as unqualified for selection. The generic calculator's nine tests pass. Dated API observations are preserved in `openrouter_priority_models_20260905.json` and `glm_openrouter_endpoints_20260905.json` beside this report. No paid API request or GPU benchmark was run to create this audit.
