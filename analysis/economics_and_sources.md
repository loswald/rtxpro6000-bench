# Throughput economics and source audit

Audit date: 5 September 2026. Scope: **GLM-5.3-Flash, DeepSeek-V4-Flash-0731, Qwen3.8-Flash-Next**. No remote benchmark, model download, or paid API call was performed for this audit. New calculations are scenarios, not new GPU measurements.

**Later live evidence:** a read-only follow-up recovered a completed 403-task DeepSeek-0731 evaluation and newer high-concurrency measurements from a managed host. It also captured Qwen's repaired scale loader followed by compilation OOM, plus a reproducible package/patch bundle. See `analysis/host2_live_audit.md`; those observations supersede the initial local-file quality gap below. They do not by themselves establish matched-runtime quality parity or production goodput.

## What the project establishes

The project has not established that an optimized Scan node loses a matched-quality cost comparison. It has established software and interconnect constraints on a rented four-GPU system, with an incomplete accuracy assessment and no Scan quote in the files reviewed.

| Evidence | Finding | Decision implication |
|---|---|---|
| `results/REPORT.md` | Recorded whole-node rental cost $4/hour; PCIe P2P works, but cross-switch pair all-reduce is reported around 38 GB/s against 21 GB/s for same-switch pairs and 19 GB/s for TP4. ACS is the report's explanation. | Use the measured topology when comparing TP4 with two cross-switch TP2 replicas. Do not turn a suspected topology explanation into a guaranteed Scan speedup. |
| `results/REPORT.md` | DeepSeek needed b12x linear kernels and a cached BF16 attention output-projection fallback. Its native checkpoint was retained. DSpark k=7 reportedly reduced throughput, with about 1.8 mean acceptance length. | A working configuration is not necessarily a fast native SM120 path. Profile the fallback and vary speculation length on real text before ruling speculation out. |
| `results/REPORT.md` | Twenty temperature-zero coherence prompts; explicitly no accuracy benchmark in that session. Throughput used random tokens, fixed generation lengths and ignored EOS. | This detects gross corruption, not preservation of reasoning, coding, tool use, or leaderboard quality. Synthetic output throughput is not useful-work throughput. |
| `results/openrouter/deepseek_deepseek-v4-flash.json` | Its model name is **DeepSeek V4 Flash 0423**; endpoint names end in `20260423`. The first stored endpoint charges $0.0679 input and $0.168 output per million tokens. | These stored cheap prices are for a differently labelled checkpoint from the local **0731**. They cannot establish matched-model economics without an explicit equivalence evaluation. |
| `bench/summarise.py` and `vast/COST.md` | The summary reports both output and total token unit costs, while the cost note's example divides by total tokens. Neither represents provider input/cache/output billing for a specified traffic mix or paid idle hours. | Always retain the denominator. A total-token price, output-token price, and provider blended price are different quantities. |

The hypothesis that inference providers have large margins is not demonstrated by these files or the sources below. Public retail price is not evidence of their hardware cost, utilization, subsidies, temporary discounts, or margin. None of those assumptions is needed to set a measurable break-even target.

## Current identity and capability evidence

Artificial Analysis' current comparison labels DeepSeek explicitly **0731 (Reasoning, Max Effort)**, scores it 41, and labels Qwen 46 with an asterisk indicating an estimate pending independent evaluation. The page specifies Intelligence Index **v4.2**. These establish identity and the evaluation status, not the quality of this project's locally served configurations. [Artificial Analysis comparison](https://artificialanalysis.ai/models/comparisons/qwen3-8-flash-next-vs-deepseek-v4-flash)

The current GLM page reports 46 under **v4.2**; a trailing-slash/search snapshot still showed 57 under **v4.1.1** during this audit. Do not compare those numbers across versions. Class ranks on that page are also not ranks across every model. [Artificial Analysis GLM page](https://artificialanalysis.ai/models/glm-5-3-flash)

Epoch's available DeepSeek-V4-Flash page shows a 24 April release, ECI 146 and 90% interval 144–148. It does not identify the **0731** revision in the retrieved content. Exact current ECI entries for Qwen3.8-Flash-Next and GLM-5.3-Flash were not verified, so no ECI rank is assigned here. [Epoch DeepSeek page](https://epoch.ai/models/deepseek-v4-flash)

## Engineering priorities supported by primary sources

### GLM-5.3-Flash: establish a correct SM120 lane first

The official vLLM recipe gives about **306 GiB native FP8 weights**, requires a dedicated Docker integration and FlashInfer 0.6.17+, and demonstrates TP4 plus MTP5 on **GB200**. Its generic Blackwell claims do not verify RTX PRO 6000. Its NVFP4 alternative is a different quantized checkpoint. [GLM vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)

Upstream vLLM issue **#53963 remained open** when checked. It records three distinct SM120 failures: packed FP8 cache writing assumes a 64-wide RoPE block although this model has none; BF16 has no selected sparse backend; allowing the SM100 backend through dispatch still does not supply SM120 kernels. This is a kernel/layout problem, not a scheduler parameter that can be fixed by increasing concurrency. [vLLM #53963](https://github.com/vllm-project/vllm/issues/53963)

Newer SGLang tracking issue **#37813**, opened 3 September, reports a working SM120 configuration with native MTP and FP8 KV, using a **W4A16/NVFP4 re-quantized checkpoint** and several unmerged patches. It identifies FlashInfer **#4802** for NoPE sparse MLA, **#4687/#4827** for MoE addressing/workspace lifetime, and SGLang correctness fixes including kpool selection and captured-tensor lifetimes. The issue explicitly attributes measurements to its author and distinguishes internal and published image digests. This is a concrete experimental route, not an upstream support guarantee or evidence of quality parity. Start with pinning/reviewing those changes and qualifying a no-speculation lane; use a native FP8 reference before accepting the smaller checkpoint. [SGLang #37813](https://github.com/sgl-project/sglang/issues/37813)

For quality comparisons, GLM's card says to preserve **max** reasoning for leaderboard reproduction; `low` or `high` changes the thinking budget. A lower reasoning budget must not be presented as a throughput optimization that preserves the tested capability. [GLM model card](https://huggingface.co/zai-org/GLM-5.3-Flash)

### DeepSeek-V4-Flash-0731: optimize the measured bottlenecks

The current official RTX PRO recipe verifies **TP8+EP**, not this four-card deployment. For DSpark on 0731 it requires a nightly build containing vLLM #51538 and a FlashInfer SM120 sparse-MLA decode instantiation for `topk=192`. It explicitly disables the SM100 FP4 indexer cache and mega-MoE path. **0731 has no MTP head: use DSpark, not MTP.** These constraints must be checked against the pinned runtime; a new nightly alone does not prove correctness. [DeepSeek vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)

The model card supplies DSpark settings and recommends temperature 1.0, top-p 0.95 for agentic tasks and 1.0 otherwise. It recommends generous high/max generation budgets. Hold those settings, chat encoding and task budgets fixed while testing topology, batching, projection kernels and speculation. [DeepSeek 0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

Project-specific next experiment: compare the known working no-speculation lane against shorter DSpark drafts on representative text at both low and target concurrency; record accepted draft length and actual task completion. Random tokens are an especially weak proxy for a trained draft model. This proposal is an inference from the project measurements and the general guidance that speculative gains depend on workload, model and sampling, rather than a promised gain. [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

### Qwen3.8-Flash-Next: offload, topology, then native MTP

The official recipe's starting point is a dedicated Qwen image. It describes a 125B main model plus a 51B N-gram table, host offload, TP2 minimum on **GB300**, TP4/TEP4 validation and MTP3. PLE offload is required for its four-H100 example. Initial pipeline parallelism is unsupported. These are reasons to verify offload allocation and compare TP4 against cross-switch TP2 replicas, not proof of RTX PRO support. Validate exact tensor/cache layouts and host-memory duplication in runtime logs rather than estimating per-rank allocation from aggregate parameter counts. [Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)

That recipe also reports MTP losing throughput on its tested H100 workloads. Keep speculation an A/B option rather than a universal default. [Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)

**The targeted offload mechanism is real:** `VLLM_PLE_CPU_OFFLOAD=1` in the dedicated image registers `PleOffloadLayer` objects and uses `vllm/v1/ple_offload/connector.py` with IPC and a CPU worker serving N-gram rows. Issue #53960 provides logs and stack traces; its deadlock is specifically a TP1 GB10 report, not proof of a TP2 RTX failure. Verify registration, actual host/device allocation and a completed forward pass. A generic whole-layer `--cpu-offload-gb` is not evidence that this asynchronous lookup path works. [vLLM #53960](https://github.com/vllm-project/vllm/issues/53960)

The native FP8 checkpoint is about 172.8 GiB. Issue #54765 notes that a two-card allocation at memory utilization 0.93 leaves only about 177.8 GiB total, before runtime/KV overhead; targeted offload is therefore critical to the native TP2 plan. That issue concerns a different NVFP4 checkpoint's FP8 PLE scale-loader mismatch. Do not copy its mixed-quantization fix onto the official FP8 weights or infer successful native TP2 from its quantized results. Host-memory demand also changes if the PLE table is expanded to BF16. [vLLM #54765](https://github.com/vllm-project/vllm/issues/54765)

**Concrete native-FP8 performance candidate:** draft SGLang PR #36787 adds direct-paged Triton QSA, Triton MQA scoring, the SM120 FP32 GDN-state contract, targeted PLE load masking/NUMA allocation, and FP8 MoE configurations. It reports official FP8 **TP4/EP4**, BF16 KV, 1024-in/512-out throughput of **1682.2 tok/s at C64** without speculation and **1754.2** with NEXTN; its reported quality includes GSM8K 98.10%, GPQA 80.81% and IFEval prompt-strict 0.9316. These are the PR author's measurements on RTX PRO 6000 Server Edition, not this project's measurements or proof of parity across capabilities. The PR still depends on #36497 and remains a draft. Plain TP4 without EP fails its FP8 partition-block check (160 is not divisible by 128). Reproduce TP4/EP4 first, retaining the official checkpoint. [SGLang #36787](https://github.com/sgl-project/sglang/pull/36787)

The narrower exact-SM120 FlashInfer routing fix #36806 merged on 28 August into **qwen4-main-squashed**, not necessarily the installed `main` revision. It repairs dispatch; the direct-paged optimization is separate. Confirm the branch/commit rather than assuming every image includes it. [SGLang #36806](https://github.com/sgl-project/sglang/pull/36806)

The model card defaults to thinking and preserves historical thinking blocks. It describes benefits for consistency and cache reuse in agent conversations. Hold that behavior constant in comparisons; changes to historical reasoning or reasoning budgets alter the workload and can alter quality. [Qwen FP8 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)

## Matched-workload break-even calculation

Let `C` be the complete node cost per **paid calendar hour**, `T` measured whole-node output tokens/second on the chosen workload, `u` realized demand as a fraction of benchmark capacity, and `g` the fraction of generated output tokens retained after quality/latency qualification. Do not apply losses twice if `T` already measures goodput. Let `r = input_tokens / output_tokens`, `h` be the fraction of **input tokens** billed at a cache-hit rate, and `Pi`, `Pc`, `Po` be provider currency per million uncached input, cached input, and output tokens.

```text
Provider cost per million output tokens, including their associated inputs:
P = Po + r * ((1 - h) * Pi + h * Pc)

Local cost for the same quantity of useful output:
L = 1,000,000 * C / (3,600 * T * u * g)

Throughput required to break even:
T_required = 1,000,000 * C / (3,600 * u * g * P)

Utilization required at the measured throughput:
u_required = 1,000,000 * C / (3,600 * T * g * P)

Maximum affordable node hourly cost at that throughput/utilization:
C_max = 3,600 * T * u * g * P / 1,000,000
```

Count reasoning as output. If accepted requests differ in token length or require different retries, calculate token-weighted goodput and the accepted input/output mix from request-level records. For comparisons that change response length, also compare **cost per completed task**, not only per token. The calculator assumes provider cost represents that same accepted workload; extra cache-write/storage, tools, retrieval, minimum spend or retry charges must be added to effective provider rates before use. It does not model those charges separately.

The cache-hit fraction prices provider traffic only. A local prefix-cache speedup must be separately measured. `u` is not GPU SM utilization: a short burst with 100% GPU utilization can still yield low average use over a month. For a varied workload, segment by context/output/cache shape, determine each segment's cost per accepted task, then sum costs using the same task counts. Do not average unrelated token rates.

`C` should include the node quote or amortization/lease, hosting/power, financing, support, failure headroom and relevant storage/network costs. For an owned node use the chosen lifetime and residual value explicitly; avoid double-counting a capital cost already included in a lease. Use one currency and a documented exchange rate if conversion is needed. No Scan quote or exchange rate has been invented here.

For an ex-VAT expense `E`, VAT rate `v` and recoverable fraction `q`, effective cost is `E * (1 + v * (1-q))`. If full VAT recovery applies, a VAT-inclusive price is divided by `1+v`; it is not reduced by `v` percent and VAT is not removed twice. Apply consistent recovery treatment to node and API costs. Eligibility depends on the business and supporting invoices. [HMRC reclaiming VAT](https://www.gov.uk/charge-reclaim-record-vat/reclaim-vat-business-expenses)

### Price observations, not hard-coded defaults

DeepSeek's pricing page explicitly maps `deepseek-v4-flash` to **0731**. As retrieved, USD/million rates are **0.44 input / 0.014 cached input / 1.32 output** at peak and **0.22 / 0.007 / 0.66** off-peak. Peak hours are 01:00–04:00 and 06:00–10:00 UTC Monday–Friday. Weight by actual billed traffic in each time band, rather than merely hours in the week. These prices differ materially from the project's stored 0423 endpoints. [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/?article_id=article_1779470751466_8)

Z.ai's GLM-5.3-Flash list prices are **0.15 input / 0.03 cached input / 0.50 output** USD/million. A 50% promotion currently halves all three until **9 September 2026, 24:00 UTC+8**; cached-input storage is marked temporarily free. Run both promotional and list-price scenarios rather than extending a launch promotion over a two-year purchase. [Z.ai pricing](https://docs.z.ai/guides/overview/pricing)

The retrieved Alibaba pricing page includes `qwen3.8-flash`, but did not verify an exact `qwen3.8-flash-next` serving revision/rate. Do not substitute that similarly named endpoint or the price of Qwen3.8-27B. Require an exact provider-model mapping or a paired task-quality comparison. [Alibaba pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)

Illustration using **the project's recorded $4/hour rental cost**, input/output ratio 8, no cache hits, all output qualified, and 100% realized utilization:

| Provider scenario | Associated input + output cost per 1M output tokens | Required output tok/s |
|---|---:|---:|
| DeepSeek 0731 off-peak | $2.42 | 459.1 |
| DeepSeek 0731 peak | $4.84 | 229.6 |
| GLM-5.3-Flash list | $1.70 | 653.6 |
| GLM-5.3-Flash promotion | $0.85 | 1307.2 |

At 50% utilization each throughput threshold doubles. With 90% cached input tokens, the DeepSeek off-peak threshold becomes about **1253.5 tok/s** at full utilization: cheap cache hits make the API harder to beat. These are algebraic targets with explicit assumptions, not throughput measurements, Scan forecasts, or quality-qualified deployment recommendations.

## Using the calculator

`bench/economics.py` is Python standard-library-only. Node cost, common currency, utilization, cache fraction and provider input/output prices are required. Pass either measured throughput and observed mean token counts, or a raw whole-node vLLM benchmark JSON. It rejects failed/incomplete runs and conflicting token-throughput arithmetic. It does not infer a provider identity from a model nickname.

```bash
# Illustrative inputs; replace them with a matched run and dated quote/rates.
python bench/economics.py --currency USD --node-hourly-cost 4 --utilization .5 \
  --output-tps 1000 --input-tokens 4096 --output-tokens 512 \
  --input-rate .22 --cached-input-rate .007 --output-rate .66 \
  --cache-hit-fraction 0 --model DeepSeek-V4-Flash-0731-max \
  --provider DeepSeek-off-peak --out results/economics-example.json

python -m unittest discover -s tests -p test_economics.py -v
```

The result remains explicitly **provisional** unless both quality and latency gates are attested as passed. Those flags record externally established gates; they do not run them. The nine local tests cover matched billing, utilization, cache discount reversal, goodput, VAT arithmetic, zero-priced APIs, invalid numbers, benchmark consistency and the provisional default.
