# Primary-source engineering review, 5 September 2026

This preserves the primary-source model/runtime review from the earlier workspace. It is a source review, not a report of newly reproduced GPU results. Current GLM pricing, verified Scan cost, workload billing and excluded failed/corrupt measurements are in [the current GLM economics audit](glm_economics_20260905.md); that report supersedes the earlier economic assumptions. Captured host2 evidence is in [the host2 audit](host2_live_audit.md).

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

