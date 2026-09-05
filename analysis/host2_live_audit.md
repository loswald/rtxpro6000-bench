# Read-only host2 evidence, 5 September 2026

The second managed machine has a completed DeepSeek-V4-Flash-0731 quality run. The Qwen3.8-Flash-Next NVFP4 attempts inspected here had no completed serving result. No existing process, configuration, GPU setting, or file on a managed host was changed during this audit.

Evidence is staged in `results/live_20260905_host2/`. Initial `snapshot_bundle.json` preserves source hashes and collection metadata; `runtime_inventory.json`, `clone_bundle.json`, and `latest_status.json` extend it. Individual benchmark/evaluation files are unpacked under `results/`. The byte-exact implementation bundle is `baseline_bundle/`, with SHA-256 comparisons in its manifest. The last captured controller event is at 20:29 UTC; ongoing controller actions can supersede that status.

## DeepSeek: native checkpoint, completed bounded quality run

The local directory `/workspace/models/DeepSeek-V4-Flash` contains a model card explicitly identifying **DeepSeek-V4-Flash-0731**. Download metadata identifies revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. This resolves the ambiguity introduced by the shortened local directory name.

The completed `ds4flash_dp4ep4_s512_b12x--` evaluation scored **340/403 correct (84.37%; recorded 95% interval 80.50–87.59%)**, with all 403 planned items attempted and scored, zero request errors/retries, 17 truncations and one unparsed answer. The generated-output throughput during this evaluation was 1,048 tokens/s. This is generated output, including reasoning and failed/truncated tasks; it is not correct-answer goodput.

| Family | Correct / scored | Accuracy | Truncated |
|---|---:|---:|---:|
| Code | 56 / 75 | 74.67% | 10 |
| Instruction following | 56 / 60 | 93.33% | 1 |
| Knowledge | 45 / 70 | 64.29% | 1 |
| Long context | 45 / 48 | 93.75% | 0 |
| Math | 75 / 80 | 93.75% | 5 |
| Tools | 63 / 70 | 90.00% | 0 |

The request settings used thinking, temperature 1.0, top-p 0.95, evaluation concurrency 64, maximum model length 40,960, and per-family output caps between 6,144 and 32,768 tokens. Code was capped at 20,480; its truncation rate was 13.33%. Tools were evaluated in prompt mode by parsing JSON lists, so this run does not validate the server's native tool-call parser. The serving alias was `m`; the result's suite Git revision was null. Preserve these limits and provenance gaps when interpreting the score. A same-checkpoint reference run with the same harness/settings is still needed to quantify any runtime quality loss.

## DeepSeek: high-concurrency capacity and latency

Captured `summary_full.tsv` for the same DP4+EP4, maximum-sequences-512 configuration:

| Workload label | Concurrency | Completed | Output tok/s | Input tok/s | Mean TTFT | p99 TTFT | p99 TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| router | 1,024 | 8,192 | 1,640 | 13,124 | 9.888 s | 59.515 s | 627.84 ms |
| promptopt | 1,024 | 8,192 | 4,430 | 62,024 | 7.943 s | 33.371 s | 229.18 ms |

The matching TP4/b12x run reports 1,245 and 3,031 output tokens/s respectively. DP4+EP4 with maximum sequences 1,024 reports 1,584 and 4,293, so increasing that scheduler limit did not improve these two runs. These are queue-saturating capacity measurements with very high tail latency. Prefix caching was enabled; cache-state isolation still needs separate verification. Neither the output total nor input-plus-output total is a measured production goodput or a provider billing comparison.

The implementation retains the native checkpoint, uses b12x and a captured SM120 attention output-projection fallback, and disables the unsupported FP4 indexer-cache path. The fallback explicitly dequantizes FP8 values and performs BF16 einsum; it is not an unchanged native FP8 kernel. Its exact original and modified source files are in the baseline bundle.

## Qwen: the scale loader was repaired, then compilation ran out of memory

The managed host downloaded `RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, to `/workspace/models/Qwen3.8-Flash-Next-NVFP4`. Its card describes routed-expert W4A4 quantization, FP8 PLE n-gram tables, and BF16 attention/shared experts/other components. Its approximate file size is 135 GB (126 GiB displayed by the host).

The initial DP4+EP4 attempt failed while allocating 95.37 GiB per rank. Two TP2 replicas initially failed because the mixed checkpoint's FP8 PLE scale did not have a registered parameter. The managed controller then applied `patch_ple.py`, which selects the existing FP8 PLE method when `VLLM_QWEN4EXP_PLE_FP8=1`. The selected method registers and validates `weight_scale`; the layer applies the scale after embedding lookup. The patch does not discard the scale.

After that change, both TP2 replicas loaded weights but failed at about 20:26 UTC during Torch Inductor's `generate_and_run_autotune_block`. The additional allocation was **51,201,966,080 bytes (47.69 GiB)** with only 30.84 GiB free and 64.12 GiB already resident per rank. The stack establishes compilation/autotuning OOM, not a second scale-loader failure. It suggests the giant PLE table contributes to compiler temporary allocations, but the exact allocation path was not recovered from generated code. An opaque lookup operation or an eager startup diagnostic can preserve the embedding/scaling math while testing that hypothesis.

The last captured controller status was `chainc7`, starting a single TP4 Qwen engine and then a W4A4 linear-kernel arm around 20:29 UTC. These ongoing managed experiments should not be duplicated blindly on the separate new machine. There was no valid Qwen throughput or quality result in the captured snapshots. The launch used BF16/auto KV, maximum length 40,960, sequences 512, batched tokens 8,192, no targeted PLE CPU offload, and `qwen3_coder` as tool parser; the official Qwen recipe uses `qwen3_xml`.

## Reproduce the package baseline on the separate new machine

Python is `/usr/bin/python3`, version 3.12. Packages are in `/usr/local/lib/python3.12/dist-packages`. The staged `baseline_bundle/setup_pinned.sh` specifies vLLM `0.28.1rc1.dev446+g798544433`, Torch `2.13.0+cu130`, Transformers `5.15.1`, Triton `3.7.1`, b12x `1.3.0`, compressed-tensors `0.17.0`, FlashInfer Python/cubin `0.6.18`, and FlashInfer JIT cache `0.6.18+cu130`.

The exact vLLM wheel URL was recovered from the host's uv cache and verified with an HTTP HEAD (200; 316,905,401 bytes). Installed `direct_url.json` was absent. The commit is `7985444339e2ad7e249b88a50081e16e34637dfd`. The recipe uses the immutable [vLLM wheel URL](https://wheels.vllm.ai/7985444339e2ad7e249b88a50081e16e34637dfd/vllm-0.28.1rc1.dev446%2Bg798544433-cp38-abi3-manylinux_2_28_x86_64.whl), not a moving nightly version. FlashInfer cubin 0.6.18 must be obtained from its root wheel index. Record a complete new-host package inventory after installing; the staged script pins the relevant baseline packages but not every transitive dependency.

`pull_image.py` is also captured. It extracts selected Python packages from OCI layers without Docker. No vendor-image extraction tree was present on host2 at collection; active Qwen and DeepSeek used the system nightly package. The existing scripts contain many campaign controllers and destructive cleanup functions, so copy only the inspected baseline pieces needed for the new experiment.

## First independent new-host arm

Use official `Qwen/Qwen3.8-Flash-Next-FP8` at revision **`236dfdf285828023ca3bcd3f37366c58a3469b13`** as the quality anchor while host2 pursues the smaller NVFP4 checkpoint. The Hub API and pinned index show **185,502,232,570 tensor bytes** (185,563,783,577 repository bytes). The checkpoint contains **128 FP8 PLE shards and one global BF16 scale**; a ranged safetensors-header read verifies first-shard shape `[2500012,160]` and scale shape `[1]`. Native quantization metadata correctly selects the FP8 PLE loader. [Official checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13)

Begin with TP4 plus expert parallelism, BF16 KV, FP32 GDN state and no speculation. This is an experimental recommendation, not a measured result. The expert intermediate width is 640; dividing it by TP4 gives 160, which is not aligned with the native 128-wide FP8 block format. The native SM120 SGLang work explicitly uses TP4+EP4 for that reason. Exact vLLM behavior must still be checked. Confirm one forward pass and scale-preserving output before running the bounded quality suite, then test concurrency and the safe PLE compiler mitigation. [SGLang native-FP8 SM120 work](https://github.com/sgl-project/sglang/pull/36787)
