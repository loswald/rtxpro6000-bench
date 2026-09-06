# Qwen PLE compilation memory workaround

Status: **CPU numerical and graph-boundary tests pass; the patch's GPU memory and throughput effects are not yet measured.** Stock native FP8 TP4+EP4 now loads and compiles, but a full-table autotuning example inflates its profiled peak and leaves only 3.13 GiB of cache. This is an opt-in diagnostic patch for that compiler allocation as well as the earlier TP2 OOM.

## Evidence and scope

The captured host ran vLLM `0.28.1rc1.dev446+g798544433`, PyTorch `2.13.0+cu130`, and `RadixArk/Qwen3.8-Flash-Next-NVFP4` with two TP2 replicas. Its prior loader fix selects `Qwen4ExpPLEFp8EmbeddingMethod` for the checkpoint's FP8 PLE shards and retains their global scale. Loading completed before Inductor failed in `generate_and_run_autotune_block`: an allocation of **51,201,966,080 bytes (47.69 GiB)** with **30.84 GiB free** and **64.12 GiB resident** on a 94.97 GiB GPU.

The earlier TP2 failure had no surviving generated autotune program, so its exact allocation was initially unproven. A subsequent **native FP8 TP4** run retained the generated code: autotuning creates a random FP8 example with the full per-rank table shape `[80000384, 160]`. Installed Torch first allocates an FP16 random tensor, then casts it to FP8, temporarily holding approximately 23.84 GiB plus 11.92 GiB. The profiler includes compilation and counts that transient when reserving memory. This mechanism is now directly evidenced for the native TP4 run; it is not a claim that the compiler clones the checkpoint's values. See the [source and startup audit](../analysis/qwen_native_fp8_audit_20260905.md).

The earlier TP2 evidence is in `results/live_20260905_host2/detail_bundle.json` (`compile_evidence`) and `clone_bundle.json` (`failures`). The preserved source is under `baseline_bundle/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/`.

This patch targets exactly these two audited files:

| Input | SHA256 |
|---|---|
| Original `ple_layer.py` from g798544433 | `859ae689a7a74b8e4d8ea8c62b3479dbb214314f4fb168ebc2cb963ab3e4a664` |
| Same file with the captured FP8 loader-selection fix | `78969d27e1feead35e2c9207d44c383c440b80e1bf563ab3dce50320409c195a` |

An unknown source version is refused. The loader-selection fix is preserved when present and is not added to a native FP8 deployment. No checkpoint tensor, quantization scheme, global scale, attention state dtype, token budget or model identity changes.

## Mechanism

`Qwen4ExpPLELayer.forward` normally exposes the embedding lookup and dequantization to compilation. With `VLLM_QWEN4EXP_PLE_OPAQUE_LOOKUP=1`, it allocates a small graph-owned output and invokes `vllm::qwen4_exp_ple_lookup_with_output`. The op gets the actual layer through the existing `get_forward_context().no_compile_layers` registry, then executes the **original** `ple_embedding(...)` and `_dequantize_embeddings(...)` functions and copies their result into that output.

The op has only token IDs, query offsets, n-gram context, output and a layer-name string as arguments. The table and scale cannot become direct FX/autotuning input tensors for this region. Original n-gram hashing, checkpoint/TP row mapping, masked FP8-byte reduction, cast-then-scale order and missing-scale/device checks remain in place. A shape/dtype guard prevents the final copy from silently changing an unexpected dtype.

The op is appended to the resolved `compilation_config.splitting_ops` list during model construction. It includes request-dependent n-gram generation, so it must remain outside token-count-only CUDA graphs, as the existing n-gram-ID op does. The rest of the model retains its original compile path. Using `torch.compiler.disable` directly would introduce a graph break into the captured AOT/fullgraph path; this custom-op boundary avoids that approach.

This does **not** offload PLE weights. The captured [common PLE implementation](https://github.com/vllm-project/vllm/blob/798544433/vllm/models/qwen4_exp/common/ple.py) and [NVIDIA PLE implementation](https://github.com/vllm-project/vllm/blob/798544433/vllm/models/qwen4_exp/nvidia/ple_layer.py) are GPU-resident. A `VLLM_PLE_CPU_OFFLOAD` setting is not an implemented optimization in these files. Permanent model memory is unchanged. The new output is 40 MiB at 8,192 tokens × 2,560 dimensions × BF16, plus the original requested-row intermediates and a copy; its storage scales with requested rows, not vocabulary size.

## Controlled GPU experiment

Use a fresh server process for each arm and retain logs, source fingerprints, launch configuration and quality captures. Use separate compile-cache roots for the stock, eager and patched arms; do not delete shared caches or alter another running server.

1. Start the **unpatched** native FP8 checkpoint on TP4+EP with 40,960 context, BF16 KV, BF16 convolution state, FP32 GDN recurrent state, and no speculative config. Leave `VLLM_QWEN4EXP_PLE_FP8` and `VLLM_QWEN4EXP_PLE_OPAQUE_LOOKUP` unset. Do not override compilation for the first attempt.
2. If compilation reproduces the OOM, an otherwise identical `--enforce-eager` arm can distinguish compilation from permanent model/lookup memory. When stock compilation succeeds but the profiler reserves a table-sized transient, compare directly with the opaque compiled arm. Eager mode disables both compilation and CUDA graphs and is a diagnostic, not an assumed throughput winner.
3. Review and apply the selective patch to that stopped server's local environment, then restart the same arm with the opaque flag set and without `--enforce-eager`.

```bash
python3 patches/qwen_ple_opaque_lookup.py --check
python3 patches/qwen_ple_opaque_lookup.py --diff
python3 patches/qwen_ple_opaque_lookup.py --apply
export VLLM_QWEN4EXP_PLE_OPAQUE_LOOKUP=1
# Start the same model/configuration in a new process and a separate compile cache.
```

The prepared `qwen38flashnext_fp8_tp4_ep_opaque` cell inherits the native anchor and selects a new `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`. `QWEN_OPAQUE_CACHE_ROOT` can specify an explicit fresh root. The cell does not apply the source patch; application remains a separate reviewed operation on a stopped deployment.

`--target /explicit/path/to/ple_layer.py` can be supplied to each command. Application saves a unique `.qwen_ple_opaque_lookup.orig` backup and JSON audit with before/after hashes. `--revert` restores that precise file, preserving any earlier loader fix, but refuses intervening source/backup changes. Audit artifacts remain after revert; reapplication requires a fresh deployment copy rather than silently replacing patch history. Setting the opaque flag to `0` also selects the original forward path in a fresh process.

For g798544433, the state controls are `--dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32`. The latter controls GDN recurrent state while the former Mamba flag keeps convolution/PLE state at BF16. These are verified in the pinned [argument definitions](https://github.com/vllm-project/vllm/blob/798544433/vllm/engine/arg_utils.py), [cache configuration](https://github.com/vllm-project/vllm/blob/798544433/vllm/config/cache.py) and Qwen model's GDN state-dtype forwarding. Keep all other engine/request settings identical across the pair.

Success requires more than server startup: inspect peak GPU memory and verify absence of the giant autotune allocation, compare warm throughput and latency at matched concurrency/context/output budgets, and run paired capability checks with preserved reasoning budgets and complete final answers. Neither matching synthetic tensors nor an allocation fix establishes whole-model quality equivalence. Compare the **same checkpoint**; a native-FP8 versus NVFP4 comparison is a separate quantization experiment.

## Local verification

Five standard-library tests check both source fingerprints, preservation of loader/scaling function ASTs, custom-op arguments, idempotent application, exact restoration and refusal of unknown/intervening edits. Three real-PyTorch CPU tests cover FP8 E4M3/E5M2 and BF16 rows, repeated IDs, empty batches, nontrivial scales, unchanged table/scale bytes, missing-scale errors, dtype mismatch, and fullgraph tracing with dynamic row counts. Captured FX graphs contain the opaque op and exclude table/scale input tensors. The graph test uses a recording backend; it does not claim to reproduce GPU Inductor or vLLM CUDA-graph execution.

```bash
python3 -m unittest discover -s tests -p 'test_qwen*.py' -v

# Reproduce all eight tests in an isolated CPU environment, if torch is absent:
uv run --no-project --with 'torch==2.8.0+cpu' \
  --index https://download.pytorch.org/whl/cpu \
  python -m unittest discover -s tests -p 'test_qwen*.py' -v
```

All eight passed locally with PyTorch `2.8.0+cpu`. The production environment's PyTorch `2.13.0+cu130` and vLLM splitting behavior still require the controlled GPU test above.
