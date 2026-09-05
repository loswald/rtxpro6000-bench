# Captured host2 baseline

This directory contains byte-exact source files captured read-only from the second managed host, plus the corresponding original files and unified diffs. `manifest.json` records original and staged SHA-256 checksums. No live host was changed to produce this bundle.

The installed baseline is vLLM `0.28.1rc1.dev446+g798544433`, commit `7985444339e2ad7e249b88a50081e16e34637dfd`, Python 3.12, Torch `2.13.0+cu130`, Transformers `5.15.1`, Triton `3.7.1`, b12x `1.3.0`, compressed-tensors `0.17.0`, FlashInfer Python/cubin `0.6.18`, and FlashInfer JIT cache `0.6.18+cu130`.

`setup_pinned.sh` installs those named versions on the new host. It is a staged recipe, not a claim that the new host has already passed runtime validation. The wheel URL was recovered from `/opt/uv/cache/simple-v24/index/bd2611c0f5c17a12/vllm.rkyv`; an HTTP HEAD returned 200 and 316,905,401 bytes. Installed `direct_url.json` was absent. The direct commit URL avoids relying on the moving nightly index. The script leaves transitive versions to the package resolver; record the new host's complete package inventory after installation.

`ple_layer.diff` selects the existing FP8 PLE embedding loader when `VLLM_QWEN4EXP_PLE_FP8=1`. It retains the checkpoint scale. The patched TP2 experiment loaded weights successfully but failed during Torch Inductor autotuning with an additional 51,201,966,080-byte allocation. It therefore has no valid throughput or quality result. Native Qwen FP8 uses a recognized FP8 configuration and does not require this override.

`o_proj.diff` adds the DeepSeek SM120 BF16 fallback. Its behavior is enabled by `VLLM_DSV4_OPROJ_SM120_FALLBACK=1`. It dequantizes activation and block-weight FP8 values and uses BF16 einsum. The captured DeepSeek result used this implementation; it must be described as this fallback, not an unchanged FP8 kernel. Retain original weights and scales.

Native Qwen download candidate: `Qwen/Qwen3.8-Flash-Next-FP8` at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`, 185,563,783,577 repository bytes. RadixArk NVFP4 on host2 is revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, approximately 135 GB. Native TP4 on the new host is a useful quality anchor while the managed host independently explores NVFP4 TP4.
