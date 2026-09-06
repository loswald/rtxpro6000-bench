# Native Qwen FP8: scope check before a linear-backend experiment

The pinned vLLM `g798544433` source supports `--kernel-config.linear_backend b12x` for ordinary block-FP8 linear layers on SM12x. **That does not yet justify a benchmark arm for this checkpoint.** The actual native-Qwen compilation graph inspected by the quality audit shows BF16 dense QSA/GDN projections, hyperconnections, and PLE projections; its clearly observed FP8 storage is the PLE table and MoE experts. Those paths do not automatically pass through the ordinary FP8 linear chooser. First establish whether the checkpoint contains eligible dense layers and whether the baseline uses a different kernel for them. If it does not, this flag has no meaningful work to accelerate.

Read-only installed-source capture: `analysis/qwen_native_linear_sources_20260905.json`. No GPU process was imported, changed, or benchmarked for this audit.

Verified source facts:

- `model_executor/kernels/linear/scaled_mm/b12x.py`: `B12xFp8BlockScaledMMKernel` requires an installed/supported B12X backend on CUDA capability family 120, 128×128 block-FP8 weights, dynamic activation groups `(1,128)`, matching BF16/FP16 input/output, and positive partition dimensions divisible by 128.
- `model_executor/kernels/linear/__init__.py`: `b12x` maps to that class. The automatic FP8 block list tries CUTLASS before B12X after the DeepGEMM paths. Therefore the actual baseline selection log, not the requested flag alone, establishes whether an A/B differs.
- `model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py`: B12X inherits the usual `QuantFP8` activation path with `use_ue8m0=False`. CUTLASS block FP8 uses the same group key and non-UE8M0 scale semantics, with column-major scale storage. Switching those two preserves the specified FP8 quantization format; it does not guarantee bitwise-identical GEMM accumulation.
- B12X expands a UE8M0 weight-scale representation to FP32 when necessary; it does not convert FP8 weights to four-bit weights. This conditional branch is not evidence that this native checkpoint uses UE8M0 scales.
- Explicit backend filtering can reject an ineligible layer when a B12X implementation exists for its layer class but cannot implement its geometry. Generic fallback for an entirely unsupported layer class is a different case. Preserve the startup log and per-layer kernel selection.

Do not force the MoE backend to B12X for the native FP8 checkpoint. Keep native weights, BF16 main KV, FP32 GDN state, and no speculation in the first baseline. Any later eligible linear-only arm still needs the paired quality gate and a repeated complete serving measurement.
