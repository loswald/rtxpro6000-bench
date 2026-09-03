#!/usr/bin/env bash
# Upcast UE8M0 (E8M0) FP8 block scales to fp32 in the shared block-scaled linear
# base class, so CUTLASS and Triton stop crashing on DeepSeek-V3.1/V3.2/V4 style
# checkpoints (config.json: "scale_fmt": "ue8m0").
#
# Why: vllm/model_executor/layers/quantization/fp8.py:348 allocates
# weight_scale_inv as torch.float8_e8m0fnu when the quant config says ue8m0.
# process_fp8_weight_block_strategy() only rewrites e8m0 on ROCm-FNUZ, so on
# CUDA the e8m0 tensor reaches the kernels unchanged:
#   CUTLASS -> STD_TORCH_CHECK(b_scales.scalar_type()==Float) at
#              csrc/.../w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
#   Triton  -> KeyError: 'float8_e8m0fnu'
#   b12x    -> works, because scaled_mm/b12x_block.py already does this upcast
#
# The conversion is bit-exact: E8M0 encodes 2^(e-127); an fp32 with exponent
# field e and zero mantissa is the same number. Same helper vLLM already ships
# and already uses for b12x and DeepGEMM.
#
# Idempotent. Writes a .orig backup. Revert: restore the .orig, or run with
# VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel to fall back to b12x.
set -euo pipefail

PY="${PY:-python3}"
F="$("$PY" -c 'import vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel as m; print(m.__file__)')"
echo "target: $F"

if grep -q '_upcast_e8m0_to_fp32' "$F"; then
  echo "already patched; nothing to do"; exit 0
fi
cp -n "$F" "$F.orig"

"$PY" - "$F" <<'PYEOF'
import sys, io
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old_imp = """from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.utils import replace_parameter
"""
new_imp = """from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
"""
assert s.count(old_imp) == 1, "import anchor not found / not unique"
s = s.replace(old_imp, new_imp)

old_body = """        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)"""
new_body = """        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        # UE8M0 block scales (DeepSeek "scale_fmt": "ue8m0") reach the kernels
        # as float8_e8m0fnu. Every CUDA block-scaled kernel except b12x and
        # DeepGEMM assumes fp32: CUTLASS asserts b_scales.scalar_type()==Float
        # (scaled_mm_helper.hpp:17), Triton raises KeyError('float8_e8m0fnu').
        # 2^(e-127) == fp32{exponent=e, mantissa=0}, so this is bit-exact.
        # Skipped on ROCm-FNUZ, where process_fp8_weight_block_strategy
        # deliberately re-encodes the scale as e8m0 with exponent+1.
        if current_platform.is_cuda() and new_weight_scale.dtype in (
            torch.float8_e8m0fnu,
            torch.uint8,
        ):
            new_weight_scale = _upcast_e8m0_to_fp32(new_weight_scale).contiguous()

        replace_parameter(layer, params.WEIGHT, new_weight.data)"""
assert s.count(old_body) == 1, "body anchor not found / not unique"
s = s.replace(old_body, new_body)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched ok")
PYEOF

"$PY" -c 'import vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel as m; print("import ok:", m.__file__)'
echo "done. backup at $F.orig"
