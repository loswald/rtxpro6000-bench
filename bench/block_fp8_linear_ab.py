#!/usr/bin/env python3
"""A/B every FP8 block-scaled linear kernel vLLM can build on this GPU.

Answers "is b12x actually the fastest, or merely the only one that works?" by
constructing each Fp8BlockScaledMMLinearKernel directly (no auto-selection, no
server) and timing apply_weights() over DeepSeek/Qwen-shaped GEMMs.

Runs on an UNPATCHED vLLM 0.28.0: with fp32 weight scales (the Qwen FP8 case)
CutlassFp8BlockScaledMMKernel already works on sm_120, so the comparison against
B12xFp8BlockScaledMMKernel is available today.

    # fp32 scales - works unpatched, this is the real speed comparison
    python3 bench/block_fp8_linear_ab.py

    # UE8M0 scales - the DeepSeek "scale_fmt": "ue8m0" case.
    # Before patches/apply_ue8m0_block_scale_upcast.sh only b12x survives;
    # after it, cutlass and triton should produce identical numerics.
    python3 bench/block_fp8_linear_ab.py --scale-dtype e8m0

    python3 bench/block_fp8_linear_ab.py --shapes 7168x2048,4096x7168 --m 1,64,256,1024

Every kernel is checked against the same bf16 reference, so a kernel that is
fast and wrong is reported as wrong.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import torch

from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
)

# (N, K) = (out_features, in_features). DeepSeek-V3/V4 + Qwen3 dense shapes.
# All dims are multiples of 128: b12x rejects anything else, and a padded
# weight would make the correctness check compare mismatched shapes.
DEFAULT_SHAPES = [
    (24576, 1536),
    (12288, 7168),
    (4096, 7168),
    (7168, 2048),
    (7168, 16384),
    (18432, 7168),
]
DEFAULT_M = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
BLOCK = 128


def candidate_kernels() -> list[tuple[str, type]]:
    """Every block-scaled FP8 linear kernel class, by --kernel-config name."""
    out: list[tuple[str, type]] = []

    def add(backend: str, module: str, cls_name: str) -> None:
        import importlib

        for mod in (module, module.replace("b12x_block", "b12x")):
            try:
                m = importlib.import_module(mod)
            except ImportError:
                continue
            cls = getattr(m, cls_name, None)
            if cls is not None:
                out.append((backend, cls))
                return

    base = "vllm.model_executor.kernels.linear.scaled_mm."
    add("cutlass", base + "cutlass", "CutlassFp8BlockScaledMMKernel")
    # 0.28.0 ships this as b12x_block.py; main renamed it to b12x.py.
    add("b12x", base + "b12x_block", "B12xFp8BlockScaledMMKernel")
    add("triton", base + "triton", "TritonFp8BlockScaledMMKernel")
    add("deep_gemm", base + "deep_gemm", "DeepGemmFp8BlockScaledMMKernel")
    add("flashinfer_cutlass", base + "flashinfer",
        "FlashInferFp8DeepGEMMDynamicBlockScaledKernel")
    add("torch", base + "pytorch", "BlockWiseTorchFP8ScaledMMLinearKernel")
    return out


def make_layer(N: int, K: int, scale_dtype: str, device: str) -> torch.nn.Module:
    finfo = torch.finfo(torch.float8_e4m3fn)
    w_bf16 = (torch.rand(N, K, device=device, dtype=torch.bfloat16) - 0.5) * 2 * finfo.max
    weight = w_bf16.clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)

    n_tiles, k_tiles = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
    if scale_dtype == "e8m0":
        # E8M0 encodes 2^(e-127); byte 127 == 1.0. Small spread around 2^-6.
        exps = torch.randint(115, 128, (n_tiles, k_tiles), device=device,
                             dtype=torch.uint8)
        scale = exps.view(torch.float8_e8m0fnu)
    else:
        scale = torch.rand(n_tiles, k_tiles, device=device,
                           dtype=torch.float32) * 1e-2

    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
    )
    layer.input_scale = None
    layer.input_scale_ub = None
    return layer


def dequant_ref(layer: torch.nn.Module, N: int, K: int) -> torch.Tensor:
    """bf16 reference weight, for the correctness check."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _upcast_e8m0_to_fp32,
    )

    s = layer.weight_scale_inv.data
    if s.dtype in (torch.float8_e8m0fnu, torch.uint8):
        s = _upcast_e8m0_to_fp32(s)
    s = s.float()
    n_tiles, k_tiles = s.shape
    full = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:N, :K]
    return (layer.weight.data.to(torch.float32) * full).to(torch.bfloat16)


def timeit(fn, warmup: int = 12, iters: int = 60) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@default_vllm_config()
def run(args) -> None:
    device = "cuda"
    torch.set_default_dtype(torch.bfloat16)

    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
    print(f"weight-scale dtype: {args.scale_dtype}\n")

    cfg_key_w = create_fp8_quant_key(static=True, group_shape=GroupShape(BLOCK, BLOCK))
    cfg_key_a = create_fp8_quant_key(static=False, group_shape=GroupShape(1, BLOCK))

    for N, K in args.shapes:
        cfg = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=cfg_key_w,
            activation_quant_key=cfg_key_a,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(N, K),
        )

        print(f"\n=== N={N} K={K} ===")
        usable: list[tuple[str, str, object, torch.nn.Module]] = []
        for backend, cls in candidate_kernels():
            if not issubclass(cls, Fp8BlockScaledMMLinearKernel):
                continue
            ok, why = cls.is_supported(cap[0] * 10 + cap[1])
            if not ok:
                print(f"  [skip] {cls.__name__:<42} is_supported: {why}")
                continue
            ok, why = cls.can_implement(cfg)
            if not ok:
                print(f"  [skip] {cls.__name__:<42} can_implement: {why}")
                continue
            layer = make_layer(N, K, args.scale_dtype, device)
            ref_w = dequant_ref(layer, N, K)
            try:
                kern = cls(cfg)
                kern.process_weights_after_loading(layer)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {cls.__name__:<42} process_weights: "
                      f"{type(exc).__name__}: {exc}")
                continue
            usable.append((backend, cls.__name__, kern, layer))
            layer._ref_w = ref_w

        hdr = f"{'M':>6} " + "".join(f"{b:>26}" for b, _, _, _ in usable)
        print(hdr)

        for M in args.m:
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.2
            cells = []
            for _, cls_name, kern, layer in usable:
                try:
                    out = kern.apply_weights(layer, x)
                    ref = torch.nn.functional.linear(x, layer._ref_w)
                    denom = ref.abs().mean().clamp_min(1e-6)
                    err = ((out.float() - ref.float()).abs().mean() / denom).item()
                    ms = timeit(lambda k=kern, l=layer: k.apply_weights(l, x))
                    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
                    flag = "" if err < args.rtol else f" BAD(rel={err:.3f})"
                    cells.append(f"{tflops:>18.1f} TF/s{flag}"[:26].rjust(26))
                except Exception as exc:  # noqa: BLE001
                    cells.append(f"{type(exc).__name__[:22]:>26}")
            print(f"{M:>6} " + "".join(cells))
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale-dtype", choices=["fp32", "e8m0"], default="fp32",
                   help="weight-scale format; e8m0 == DeepSeek 'scale_fmt: ue8m0'")
    p.add_argument("--shapes", default=None,
                   help="comma-separated NxK list, e.g. 7168x2048,4096x7168")
    p.add_argument("--m", default=None,
                   help="comma-separated token counts, e.g. 1,64,256,4096")
    p.add_argument("--rtol", type=float, default=0.05,
                   help="mean relative error above which a kernel is flagged BAD")
    args = p.parse_args()

    args.shapes = (
        [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
        if args.shapes else DEFAULT_SHAPES
    )
    args.m = [int(v) for v in args.m.split(",")] if args.m else DEFAULT_M
    run(args)


if __name__ == "__main__":
    main()
