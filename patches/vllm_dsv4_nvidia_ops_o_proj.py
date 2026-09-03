# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    # sqwish patch: SM12x (RTX PRO 6000) has no SM100-style packed scales in DeepGEMM;
    # use the SM90 layout there (VLLM_DSV4_OPROJ_SM120_RECIPE=sm100 to test the other).
    if cap.major == 12 and os.environ.get("VLLM_DSV4_OPROJ_SM120_RECIPE", "sm90") == "sm90":
        return (1, 128, 128), False
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    weight_scale = (
        wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    )
    if _SM120_FALLBACK:
        z = _sm120_o_proj_einsum(o_fp8, o_scale, wo_a, weight_scale)
    else:
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    return wo_b(z.flatten(1))


# ---- sqwish patch: bf16 fallback for SM12x when DeepGEMM fp8_einsum is unavailable ----
_SM120_FALLBACK = os.environ.get("VLLM_DSV4_OPROJ_SM120_FALLBACK", "0") == "1"
_LOGGED = False


def _block_dequant_weight(w: torch.Tensor, ws: torch.Tensor, h: int) -> torch.Tensor:
    # w: [h, d, r] or [h*d, r] fp8; ws: matching block scales (128x128), fp32 (ue8m0 values ok)
    if w.dim() == 2:
        w = w.view(h, -1, w.shape[-1])
    if ws.dim() == 2:
        ws = ws.view(h, -1, ws.shape[-1])
    hh, d, r = w.shape
    wf = w.to(torch.float32)
    scale = ws.to(torch.float32).repeat_interleave(128, dim=1)[:, :d].repeat_interleave(128, dim=2)[:, :, :r]
    return (wf * scale).to(torch.bfloat16)


def _sm120_o_proj_einsum(o_fp8, o_scale, wo_a, weight_scale):
    global _LOGGED
    b, h, r = o_fp8.shape
    if not _LOGGED:
        _LOGGED = True
        print(f"[sqwish sm120 o_proj fallback] o_fp8 {tuple(o_fp8.shape)} o_scale {tuple(o_scale.shape)} {o_scale.dtype} w {tuple(wo_a.weight.shape)} {wo_a.weight.dtype} ws {tuple(weight_scale.shape)} {weight_scale.dtype}", flush=True)
    os_ = o_scale.to(torch.float32)
    if os_.dim() == 3 and os_.shape[0] == b and os_.shape[1] == h:
        nblk = os_.shape[2]
    else:
        os_ = os_.reshape(b, h, -1)
        nblk = os_.shape[2]
    o_deq = (o_fp8.to(torch.float32).view(b, h, nblk, r // nblk) * os_.view(b, h, nblk, 1)).view(b, h, r)
    w_bf16 = getattr(wo_a, "_sqwish_w_bf16", None)
    if w_bf16 is None:
        w_bf16 = _block_dequant_weight(wo_a.weight, weight_scale, h)
        wo_a._sqwish_w_bf16 = w_bf16
    return torch.einsum("bhr,hdr->bhd", o_deq.to(torch.bfloat16), w_bf16)
