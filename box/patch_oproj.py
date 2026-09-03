#!/usr/bin/env python3
"""Apply the sm_120 o_proj fix to WHATEVER vllm version is installed, deriving from its own source.
Safer than copying a patched file from another version."""
import os, sys, importlib, shutil
import vllm
T = os.path.join(os.path.dirname(vllm.__file__), "models", "deepseek_v4", "nvidia", "ops", "o_proj.py")
if not os.path.exists(T):
    print("no deepseek_v4 o_proj at", T); sys.exit(1)
orig = T + ".orig"
if not os.path.exists(orig):
    shutil.copy(T, orig)
s = open(orig, encoding="utf-8").read()

old_recipe = "    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)\n    tma_aligned_scales = cap.major >= 10\n"
if old_recipe not in s:
    print("WARNING: recipe block not found; upstream source changed. Leaving unpatched.")
    print("---- current compute_fp8_einsum_recipe ----")
    i = s.find("def compute_fp8_einsum_recipe")
    print(s[i:i+900] if i >= 0 else "(function not found)")
    sys.exit(2)
new_recipe = ("    # sqwish sm_120: DeepGEMM has no SM100-style packed-scale path on 12.x.\n"
              "    if cap.major == 12 and os.environ.get('VLLM_DSV4_OPROJ_SM120_RECIPE', 'sm90') == 'sm90':\n"
              "        return (1, 128, 128), False\n" + old_recipe)
s = s.replace(old_recipe, new_recipe)

old_call = ('    fp8_einsum(\n        "bhr,hdr->bhd",\n        (o_fp8, o_scale),\n        (wo_a.weight, weight_scale),\n        z,\n        recipe=einsum_recipe,\n    )\n')
if old_call not in s:
    print("WARNING: fp8_einsum call site not found; leaving recipe-only patch in place.")
else:
    new_call = ("    if _SM120_FALLBACK:\n        z = _sm120_o_proj_einsum(o_fp8, o_scale, wo_a, weight_scale)\n    else:\n"
                '        fp8_einsum(\n            "bhr,hdr->bhd",\n            (o_fp8, o_scale),\n            (wo_a.weight, weight_scale),\n            z,\n            recipe=einsum_recipe,\n        )\n')
    s = s.replace(old_call, new_call)
    s += '''

# ---- sqwish sm_120 bf16 fallback for DeepGEMM fp8_einsum ----
_SM120_FALLBACK = os.environ.get("VLLM_DSV4_OPROJ_SM120_FALLBACK", "0") == "1"


def _block_dequant_weight(w, ws, h):
    if w.dim() == 2:
        w = w.view(h, -1, w.shape[-1])
    if ws.dim() == 2:
        ws = ws.view(h, -1, ws.shape[-1])
    _, d, r = w.shape
    scale = ws.to(torch.float32).repeat_interleave(128, dim=1)[:, :d].repeat_interleave(128, dim=2)[:, :, :r]
    return (w.to(torch.float32) * scale).to(torch.bfloat16)


def _sm120_o_proj_einsum(o_fp8, o_scale, wo_a, weight_scale):
    b, h, r = o_fp8.shape
    os_ = o_scale.to(torch.float32)
    if not (os_.dim() == 3 and os_.shape[0] == b and os_.shape[1] == h):
        os_ = os_.reshape(b, h, -1)
    nblk = os_.shape[2]
    o_deq = (o_fp8.to(torch.float32).view(b, h, nblk, r // nblk) * os_.view(b, h, nblk, 1)).view(b, h, r)
    w_bf16 = getattr(wo_a, "_sqwish_w_bf16", None)
    if w_bf16 is None:
        w_bf16 = _block_dequant_weight(wo_a.weight, weight_scale, h)
        wo_a._sqwish_w_bf16 = w_bf16
    return torch.einsum("bhr,hdr->bhd", o_deq.to(torch.bfloat16), w_bf16)
'''
if "import os" not in s.split("\n\n")[0]:
    s = s.replace("import torch\nimport torch.nn as nn\n", "import os\n\nimport torch\nimport torch.nn as nn\n", 1)
open(T, "w", encoding="utf-8").write(s)
importlib.invalidate_caches()
m = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.o_proj")
print("patched OK:", T)
print("  fallback flag currently:", getattr(m, "_SM120_FALLBACK", "MISSING"))
print("  vllm version:", vllm.__version__)
