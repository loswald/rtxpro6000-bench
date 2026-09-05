#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (Qwen4Exp) on sm_120: make vLLM load RadixArk's NVFP4 checkpoint.

The checkpoint stores the per-layer n-gram embedding tables ("PLE") as FP8 shards with one global BF16 scale
(the model-plefp8-*.safetensors files), but its quantization_config is modelopt NVFP4 with "*.ple.*" in the
ignore list. vLLM's Qwen4Exp module therefore builds the PLE as unquantised BF16 and dies on the checkpoint's
extra `ngram_embedding.weight_scale` ("There is no module or parameter named ..."). The build already has the
FP8 PLE path (Qwen4ExpPLEFp8EmbeddingMethod) for Qwen's own FP8 release; this patch lets an environment
variable, VLLM_QWEN4EXP_PLE_FP8=1, select it regardless of the quantization config. Idempotent; keeps a backup.
"""
import os, shutil, sys
try:
    import vllm
except ImportError:
    sys.exit("vllm not importable")
p = os.path.join(os.path.dirname(vllm.__file__), "models", "qwen4_exp", "nvidia", "ple_layer.py")
s = open(p, encoding="utf-8").read()
if "VLLM_QWEN4EXP_PLE_FP8" in s:
    print("already patched:", p); sys.exit(0)
anchor = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n'''
if s.count(anchor) != 1:
    sys.exit("anchor not found in " + p)
s = s.replace(anchor, anchor + '''
    # sm_120 campaign: RadixArk's NVFP4 checkpoint ships FP8 PLE shards under a modelopt NVFP4 config that ignores
    # "*.ple.*"; force the FP8 PLE method when asked, whatever the quantization config says.
    import os as _os
    if _os.environ.get("VLLM_QWEN4EXP_PLE_FP8") == "1":
        return Qwen4ExpPLEFp8EmbeddingMethod()
''')
shutil.copyfile(p, p + ".orig")
open(p, "w", encoding="utf-8").write(s)
print("patched:", p)
