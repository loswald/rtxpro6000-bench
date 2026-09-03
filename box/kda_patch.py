#!/usr/bin/env python3
"""Add an env knob that forces the Triton causal-conv in SGLang's hybrid linear-attention
backend. The default CUDA causal_conv1d is an sgl_kernel op; on sm_120 that is a suspect
for the GLM-5.3-Flash decode drift (34 of 45 layers are KDA linear attention).

usage: python3 kda_patch.py /path/to/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
"""
import sys, ast

p = sys.argv[1]
s = open(p).read()
old = ("        use_triton_causal_conv = (\n"
       "            use_triton_causal_conv or get_memory().enable_page_major_kv_layout\n"
       "        )")
new = ("        use_triton_causal_conv = (\n"
       "            use_triton_causal_conv or get_memory().enable_page_major_kv_layout\n"
       "            or __import__(\"os\").environ.get(\"SGLANG_FORCE_TRITON_CONV\") == \"1\"  # sm_120 discriminator\n"
       "        )")
if "SGLANG_FORCE_TRITON_CONV" in s:
    print("already patched")
elif old in s:
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(p, "w").write(s)
    print("patched; parses")
else:
    print("ANCHOR MISSING")
    sys.exit(1)
