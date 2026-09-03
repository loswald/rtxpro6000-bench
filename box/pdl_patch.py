#!/usr/bin/env python3
"""Env override for SGLang's Programmatic-Dependent-Launch gate.

`is_arch_support_pdl()` returns `major >= 9`, so PDL is on for sm_120 too. The DGX Spark
(sm_121) vLLM recipe found PDL corrupts the Triton kernels carrying the KDA recurrent
state; in SGLang the KDA gated RMSNorm (fla/layernorm_gated.py) launches with PDL under
this gate. SGLANG_DISABLE_PDL=1 forces it off.

usage: python3 pdl_patch.py /path/to/sglang/kernels/jit/utils/arch.py
"""
import sys, re, ast

p = sys.argv[1]
s = open(p).read()
if "SGLANG_DISABLE_PDL" in s:
    print("already patched")
    sys.exit(0)
m = re.search(r"(def is_arch_support_pdl\([^)]*\)[^:]*:\n)", s)
assert m, "def is_arch_support_pdl not found"
ind = "    "
guard = (ind + "import os as _os  # sm_120 discriminator\n"
         + ind + "if _os.environ.get(\"SGLANG_DISABLE_PDL\") == \"1\":\n"
         + ind + "    return False\n")
s = s[:m.end()] + guard + s[m.end():]
ast.parse(s)
open(p, "w").write(s)
print("patched; parses")
