#!/usr/bin/env python3
"""Port Z.AI's vLLM Hopper NoPE sparse-MLA backend to sm_120 (the DGX Spark community's fix).

Why: the vendor build's only capability-12 sparse backend is hard-wired to DeepSeek's packed
fp8 layout (pe_dim must be 64) and dies for GLM-5.3-Flash (qk_rope_head_dim=0). Its SM90
backend supports rope 0 by construction (via FlashInfer >= 0.6.18) but is gated to
major == 9 and builds its wrapper with FlashAttention-3. On sm_120 FlashInfer's FA2 MLA
path is what runs. Three edits:
  1. flashinfer_mla_sparse_sm90.py : accept capability major 12
  2. flashinfer_mla_sparse_sm90.py : backend="fa2" on major 12 (fa3 elsewhere)
  3. platforms/cuda.py             : put FLASHINFER_MLA_SPARSE_SM90 first in the sm_120 list

usage: python3 vllm_sm120_nope.py /path/to/dist-packages/vllm
"""
import sys, os, ast

V = sys.argv[1]
S = os.path.join(V, "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py")
C = os.path.join(V, "platforms/cuda.py")


def edit(path, pairs):
    s = open(path).read()
    if "sm_120 port" in s:
        print(f"  {os.path.relpath(path, V)}: already patched")
        return
    for old, new in pairs:
        assert s.count(old) == 1, f"{os.path.relpath(path, V)}: anchor count {s.count(old)} for {old[:50]!r}"
        s = s.replace(old, new, 1)
    ast.parse(s)
    open(path, "w").write(s)
    print(f"  {os.path.relpath(path, V)}: patched, parses")


edit(S, [
    ("    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
     "        return capability.major == 9\n",
     "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
     "        return capability.major in (9, 12)  # sm_120 port: FA2 path below\n"),
    ("            use_cuda_graph=True,\n"
     "            backend=\"fa3\",\n"
     "        )\n",
     "            use_cuda_graph=True,\n"
     "            backend=(\"fa2\" if torch.cuda.get_device_capability(device)[0] == 12 else \"fa3\"),\n"
     "        )\n"),
])
edit(C, [
    ("        elif device_capability.major == 12:\n"
     "            return [\n"
     "                AttentionBackendEnum.TRITON_MLA,\n"
     "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n"
     "            ]\n",
     "        elif device_capability.major == 12:\n"
     "            return [  # sm_120 port: NoPE sparse MLA via the SM90 backend's FA2 path\n"
     "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,\n"
     "                AttentionBackendEnum.TRITON_MLA,\n"
     "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n"
     "            ]\n"),
])
print("vllm sm_120 NoPE port applied")
