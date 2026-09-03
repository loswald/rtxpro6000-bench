#!/usr/bin/env bash
# Restore the nightly build (has the b12x MoE backend) and re-derive the sm_120 o_proj patch from ITS source.
set -x
V=$(pip index versions vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130 2>/dev/null | head -1 | grep -oE "0\.[0-9]+\.[0-9]+rc[0-9]+\.dev[0-9]+\+g[a-f0-9]+")
echo "nightly target: $V"
uv pip install --system --prerelease=allow --index-strategy unsafe-best-match --torch-backend=cu130 \
  "vllm==$V" --extra-index-url https://wheels.vllm.ai/nightly/cu130
echo "vllm-rc=$?"
uv pip install --system "flashinfer-python==0.6.18"
uv pip install --system "flashinfer-jit-cache==0.6.18" --extra-index-url https://flashinfer.ai/whl/cu130
uv pip install --system b12x
set +x
python3 - <<PY
import importlib
for m in ("torch","vllm","flashinfer","b12x"):
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","(ok)"))
    except Exception as e: print(m, "FAIL", type(e).__name__, str(e)[:110])
from vllm.config.kernel import MoEBackend, LinearBackend
import typing
print("moe backends:", typing.get_args(MoEBackend))
print("b12x MoE available:", "b12x" in typing.get_args(MoEBackend))
PY
python3 /workspace/bench/patch_oproj.py
echo NIGHTLY-FIX-DONE
