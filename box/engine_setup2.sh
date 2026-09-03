#!/usr/bin/env bash
set -x
IDX="--extra-index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match --torch-backend cu130"
uv pip install --system --pre "vllm[b12x]" "flashinfer-python==0.6.18" "flashinfer-jit-cache==0.6.18" $IDX 2>&1 | tail -8
FV=$(python3 -c "import importlib.metadata as m; print(m.version(\"flashinfer-python\"))" 2>/dev/null)
uv pip install --system --no-deps "flashinfer-cubin==$FV" --index-url https://flashinfer.ai/whl/ 2>&1 | tail -2
uv pip list --system 2>/dev/null | grep -iE "^(flashinfer|b12x|vllm|torch) "
python3 - <<PY
import importlib
for m in ("vllm","flashinfer","b12x","torch","transformers","triton"):
    try: print(f"  {m:14s}", getattr(importlib.import_module(m),"__version__","?"))
    except Exception as e: print(f"  {m:14s} - ({type(e).__name__}: {str(e)[:80]})")
import torch; print("  torch.cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
PY
echo ENGINE-SETUP2-DONE
