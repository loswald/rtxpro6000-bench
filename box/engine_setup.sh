#!/usr/bin/env bash
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
IDX="--extra-index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match --torch-backend cu130"
# the first PRO 6000 box ran 0.28.1rc1.dev332+gad127d9a0; the 5090 box got dev353. Prefer dev353 so both
# Blackwell boxes run the same build for the like-for-like comparison.
uv pip install --system --pre "vllm[b12x]==0.28.1rc1.dev353+gbf95f58d1" "flashinfer-python==0.6.18" "flashinfer-jit-cache==0.6.18" $IDX 2>&1 | tail -6 \
  || uv pip install --system --pre "vllm[b12x]" "flashinfer-python==0.6.18" "flashinfer-jit-cache==0.6.18" $IDX 2>&1 | tail -6
python3 - <<PY
import importlib
for m in ("vllm","flashinfer","b12x","torch","transformers","triton"):
    try: print(f"  {m:14s}", getattr(importlib.import_module(m),"__version__","?"))
    except Exception as e: print(f"  {m:14s} - ({type(e).__name__}: {str(e)[:80]})")
import torch; print("  torch.cuda", torch.version.cuda, torch.cuda.get_arch_list()[-3:])
PY
uv pip list --system 2>/dev/null | grep -iE "^flashinfer"
echo ENGINE-SETUP-DONE
