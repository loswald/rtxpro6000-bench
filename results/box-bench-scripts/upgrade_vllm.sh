#!/usr/bin/env bash
set -o pipefail
LOG=/workspace/results/upgrade2.log
{
echo "[$(date -Is)] START"
set -x
uv pip install --system --prerelease=allow --index-strategy unsafe-best-match --torch-backend=cu130 \
  "vllm==0.28.1rc1.dev312+g41848caa6" --extra-index-url https://wheels.vllm.ai/nightly/cu130
echo "vllm-rc=$?"
uv pip install --system --prerelease=allow "flashinfer-python==0.6.18" || uv pip install --system --prerelease=allow -U flashinfer-python
echo "flashinfer-rc=$?"
uv pip install --system --prerelease=allow "vllm[b12x]==0.28.1rc1.dev312+g41848caa6" --extra-index-url https://wheels.vllm.ai/nightly/cu130 --torch-backend=cu130 --index-strategy unsafe-best-match || uv pip install --system b12x
echo "b12x-rc=$?"
set +x
python3 - <<PY
import importlib
for m in ("vllm","torch","flashinfer","b12x","triton","transformers"):
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m, "IMPORT-FAIL", type(e).__name__, str(e)[:120])
import torch; print("cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list()[-3:])
PY
python3 -c "import vllm.platforms.cuda as c; print(\"vllm cuda platform ok\")" 2>&1 | tail -1
echo "[$(date -Is)] UPGRADE-DONE"
} 2>&1 | tee "$LOG"
