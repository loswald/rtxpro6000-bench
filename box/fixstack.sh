#!/usr/bin/env bash
set -x
uv pip install --system --prerelease=allow --index-strategy unsafe-best-match --torch-backend=cu130 \
  "vllm==0.28.1rc1.dev312+g41848caa6" --extra-index-url https://wheels.vllm.ai/nightly/cu130
uv pip install --system "flashinfer-python==0.6.18"
uv pip install --system "flashinfer-jit-cache==0.6.18" --extra-index-url https://flashinfer.ai/whl/cu130
uv pip uninstall --system flashinfer-cubin
uv pip install --system b12x
set +x
python3 - <<PY
import importlib
for m in ("torch","vllm","flashinfer","b12x"):
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","(ok)"))
    except Exception as e: print(m, "FAIL", type(e).__name__, str(e)[:110])
import torch; print("cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
PY
# apply the sm_120 DeepSeek o_proj patch
T=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)/models/deepseek_v4/nvidia/ops/o_proj.py
if [ -f "$T" ] && [ -f /workspace/vllm_dsv4_nvidia_ops_o_proj.py ]; then
  [ -f "$T.orig" ] || cp "$T" "$T.orig"
  cp /workspace/vllm_dsv4_nvidia_ops_o_proj.py "$T"
  python3 -c "import vllm.models.deepseek_v4.nvidia.ops.o_proj as m; print(o_proj patch applied, fallback flag =, m._SM120_FALLBACK)" 2>&1 | tail -1
else
  echo "o_proj target not found at $T"
fi
echo FIXSTACK-DONE
