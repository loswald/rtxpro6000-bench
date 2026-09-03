#!/usr/bin/env bash
# wait for the engine install to finish, then align flashinfer-cubin with flashinfer-python
for i in $(seq 1 90); do grep -q ENGINE-SETUP-DONE /workspace/results/engine_setup.log 2>/dev/null && break; sleep 20; done
FV=$(python3 -c "import importlib.metadata as m; print(m.version(\"flashinfer-python\"))" 2>/dev/null)
echo "flashinfer-python $FV"
uv pip install --system --no-deps "flashinfer-cubin==$FV" --index-url https://flashinfer.ai/whl/ 2>&1 | tail -2
uv pip list --system 2>/dev/null | grep -iE "^flashinfer"
python3 -c "
import importlib
for m in (\"vllm\",\"flashinfer\",\"b12x\",\"torch\"):
    try: print(\"  \", m, getattr(importlib.import_module(m),\"__version__\",\"?\"))
    except Exception as e: print(\"  \", m, \"FAIL\", type(e).__name__, str(e)[:120])
" 2>&1 | grep -v Warning
echo CUBIN-FIX-DONE
