#!/usr/bin/env bash
set -x
V=/workspace/venv-sgl2
uv venv --python 3.12 "$V"
timeout 1800 uv pip install --python "$V/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
  sglang --extra-index-url https://docs.sglang.ai/whl/cu130/ --extra-index-url https://download.pytorch.org/whl/cu130
echo "rc=$?"
"$V/bin/python" -c "import sglang, torch; print(\"sglang\", sglang.__version__, \"torch\", torch.__version__, torch.version.cuda)"
echo SGLANG-INSTALL-DONE
