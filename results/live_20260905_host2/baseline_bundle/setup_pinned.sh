#!/usr/bin/env bash
# Reproduce the package versions observed on host2 on 2026-09-05.
# Run only on the newly provisioned experiment host. Python 3.12, Linux x86_64.
set -euo pipefail
if [ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]; then
  echo "This captured baseline requires Python 3.12" >&2
  exit 2
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq tmux git jq rsync pciutils curl
command -v uv >/dev/null || python3 -m pip install uv
mkdir -p /workspace/bench /workspace/results /workspace/models
ulimit -n 65536

VLLM_WHEEL='https://wheels.vllm.ai/7985444339e2ad7e249b88a50081e16e34637dfd/vllm-0.28.1rc1.dev446%2Bg798544433-cp38-abi3-manylinux_2_28_x86_64.whl'
uv pip install --system --pre --index-strategy unsafe-best-match --torch-backend cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  "vllm[b12x] @ ${VLLM_WHEEL}" \
  'torch==2.13.0+cu130' 'transformers==5.15.1' 'triton==3.7.1' \
  'b12x==1.3.0' 'compressed-tensors==0.17.0' \
  'flashinfer-python==0.6.18' 'flashinfer-jit-cache==0.6.18+cu130' \
  'huggingface-hub==1.28.0' hf_transfer
# The cubin package is served by the root FlashInfer index, not the cu130 index.
uv pip install --system --no-deps 'flashinfer-cubin==0.6.18' \
  --index-url https://flashinfer.ai/whl/
python3 - <<'PY'
import importlib.metadata as m, json
names = ['vllm', 'torch', 'transformers', 'triton', 'b12x',
         'compressed-tensors', 'flashinfer-python', 'flashinfer-jit-cache',
         'flashinfer-cubin', 'huggingface-hub']
print(json.dumps({name: m.version(name) for name in names}, indent=2))
PY
# Captured patches remain separate for inspection. Native Qwen FP8 selects its
# PLE FP8 loader from the checkpoint config and does not need the NVFP4 override.
