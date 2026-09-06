#!/usr/bin/env bash
set -euo pipefail
cd /workspace/priority
vendor=/workspace/glmimg/usr/local/lib/python3.12/dist-packages/vllm
base=/workspace/glm-vendor-baseline
rel=v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
test -d "$vendor"
test ! -e "$base"
mkdir "$base"
cp -a "$vendor" "$base/vllm"
python3 vast/vllm_sm120_nope.py "$base/vllm"
printf '%s  %s\n' 7a19dafb16f1a2f9ac58992ce78e4d27b8f52edf08059c387d4f32d70d0edab3 "$base/vllm/$rel" | sha256sum -c -
PYTHONPATH="$base" python3 -c 'import vllm; print(vllm.__version__)'
python3 bench/capture_provenance.py --model-dir /workspace/models/GLM-5.3-Flash-NVFP4 \
  --out results/glm_model_identity.json
