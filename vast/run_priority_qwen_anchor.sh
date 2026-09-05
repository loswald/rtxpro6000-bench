#!/usr/bin/env bash
set -euo pipefail
cd /workspace/priority
deadline=$((SECONDS + 1800))
while [ ! -f /workspace/models/Qwen3.8-Flash-Next-FP8/.complete ]; do
  if [ -f download_qwen.exit ] && [ "$(cat download_qwen.exit)" != 0 ]; then
    echo 'Model download failed; anchor not launched.' >&2
    exit 1
  fi
  if [ "$SECONDS" -gt "$deadline" ]; then echo 'Model download wait timed out.' >&2; exit 1; fi
  sleep 10
done
[ "$(cat setup.exit)" = 0 ]
[ "$(cat hardware.exit)" = 0 ]
python3 bench/capture_provenance.py --model-dir /workspace/models/Qwen3.8-Flash-Next-FP8 \
  --out results/qwen_native_identity.json
export RUN_TAG=stock_compile HEALTH_TIMEOUT=1200
bash bench/launch.sh qwen38flashnext_fp8_tp4_ep_anchor --no-prefetch --no-smoke
