#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/Qwen3.8-27B-FP8 qwen27b \
  --kv-cache-dtype fp8 --kernel-config.linear_backend b12x
