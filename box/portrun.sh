#!/usr/bin/env bash
EXTRA_ARGS="--block-size 64" QUALITY_ONLY=1 ARMS=base bash /workspace/bench/glm_vllm.sh > /workspace/results/glm_vllm.log 2>&1
bash /workspace/bench/cleanup.sh >/dev/null 2>&1
bash /workspace/bench/nvtier2.sh > /workspace/results/nvtier2.log 2>&1
