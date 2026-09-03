#!/usr/bin/env bash
EXTRA_ARGS="--block-size 1024" QMAX=2048 ARMS=base,mtp bash /workspace/bench/glm_vllm.sh > /workspace/results/glm_vllm_full.log 2>&1
bash /workspace/bench/cleanup.sh >/dev/null 2>&1
bash /workspace/bench/nvtier2.sh >> /workspace/results/nvtier2.log 2>&1
