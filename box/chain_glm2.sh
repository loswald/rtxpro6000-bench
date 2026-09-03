#!/usr/bin/env bash
# fires once the clean re-pull lands: fix-up, take the GPUs, launch GLM, hand back to NVFP4
while ! grep -q EXTRACTED /workspace/results/sglimg3.log 2>/dev/null; do sleep 20; done
bash /workspace/bench/sglfix.sh /workspace/sglimg2 > /workspace/results/sglfix.log 2>&1
for s in chainnv nv; do tmux kill-session -t "=$s" 2>/dev/null; done
bash /workspace/bench/cleanup.sh > /dev/null 2>&1
IMG=/workspace/sglimg2 DSA_PREFILL=tilelang DSA_DECODE=tilelang KV_DTYPE=bfloat16 \
  bash /workspace/bench/glm_sgl.sh > /workspace/results/glm_sgl.log 2>&1
sleep 20; bash /workspace/bench/nvtier.sh > /workspace/results/nvtier.log 2>&1
