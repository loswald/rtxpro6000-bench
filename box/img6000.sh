#!/usr/bin/env bash
cd /workspace/bench
for spec in "vllm/vllm-openai:glm53-flash-x86_64-cu130 /workspace/glmimg" "ghcr.io/motiftechnologies/vllm:v0.26.0-motif3-patch1 /workspace/motifimg" "vllm/vllm-openai:qwen38-flash-next /workspace/fnimg"; do
  set -- $spec
  echo "[$(date +%H:%M:%S)] pulling $1 -> $2"
  python3 pull_image.py "$1" "$2" > "/workspace/results/pull_$(basename $2).log" 2>&1 && echo "[$(date +%H:%M:%S)] done $2 ($(du -sh $2 2>/dev/null | cut -f1))" || echo "[$(date +%H:%M:%S)] FAILED $2 ($(tail -1 /workspace/results/pull_$(basename $2).log | cut -c1-120))"
done
echo IMG6000-DONE
