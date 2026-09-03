#!/usr/bin/env bash
# launch_x4.sh <model_dir> <alias> [extra vllm args...]  -> 4 independent TP1 servers on ports 8000-8003 (one per GPU)
MODEL=$1; ALIAS=$2; shift 2
export VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
mkdir -p /workspace/results/smoke
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" --served-model-name "$ALIAS" --host 0.0.0.0 --port $((8000+i)) \
    --max-model-len 40960 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
    --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}" --no-enable-flashinfer-autotune \
    --enable-prefix-caching --trust-remote-code "$@" > /workspace/results/smoke/${ALIAS}_p$((8000+i)).log 2>&1 &
  sleep 2
done
wait
