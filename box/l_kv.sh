#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
  --host 0.0.0.0 --port 8000 --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
  --trust-remote-code --disable-uvicorn-access-log > /workspace/results/smoke/kv_fp8.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
  --host 0.0.0.0 --port 8001 --kv-cache-dtype auto --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
  --trust-remote-code --disable-uvicorn-access-log > /workspace/results/smoke/kv_bf16.log 2>&1 &
wait
