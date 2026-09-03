#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1
serve_one () {  # gpu port kvdtype
  CUDA_VISIBLE_DEVICES=$1 vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
    --host 0.0.0.0 --port $2 --kv-cache-dtype $3 --max-model-len 8192 --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
    --trust-remote-code --disable-uvicorn-access-log \
    > /workspace/results/smoke/qg_$2.log 2>&1 &
}
serve_one 0 8000 fp8
sleep 2
serve_one 1 8001 auto
sleep 2
serve_one 2 8002 fp8
wait
