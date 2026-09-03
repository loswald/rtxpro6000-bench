#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
    --host 0.0.0.0 --port $((8000+i)) \
    --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.96 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code \
    --disable-uvicorn-access-log --moe-backend flashinfer_cutedsl --quantization-config.moe.activation mxfp8 \
    > /workspace/results/smoke/sw_gptoss_ficutedsl_p$((8000+i)).log 2>&1 &
  sleep 2
done
wait
