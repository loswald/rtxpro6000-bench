#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /workspace/models/Ornith-1.5-397B-NVFP4 --served-model-name m --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce --no-enable-flashinfer-autotune --disable-uvicorn-access-log --limit-mm-per-prompt \{\"image\":0\} --gpu-memory-utilization 0.90 > /workspace/results/smoke/ornith_p8000.log 2>&1 &
sleep 2
wait
