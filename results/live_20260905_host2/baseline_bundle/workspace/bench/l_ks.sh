#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=4 NVCC_THREADS=2
export VLLM_QWEN4EXP_PLE_FP8=1
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /workspace/models/Qwen3.8-Flash-Next-NVFP4 --served-model-name m --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.94 --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce --no-enable-flashinfer-autotune --disable-uvicorn-access-log --kv-cache-dtype auto --max-num-seqs 512 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice > /workspace/results/smoke/qwen38fn_tp4_---_p8000.log 2>&1 &
sleep 2
wait
