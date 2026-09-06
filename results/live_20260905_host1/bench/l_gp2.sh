#!/usr/bin/env bash
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null
export PYTHONPATH=/workspace/glmvllm
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/GLM-5.3-Flash-NVFP4 --served-model-name m --host 0.0.0.0 --port 8000 \
  --attention-backend FLASHINFER_MLA_SPARSE_SM90 \
  --kv-cache-dtype auto --block-size 1024 --max-model-len 40960 --max-num-seqs 512 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune --disable-uvicorn-access-log --tensor-parallel-size 4 --moe-backend cutlass
