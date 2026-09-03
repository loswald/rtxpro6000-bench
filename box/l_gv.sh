#!/usr/bin/env bash
export PYTHONPATH=/workspace/glmvllm
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/GLM-5.3-Flash-NVFP4 --served-model-name glm53f --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --attention-backend FLASHINFER_MLA_SPARSE_SM90 \
  --kv-cache-dtype auto --max-model-len 40960 --max-num-seqs 256 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --reasoning-parser deepseek_r1 --tool-call-parser glm47 \
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune \
  --disable-uvicorn-access-log --block-size 1024 --speculative-config {"method":"glm5_next_mtp","num_speculative_tokens":3}
