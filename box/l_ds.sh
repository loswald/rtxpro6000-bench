#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4 --host 0.0.0.0 --port 8000 \
  --tokenizer-mode deepseek_v4 --block-size 256 \
  --attention_config.use_fp4_indexer_cache False \
  --kv-cache-dtype fp8 --max-model-len 40960 \
  --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.96 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x \
  --disable-custom-all-reduce --enable-prefix-caching --trust-remote-code \
  --disable-uvicorn-access-log --tensor-parallel-size 4 --enable-expert-parallel --moe-backend auto
