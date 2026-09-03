#!/usr/bin/env bash
source /workspace/venv-sgl2/bin/activate
export SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS
export SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS=1024
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_OPT_USE_TILELANG_INDEXER=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
export SGLANG_RAGGED_VERIFY_MODE=compact
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
exec python -m sglang.launch_server \
  --model-path /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4 \
  --host 0.0.0.0 --port 8000 \
  --tp-size 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head --ep-size 4 \
  --mem-fraction-static 0.85 --context-length 40960 \
  --max-running-requests 256 --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 4096 \
  --disable-custom-all-reduce \
  --trust-remote-code
