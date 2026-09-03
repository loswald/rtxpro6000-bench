#!/usr/bin/env bash
export PYTHONPATH="/workspace/sglimg2/sgl-workspace/sglang/python:/workspace/sglimg2/sgl-workspace/transformers/src:/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages:/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages" LD_LIBRARY_PATH="/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages/torch/lib:/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages/nvidia/cublas/lib:/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages/nvidia/cudnn/lib:/workspace/sglimg2/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
export PYTHONHOME=/workspace/sglimg2/usr
export MAX_JOBS=6 NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
export SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0
# defaults to true and calls deep_gemm in the hyper-connection pre-norm: NameError on sm_120
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
# prompts up to this KV length take the dense one-shot prefill (patched in for sm_120 via
# dsa_sm120.py); upstream sets it to index_topk=2048, we cover the judge shape too
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=8192
# discriminator knobs: arbitrary env (space-separated KEY=VAL) into the server process

export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "/workspace/sglimg2/usr/bin/python3.12" -m sglang.launch_server \
  --model-path /workspace/models/GLM-5.3-Flash-NVFP4 --served-model-name glm53f \
  --host 0.0.0.0 --port 8000 --tp-size 4 \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bfloat16   \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mem-fraction-static 0.85 --context-length 40960 \
  --max-running-requests 256 --chunked-prefill-size 8192 \
  --disable-custom-all-reduce --trust-remote-code  
