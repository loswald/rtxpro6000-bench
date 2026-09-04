#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1 MAX_JOBS=6 NVCC_THREADS=2
for r in $(seq 0 0); do
  devs=$(seq -s, $((r*4)) $((r*4+4-1)))
  CUDA_VISIBLE_DEVICES=$devs vllm serve /workspace/models/Inkling-Small-NVFP4 --served-model-name m --host 0.0.0.0 --port $((8000+r)) \
    --tensor-parallel-size 4 --disable-custom-all-reduce --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192  --gpu-memory-utilization 0.94 --compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}  --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code --disable-uvicorn-access-log --kernel-config.linear_backend b12x --speculative-config {"method":"inkling_mtp","num_speculative_tokens":2} \
    > /workspace/results/smoke/f2_inkling_mtp_p$((8000+r)).log 2>&1 &
  sleep 2
done
wait
