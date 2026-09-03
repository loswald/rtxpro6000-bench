#!/usr/bin/env bash
# CRITICAL: VLLM_MOE_USE_DEEP_GEMM is a SEPARATE switch from VLLM_USE_DEEP_GEMM and defaulted to 1
# on every run we did before this one. DeepGEMM has no sm_120 MoE kernels.
export VLLM_MOE_USE_DEEP_GEMM=0
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss 
