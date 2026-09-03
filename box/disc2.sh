#!/usr/bin/env bash
# Round two of decode-drift discriminators for GLM-5.3-Flash on sm_120. Round one cleared
# CUDA graphs, MTP and (by rejection) the CUTE-DSL MoE runner. Each of these swaps exactly
# one decode-time component for a reference or alternative path.
set -u
B=/workspace/bench; R=/workspace/results; LOG=$R/disc2.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ # name  env-assignments...
  local name="$1"; shift
  for s in glmrun srv; do tmux kill-session -t "=$s" 2>/dev/null; done
  bash $B/cleanup.sh >/dev/null 2>&1
  : > $R/glm_sgl.log
  log "=== $name :: $*"
  env IMG=/workspace/sglimg2 DSA_PREFILL=tilelang DSA_DECODE=tilelang KV_DTYPE=bfloat16 \
      DENSE_THR=8192 QUALITY_ONLY=1 QMODE=chat QMAX=1024 ARMS=base "$@" \
      bash $B/glm_sgl.sh > $R/glm_sgl.log 2>&1
  local v=$(grep -E "quality20\[" $R/glm_sgl.log | tail -1)
  local h=$(grep -oE "healthy in [0-9]+s|FAILED after [0-9]+s" $R/glm_sgl.log | head -1)
  log "    $name -> ${h:-?} | ${v:-no verdict}"
  cp $R/probe/sglglm_base_quality20.json "$R/probe/disc2_${name}_quality20.json" 2>/dev/null
  grep -iE "error|Traceback|Exception" $R/smoke/sglglm_base.log | grep -vE "Ignore import" | tail -2 | cut -c1-160 | sed 's/^/      /' | tee -a "$LOG"
}
# FIRST: the GB10-validated TileLang tiles (block_I=32, num_stages=1, threads=128) with the
# dense-prefill patch; my earlier block_I=64 single-stage tiles are the prime suspect
run tiles_gb10_dense
# PDL off: SGLang enables Programmatic Dependent Launch for major>=9, i.e. also sm_120; the
# DGX Spark vLLM recipe found PDL corrupts the Triton kernels carrying KDA recurrent state
run pdl_off         EXTRA_ENV="SGLANG_DISABLE_PDL=1"
# the indexer's fp8 paged-MQA logits via the torch reference instead of the TileLang kernel
run indexer_torch   EXTRA_ENV="SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1"
# sparse prefill (gate closed, upstream sm_120 behaviour) as a control, then with the torch
# indexer: if the first tokens become right, the indexer was the broken piece; if they stay
# wrong, it is the TileLang sparse attention kernel itself
run sparse_prefill           DENSE_THR=0
run sparse_prefill_idx_torch DENSE_THR=0 EXTRA_ENV="SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1"
# unfused top-k selection
run topk_unfused    EXTRA_ENV="SGLANG_DSA_FUSE_TOPK=0"
# shared-experts fusion ON (the LibertAI recipe turned it off; a different MoE code path)
run shared_fused    SHARED_FUSION_FLAG=" "
# attention data-parallel instead of TP-sharded attention (different KDA/DSA sharding path)
run dp_attention    EXTRA_ARGS="--enable-dp-attention --dp-size 4"
# KDA side: fp32 recurrent state, and the Triton causal-conv instead of the CUDA sgl_kernel op
run ssm_fp32        EXTRA_ARGS="--mamba-ssm-dtype float32"
run conv_triton     EXTRA_ENV="SGLANG_FORCE_TRITON_CONV=1"
log "DISC2 DONE"
