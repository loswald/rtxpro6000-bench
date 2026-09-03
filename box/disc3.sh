#!/usr/bin/env bash
# Round two, reordered remainder. Cleared so far: CUDA graphs, MTP, MoE runner, TileLang
# tiles, PDL, indexer logits. Prefill is exact; decode drifts. What differs between prefill
# and decode besides the sparse kernel is the KDA path: chunk kernels vs the recurrent
# update, and causal_conv1d_fn vs the CUDA causal_conv1d_update, which SGLang's own comment
# says "garbles" strided state. So those go first.
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
run conv_triton     EXTRA_ENV="SGLANG_FORCE_TRITON_CONV=1"
run ssm_fp32        EXTRA_ARGS="--mamba-ssm-dtype float32"
run conv_triton_fp32 EXTRA_ENV="SGLANG_FORCE_TRITON_CONV=1" EXTRA_ARGS="--mamba-ssm-dtype float32"
run dp_attention    EXTRA_ARGS="--enable-dp-attention --dp-size 4"
run topk_unfused    EXTRA_ENV="SGLANG_DSA_FUSE_TOPK=0"
run shared_fused    SHARED_FUSION_FLAG=" "
log "DISC2 DONE"
