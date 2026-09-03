#!/usr/bin/env bash
# Decode-drift discriminators for GLM-5.3-Flash on sm_120. Each config: launch, run the
# chat-mode tripwire with a 1024-token budget, record the verdict line, tear down. A config
# that turns the verdict from "degenerate" to "ok" names the broken decode kernel.
set -u
B=/workspace/bench; R=/workspace/results; LOG=$R/disc.log
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
  cp $R/probe/sglglm_base_quality20.json "$R/probe/disc_${name}_quality20.json" 2>/dev/null
  grep -iE "error|Traceback|Exception" $R/smoke/sglglm_base.log | grep -vE "Ignore import" | tail -2 | cut -c1-160 | sed 's/^/      /' | tee -a "$LOG"
}
run graphs_off   EXTRA_ARGS="--disable-cuda-graph"
run moe_cutedsl  MOE_BACKEND=flashinfer_cutedsl
run moe_triton   MOE_BACKEND=triton
log "DISC DONE"
