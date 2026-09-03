#!/usr/bin/env bash
# Round 2, driven by the research pass. Tests the levers we had been testing WRONG or not at all.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -9 "$pid" 2>/dev/null; done
  sleep 8; }

arm(){ # $1 = tag, rest = extra vllm args
  local tag="$1"; shift
  kill_all
  log "=== $tag :: $*"
  cat > $B/l_r2.sh <<L
#!/usr/bin/env bash
# CRITICAL: VLLM_MOE_USE_DEEP_GEMM is a SEPARATE switch from VLLM_USE_DEEP_GEMM and defaulted to 1
# on every run we did before this one. DeepGEMM has no sm_120 MoE kernels.
export VLLM_MOE_USE_DEEP_GEMM=0
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss $*
L
  chmod +x $B/l_r2.sh
  tmux new-session -d -s srv "bash $B/l_r2.sh > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  local t=0 ok=0
  while [ $t -lt 720 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    if grep -q "^EXIT=" "$S/$tag.log" 2>/dev/null; then
      log "$tag REJECTED after ${t}s"
      grep -iE "invalid|does not support|not supported|Error|error" "$S/$tag.log" | grep -vE "import_utils|deep_ep" | head -2 | cut -c1-230
      return 1; fi
    sleep 10; t=$((t+10))
  done
  [ "$ok" = 1 ] || { log "$tag TIMED OUT"; return 1; }
  log "$tag healthy ${t}s | kernel: $(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend" $S/$tag.log)"
  python3 $B/quality20.py gptoss http://127.0.0.1:8000 $P/${tag}_quality20.json 2>&1 | tail -1
  bash $B/probe4.sh "$tag" gptoss $MD/gpt-oss-120b auto tune > $P/$tag.log 2>&1
  tail -n +2 $P/$tag/summary.tsv 2>/dev/null | awk -F'\t' '{printf "    %s C%s -> %s out tok/s, %s total\n",$2,$6,$10,$11}'
  return 0
}

log "R2-A: control on the nightly, with MoE DeepGEMM explicitly OFF"
arm r2_control

log "R2-B: FlashInfer CUTLASS MoE the way it actually dispatches on sm_120 (needs mxfp8 activations)"
arm r2_ficutlass_mxfp8 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8

log "R2-C: multiple API server processes - the frontend may be our real ceiling at 7,400 req/min"
arm r2_apiserver4 --api-server-count 4

log "R2-D: frontend scaling combined with the best MoE arm"
arm r2_apiserver4_ficutlass --api-server-count 4 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8

log "ROUND2 COMPLETE"
kill_all
