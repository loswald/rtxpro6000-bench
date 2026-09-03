#!/usr/bin/env bash
# PER-MODEL kernel/parameter sweep.
#
# The optimum is model-specific: gpt-oss is an MXFP4 MoE, Qwen3.8-27B is dense FP8,
# DeepSeek-V4-Flash is an MoE with sparse MLA attention, MiniMax-M3 has its own sparse
# attention. A backend that wins on one is often rejected outright by another.
#
# For each model we: enumerate candidate configurations, launch each, record whether it
# loads AND which kernel the server actually selected, take two probe points, then declare
# the winner by measured throughput. Losers are recorded too -- a rejection is a result.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 8
}
# shellcheck disable=SC1091
. "$B/satlib.sh"

RESULTS_CSV=$R/persweep.csv
[ -f "$RESULTS_CSV" ] || echo "model,arm,status,kernel,kv_tokens,shape,concurrency,out_tps,total_tps,ttft_ms" > "$RESULTS_CSV"

# ---- replica sweep (models that fit one card) --------------------------------------
sweep_x4(){ # model_dir alias probe_c "arm_name|args" ...
  local model="$1" alias="$2" c="$3"; shift 3
  local spec name args tag
  for spec in "$@"; do
    name="${spec%%|*}"; args="${spec#*|}"
    tag="sw_${alias}_${name}"
    # shellcheck disable=SC2086
    if launch_x4 "$tag" "$model" "$alias" $args; then
      local kern kv
      kern=$(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend|Selected [A-Za-z0-9]+ for Fp8LinearMethod" "$S/${tag}_p8000.log" | head -1)
      kv=$(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/${tag}_p8000.log" | grep -oE "[0-9,]+")
      log "  $name OK | ${kern:-kernel not logged} | KV ${kv:-?}"
      point "$tag" router 1024 128 0 "$c" "$alias" "$model"
      point "$tag" judge  4096 512 0 "$((c/2))" "$alias" "$model"
      python3 - "$P/$tag/summary_full.tsv" "$model" "$name" "$kern" "$kv" "$RESULTS_CSV" <<'PY'
import csv, os, sys
tsv, model, arm, kern, kv, out = sys.argv[1:7]
if not os.path.exists(tsv): raise SystemExit
with open(out, "a") as fh:
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        fh.write(",".join([os.path.basename(model), arm, "ok", (kern or "").replace(","," "),
                           (kv or ""), r["label"], r["C"], r["out_tps"], r["total_tps"], r["ttft_mean_ms"]]) + "\n")
PY
    else
      log "  $name REJECTED"
      grep -m1 -iE "does not support|invalid choice|not supported|no kernel|requires" "$S/${tag}_p8000.log" 2>/dev/null | sed 's/.*\] /    /' | cut -c1-190
      echo "$(basename "$model"),$name,rejected,,,,,,," >> "$RESULTS_CSV"
    fi
  done
}

log "############ gpt-oss-120b (native MXFP4 MoE) ############"
sweep_x4 "$MD/gpt-oss-120b" gptoss 512 \
  "marlin|--moe-backend marlin" \
  "ficutlass_mxfp8|--moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8" \
  "ficutlass_plain|--moe-backend flashinfer_cutlass" \
  "ficutedsl|--moe-backend flashinfer_cutedsl --quantization-config.moe.activation mxfp8" \
  "fib12x|--moe-backend flashinfer_b12x" \
  "triton|--moe-backend triton"

log "############ Qwen3.8-27B-FP8 (dense, block-scaled FP8 -> LINEAR kernel is what matters) ############"
sweep_x4 "$MD/Qwen3.8-27B-FP8" qwen27b 512 \
  "lin_b12x|--kernel-config.linear_backend b12x" \
  "lin_auto|--kernel-config.linear_backend auto" \
  "lin_ficutlass|--kernel-config.linear_backend flashinfer_cutlass" \
  "lin_ficudnn|--kernel-config.linear_backend flashinfer_cudnn" \
  "lin_machete|--kernel-config.linear_backend machete" \
  "lin_b12x_kvbf16|--kernel-config.linear_backend b12x --kv-cache-dtype auto"

log "PERSWEEP DONE"
kill_all
echo
log "=== summary ==="
column -s, -t < "$RESULTS_CSV" 2>/dev/null || cat "$RESULTS_CSV"
