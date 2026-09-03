#!/usr/bin/env bash
# Per-model sweep for the models that need all four GPUs (DeepSeek-V4-Flash, MiniMax-M3).
# Here the axes are different again: parallelism layout (TP4 vs TP4+EP vs DP4+EP),
# MoE backend, linear backend, and for DeepSeek the sparse-MLA attention path.
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
RESULTS_CSV=$R/persweep.csv
[ -f "$RESULTS_CSV" ] || echo "model,arm,status,kernel,kv_tokens,shape,concurrency,out_tps,total_tps,ttft_ms" > "$RESULTS_CSV"

serve_tp4(){ # tag model alias maxlen maxseqs [extra...]
  local tag="$1" model="$2" alias="$3" maxlen="$4" seqs="$5"; shift 5
  kill_all
  cat > "$B/l_ps.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve $model --served-model-name $alias --host 0.0.0.0 --port 8000 \\
  --kv-cache-dtype fp8 --max-model-len $maxlen --max-num-seqs $seqs --max-num-batched-tokens 8192 \\
  --gpu-memory-utilization 0.96 \\
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
  --no-enable-flashinfer-autotune --disable-custom-all-reduce \\
  --enable-prefix-caching --trust-remote-code --disable-uvicorn-access-log $*
L
  chmod +x "$B/l_ps.sh"
  tmux new-session -d -s srv "bash $B/l_ps.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0
  while [ "$t" -lt 1200 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && return 1
    sleep 10; t=$((t+10))
  done
  return 1
}
probe1(){ # tag label in out prefix c alias model
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 alias=$7 model=$8
  mkdir -p "$P/$tag"
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model "$alias" --tokenizer "$model" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $(( c * 4 )) --ignore-eos --seed $((9000+c)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}
arm(){ # model alias maxlen seqs probe_c name args...
  local model="$1" alias="$2" maxlen="$3" seqs="$4" c="$5" name="$6"; shift 6
  local tag="sw_${alias}_${name}"
  log "  arm $name :: $*"
  # shellcheck disable=SC2086
  if serve_tp4 "$tag" "$model" "$alias" "$maxlen" "$seqs" $*; then
    local kern kv
    kern=$(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend|Selected [A-Za-z0-9]+ for Fp8LinearMethod" "$S/${tag}.log" | head -1)
    kv=$(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/${tag}.log" | grep -oE "[0-9,]+")
    log "    OK | ${kern:-kernel not logged} | KV ${kv:-?}"
    probe1 "$tag" router 1024 128 0 "$c" "$alias" "$model"
    probe1 "$tag" judge  4096 512 0 "$((c/2))" "$alias" "$model"
    python3 - "$P/$tag/summary_full.tsv" "$alias" "$name" "$kern" "$kv" "$RESULTS_CSV" <<'PY'
import csv, os, sys
tsv, model, arm, kern, kv, out = sys.argv[1:7]
if not os.path.exists(tsv): raise SystemExit
with open(out, "a") as fh:
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        fh.write(",".join([model, arm, "ok", (kern or "").replace(","," "), (kv or ""),
                           r["label"], r["C"], r["out_tps"], r["total_tps"], r["ttft_mean_ms"]]) + "\n")
PY
  else
    log "    REJECTED"
    grep -m1 -iE "does not support|invalid choice|not supported|no kernel|requires|Error" "$S/${tag}.log" 2>/dev/null | sed 's/.*\] /      /' | cut -c1-190
    echo "$alias,$name,rejected,,,,,,," >> "$RESULTS_CSV"
  fi
}

DS=$MD/DeepSeek-V4-Flash-0731
DSA="--tokenizer-mode deepseek_v4 --block-size 256 --attention_config.use_fp4_indexer_cache False"
log "############ DeepSeek-V4-Flash-0731 (MoE + sparse MLA) ############"
arm "$DS" ds4flash 40960 512 256 tp4_ep_auto   --tensor-parallel-size 4 --enable-expert-parallel --moe-backend auto $DSA --kernel-config.linear_backend b12x
arm "$DS" ds4flash 40960 512 256 tp4_b12x      --tensor-parallel-size 4 --moe-backend b12x $DSA --kernel-config.linear_backend b12x
arm "$DS" ds4flash 40960 512 256 dp4_ep        --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel --moe-backend auto $DSA --kernel-config.linear_backend b12x
arm "$DS" ds4flash 40960 512 256 tp4_ficutlass --tensor-parallel-size 4 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8 $DSA --kernel-config.linear_backend b12x

M3=$MD/MiniMax-M3-MXFP4
if [ -f "$M3/config.json" ]; then
  log "############ MiniMax-M3 MXFP4 (biggest model that fits; KV headroom is the question) ############"
  arm "$M3" m3 16384 64 64 tp4_auto      --tensor-parallel-size 4 --moe-backend auto
  arm "$M3" m3 16384 64 64 tp4_marlin    --tensor-parallel-size 4 --moe-backend marlin
  arm "$M3" m3 16384 64 64 tp4_ficutlass --tensor-parallel-size 4 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
else
  log "MiniMax-M3 checkpoint not ready, skipping"
fi

log "PERSWEEP-TP4 DONE"
kill_all
echo
log "=== combined per-model kernel results ==="
column -s, -t < "$RESULTS_CSV" 2>/dev/null || cat "$RESULTS_CSV"
