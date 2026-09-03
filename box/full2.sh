#!/usr/bin/env bash
# Full campaign v2. Uses the WINNING kernel (FlashInfer CUTLASS + mxfp8 activations,
# +26.5% on router vs Marlin, measured against an in-session control) as the baseline
# for gpt-oss, and adds MiniMax-M3 now that its 256 GB checkpoint has landed.
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

launch_tp4(){ # tag model alias [extra args...]
  local tag="$1" model="$2" alias="$3"; shift 3
  kill_all
  cat > "$B/l_tp4.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve $model --served-model-name $alias --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size 4 --kv-cache-dtype fp8 \\
  --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192 \\
  --gpu-memory-utilization 0.96 \\
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
  --no-enable-flashinfer-autotune --disable-custom-all-reduce \\
  --enable-prefix-caching --trust-remote-code --disable-uvicorn-access-log $*
L
  chmod +x "$B/l_tp4.sh"
  log "launch $tag (TP4) :: $*"
  tmux new-session -d -s srv "bash $B/l_tp4.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0
  while [ "$t" -lt 1200 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && {
      log "$tag healthy ${t}s | $(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/${tag}.log")"
      return 0; }
    if grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null; then
      log "$tag DIED after ${t}s"
      grep -iE "does not support|invalid|Error|error" "$S/${tag}.log" | grep -vE "import_utils|deep_ep" | head -3 | cut -c1-220
      return 1
    fi
    sleep 10; t=$((t+10))
  done
  log "$tag TIMED OUT"; return 1
}
point1(){ # tag label in out prefix concurrency alias model
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 alias=$7 model=$8
  local np=$(( c * 4 ))
  mkdir -p "$P/$tag"
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model "$alias" --tokenizer "$model" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((9000+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}

WIN=(--moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8)

log "=== A: gpt-oss-120b x4 on the WINNING kernel, full ladder ==="
if launch_x4 win_gptoss "$MD/gpt-oss-120b" gptoss "${WIN[@]}"; then
  point win_gptoss short     256   64 0    512
  point win_gptoss short     256   64 0   1024
  point win_gptoss router   1024  128 0    512
  point win_gptoss promptopt 512  256 3072 512
  point win_gptoss judge    4096  512 0    256
  point win_gptoss rollout  8192 2048 0    128
  python3 "$B/quality20.py" gptoss http://127.0.0.1:8000 "$P/win_gptoss_quality20.json" 2>&1 | tail -1
fi

log "=== B: Qwen3.8-27B-FP8 x4 at saturation ==="
if launch_x4 win_qwen27b "$MD/Qwen3.8-27B-FP8" qwen27b --kernel-config.linear_backend b12x; then
  point win_qwen27b short     256   64 0    512 qwen27b "$MD/Qwen3.8-27B-FP8"
  point win_qwen27b router   1024  128 0    512 qwen27b "$MD/Qwen3.8-27B-FP8"
  point win_qwen27b promptopt 512  256 3072 256 qwen27b "$MD/Qwen3.8-27B-FP8"
  point win_qwen27b judge    4096  512 0    256 qwen27b "$MD/Qwen3.8-27B-FP8"
  python3 "$B/quality20.py" qwen27b http://127.0.0.1:8000 "$P/win_qwen27b_quality20.json" 2>&1 | tail -1
fi

log "=== C: MiniMax-M3 MXFP4 TP4 - the biggest model that fits, does KV headroom kill it? ==="
if [ -f "$MD/MiniMax-M3-MXFP4/config.json" ]; then
  if launch_tp4 win_m3 "$MD/MiniMax-M3-MXFP4" m3 --max-model-len 16384 --max-num-seqs 64; then
    point1 win_m3 router   1024  128 0   64 m3 "$MD/MiniMax-M3-MXFP4"
    point1 win_m3 judge    4096  512 0   32 m3 "$MD/MiniMax-M3-MXFP4"
    python3 "$B/quality20.py" m3 http://127.0.0.1:8000 "$P/win_m3_quality20.json" 2>&1 | tail -1
  else
    log "MiniMax-M3 did not serve; recording the failure mode is the result"
  fi
else
  log "MiniMax-M3 checkpoint incomplete, skipping"
fi

log "=== D: DeepSeek-V4-Flash TP4 at saturation ==="
if launch_tp4 win_ds4 "$MD/DeepSeek-V4-Flash-0731" ds4flash \
     --enable-expert-parallel --tokenizer-mode deepseek_v4 --block-size 256 --moe-backend auto \
     --attention_config.use_fp4_indexer_cache False --kernel-config.linear_backend b12x; then
  point1 win_ds4 router   1024  128 0   256 ds4flash "$MD/DeepSeek-V4-Flash-0731"
  point1 win_ds4 promptopt 512  256 3072 256 ds4flash "$MD/DeepSeek-V4-Flash-0731"
  point1 win_ds4 judge    4096  512 0   128 ds4flash "$MD/DeepSeek-V4-Flash-0731"
  python3 "$B/quality20.py" ds4flash http://127.0.0.1:8000 "$P/win_ds4_quality20.json" 2>&1 | tail -1
fi

log "FULL2 DONE"
kill_all
