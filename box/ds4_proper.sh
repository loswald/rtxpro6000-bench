#!/usr/bin/env bash
# DeepSeek-V4-Flash, optimised properly this time.
# Everything that won on gpt-oss, applied here, plus the saturation ladder that
# every previous DeepSeek measurement was missing. No sleep mode.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
DS=$MD/DeepSeek-V4-Flash-0731
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 8
}

serve(){ # tag  [extra vllm args...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_ds.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve $DS --served-model-name ds4 --host 0.0.0.0 --port 8000 \\
  --tokenizer-mode deepseek_v4 --block-size 256 \\
  --attention_config.use_fp4_indexer_cache False \\
  --kv-cache-dtype fp8 --max-model-len 40960 \\
  --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.96 \\
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x \\
  --disable-custom-all-reduce --enable-prefix-caching --trust-remote-code \\
  --disable-uvicorn-access-log $*
L
  chmod +x "$B/l_ds.sh"
  log "launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_ds.sh > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  local t=0
  while [ "$t" -lt 1200 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && {
      log "  $tag healthy ${t}s | $(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/$tag.log") | $(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend" "$S/$tag.log")"
      return 0; }
    if grep -q "^EXIT=" "$S/$tag.log" 2>/dev/null; then
      log "  $tag DIED"
      grep -iE "does not support|invalid|Error|error" "$S/$tag.log" | grep -vE "import_utils|deep_ep" | head -2 | cut -c1-200
      return 1
    fi
    sleep 10; t=$((t+10))
  done
  log "  $tag TIMED OUT"; return 1
}

pt(){ # tag label in out prefix concurrency   (steady state: 8x prompts)
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  local np=$(( c * 8 ))
  [ "$np" -gt 2048 ] && np=2048
  mkdir -p "$P/$tag"
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model ds4 --tokenizer "$DS" --tokenizer-mode deepseek_v4 --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((7000+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}

log "########## A: best-known kernel stack + SATURATION LADDER ##########"
if serve ds_ficutlass --tensor-parallel-size 4 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8; then
  pt ds_ficutlass router 1024 128 0 256
  pt ds_ficutlass router 1024 128 0 512
  pt ds_ficutlass router 1024 128 0 1024
  pt ds_ficutlass promptopt 512 256 3072 512
  pt ds_ficutlass judge 4096 512 0 256
  python3 "$B/quality20.py" ds4 http://127.0.0.1:8000 "$P/ds_ficutlass_quality20.json" 2>&1 | tail -1
else
  log "  ficutlass rejected for DSV4; falling back to marlin+EP at saturation"
  if serve ds_marlin_ep --tensor-parallel-size 4 --enable-expert-parallel --moe-backend auto; then
    pt ds_marlin_ep router 1024 128 0 256
    pt ds_marlin_ep router 1024 128 0 512
    pt ds_marlin_ep router 1024 128 0 1024
    pt ds_marlin_ep promptopt 512 256 3072 512
    pt ds_marlin_ep judge 4096 512 0 256
  fi
fi

log "########## B: marlin + EP at saturation (control for A) ##########"
if serve ds_ep_sat --tensor-parallel-size 4 --enable-expert-parallel --moe-backend auto; then
  pt ds_ep_sat router 1024 128 0 512
  pt ds_ep_sat router 1024 128 0 1024
fi

log "########## C: data-parallel 4, the layout that won on the ACS host ##########"
if serve ds_dp4 --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel --moe-backend auto; then
  pt ds_dp4 router 1024 128 0 512
  pt ds_dp4 router 1024 128 0 1024
  pt ds_dp4 judge 4096 512 0 256
fi

log "DS4-PROPER DONE"
kill_all
