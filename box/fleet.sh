#!/usr/bin/env bash
# The near-frontier fleet campaign.
# Replica tier first: models that fit ONE 96 GB card, which is where our economics are
# strongest (Qwen3.8-27B is 28.8x cheaper than API; gpt-oss 11x). Each model gets a
# kernel probe then a saturation point, all at steady state.
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

# ---- 4 independent replicas, one model per card -----------------------------
serve_x4(){ # tag dir alias [extra...]
  local tag="$1" dir="$2" alias="$3"; shift 3
  kill_all
  cat > "$B/l_f.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $dir --served-model-name $alias \\
    --host 0.0.0.0 --port \$((8000+i)) \\
    --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 \\
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.96 \\
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
    --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code \\
    --disable-uvicorn-access-log $* \\
    > /workspace/results/smoke/${tag}_p\$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x "$B/l_f.sh"
  log "  launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_f.sh"
  local t=0 ok=0
  while [ "$t" -lt 900 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break; sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED"
    grep -iE "does not support|no kernel|invalid|out of memory|Error" "$S/${tag}_p8000.log" 2>/dev/null | grep -vE "import_utils|deep_ep" | head -2 | cut -c1-180
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
  return 0
}

pt(){ # tag label in out prefix c_per_port alias dir
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 alias=$7 dir=$8
  local np=$(( c*8 )) tot=$(( c*4 ))
  mkdir -p "$P/$tag"
  for i in 0 1 2 3; do
    local p=$((8000+i))
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model "$alias" --tokenizer "$dir" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((4000+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "$P/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

run_model(){ # dir alias [extra vllm args...]
  local dir="$1" alias="$2"; shift 2
  [ -f "$dir/config.json" ] || { log "SKIP $alias (not downloaded)"; return; }
  log "########## $alias  ($(du -sh "$dir" 2>/dev/null | cut -f1)) ##########"
  if serve_x4 "f_$alias" "$dir" "$alias" "$@"; then
    pt "f_$alias" router    1024 128 0    256 "$alias" "$dir"
    pt "f_$alias" promptopt  512 256 3072 256 "$alias" "$dir"
    pt "f_$alias" judge     4096 512 0    128 "$alias" "$dir"
    python3 "$B/quality20.py" "$alias" http://127.0.0.1:8000 "$P/f_${alias}_quality20.json" 2>&1 | tail -1
  fi
}

log "===== REPLICA TIER: one model per card, four cards ====="
run_model "$MD/gpt-oss-20b"                gptoss20
run_model "$MD/Nemotron-3.5-Lightning-30B" nemo35   --kernel-config.linear_backend b12x
run_model "$MD/Qwen3.6-35B-A3B-FP8"        qwen36   --kernel-config.linear_backend b12x
run_model "$MD/gemma-4-26B-A4B-it"         gemma26
run_model "$MD/Muse-Glimmer-30B"           muse30
run_model "$MD/gemma-4-31B-it"             gemma31

log "===== re-baseline the champion in the same session for comparability ====="
run_model "$MD/gpt-oss-120b" gptoss120 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8

log "FLEET DONE"
kill_all
