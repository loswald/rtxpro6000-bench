#!/usr/bin/env bash
# QUALITY TIER. Qwen3.8-27B scores 52 on the Artificial Analysis index -- more than double
# gpt-oss-120b's 24 -- and fits one card at 29 GB. It is the best intelligence-per-GB model
# that exists for this hardware, so it gets the full speculative-decoding treatment.
# It has NO MTP head (64 dense layers, no num_nextn_predict_layers), so MTP is unavailable.
# Instead we test the two draft-free methods, which suit our workloads: research traffic
# (judges quoting source text, counterfactual edits, agentic rollouts) has heavy token reuse,
# which is exactly where ngram and suffix decoding pay off.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh
DIR=$MD/Qwen3.8-27B-FP8; AL=qwen27b

serve_x4(){ # tag [extra...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_q.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $DIR --served-model-name $AL \\
    --host 0.0.0.0 --port \$((8000+i)) \\
    --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 \\
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.96 \\
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
    --kernel-config.linear_backend b12x \\
    --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code \\
    --disable-uvicorn-access-log $* \\
    > /workspace/results/smoke/${tag}_p\$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x "$B/l_q.sh"
  log "  launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_q.sh"
  local t=0 ok=0
  while [ "$t" -lt 900 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "ValueError|not supported|Unknown speculative|Traceback" "$S/${tag}_p8000.log" 2>/dev/null && { sleep 20; break; }
    sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "ValueError|not supported|Unknown|no kernel|out of memory|Error" "$S/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep|WARNING" | head -3 | cut -c1-200
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
  return 0
}

pt(){ # tag label in out prefix c_per_port
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  local np=$(( c*8 )) tot=$(( c*4 ))
  mkdir -p "$P/$tag"
  for i in 0 1 2 3; do
    local p=$((8000+i))
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model "$AL" --tokenizer "$DIR" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((7000+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "$P/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

# acceptance rate is the whole story for speculative decoding: a method that drafts
# tokens the target model rejects costs throughput instead of adding it.
accept(){
  local tag=$1
  local a=$(curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null \
    | grep -E "^vllm:spec_decode_(num_accepted_tokens|num_draft_tokens)_total" | awk '{print $2}' | paste -sd,)
  [ -n "$a" ] && log "  $tag spec counters (accepted,drafted): $a"
}

sweep(){ # tag [extra...]
  local tag="$1"; shift
  if serve_x4 "$tag" "$@"; then
    pt "$tag" router    1024 128 0    256
    pt "$tag" promptopt  512 256 3072 256
    pt "$tag" judge     4096 512 0    128
    accept "$tag"
    python3 "$B/quality20.py" "$AL" http://127.0.0.1:8000 "$P/${tag}_quality20.json" 2>&1 | tail -1
  fi
}

log "===== Qwen3.8-27B (AA index 52) : speculative decoding study ====="
log "--- A: baseline, no speculation (the control) ---"
sweep q27_base

log "--- B: ngram / prompt-lookup, 5 draft tokens ---"
sweep q27_ngram --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'

log "--- C: ngram, 3 draft tokens (shorter drafts accept more often) ---"
sweep q27_ngram3 --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":2}'

log "--- D: suffix decoding, adaptive draft length ---"
sweep q27_suffix --speculative-config '{"method":"suffix","num_speculative_tokens":8}'

log "QTIER DONE"
kill_all