#!/usr/bin/env bash
# NVFP4 vs FP8 on Qwen3.8-27B (Artificial Analysis index 52), in one session so the
# comparison is a real A/B rather than two runs from different days.
# Why this matters: published RTX PRO 6000 numbers put NVFP4 at ~1.36x FP8 throughput
# with GPQA-Diamond accuracy matching BF16 while FP8 loses a little. If that holds at
# our concurrencies it is a larger win than any speculation method, and it is free.
# The NVFP4 build is the RTX 5090 one -- same sm_120 silicon as our card, unlike the
# datacentre sm_100 builds that fail here.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh

serve_x4(){ # tag dir [extra...]
  local tag="$1" dir="$2"; shift 2
  kill_all
  cat > "$B/l_n.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export MAX_JOBS=6 NVCC_THREADS=2
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $dir --served-model-name q27 \\
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
  chmod +x "$B/l_n.sh"
  log "  launch $tag :: $(basename "$dir") $*"
  tmux new-session -d -s srv "bash $B/l_n.sh"
  local t=0 ok=0
  while [ "$t" -lt 1500 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break; sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "ValueError|not supported|no kernel|out of memory|Error|Unknown" "$S/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep|WARNING" | head -3 | cut -c1-200
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
  # record which kernel actually got selected, not which one we asked for
  grep -m1 -oE "Using .* (MoE|Mxfp4|linear) backend|Selected .*Kernel|quantization method: [a-z0-9_]*" "$S/${tag}_p8000.log" 2>/dev/null | sed 's/^/    kernel: /'
  return 0
}

pt(){ # tag label in out prefix c_per_port dir
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 dir=$7
  local np=$(( c*8 )) tot=$(( c*4 ))
  mkdir -p "$P/$tag"
  for i in 0 1 2 3; do
    local p=$((8000+i))
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model q27 --tokenizer "$dir" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((8100+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "$P/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

sweep(){ # tag dir [extra...]
  local tag="$1" dir="$2"; shift 2
  if serve_x4 "$tag" "$dir" "$@"; then
    pt "$tag" router    1024 128 0     64 "$dir"
    pt "$tag" router    1024 128 0    256 "$dir"
    pt "$tag" promptopt  512 256 3072 256 "$dir"
    pt "$tag" judge     4096 512 0    128 "$dir"
    curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null \
      | grep -E "spec_decode.*(accepted|draft|emitted)" | head -4 | sed 's/^/    spec: /'
    python3 "$B/quality20.py" q27 http://127.0.0.1:8000 "$P/${tag}_quality20.json" 2>&1 | tail -1
  fi
}

log "===== Qwen3.8-27B : NVFP4 vs FP8, same session, same shapes ====="
log "--- FP8 control ---"
sweep n_fp8   "$MD/Qwen3.8-27B-FP8"          --kernel-config.linear_backend b12x
log "--- NVFP4, sm_120 build ---"
sweep n_nvfp4 "$MD/Qwen27B-NVFP4-RTX5090"
log "--- NVFP4 + DSpark drafter (acceptance rate is the thing to read) ---"
sweep n_dspark "$MD/Qwen27B-NVFP4-RTX5090" \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$MD/Qwen27B-DSpark-NVFP4\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":1}"
log "NVTIER DONE"
kill_all