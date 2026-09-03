#!/usr/bin/env bash
# Does "spin up a dedicated endpoint on demand" actually work?
# Measures, on real hardware:
#   1. cold start  : process launch -> /health, per model
#   2. sleep level 1: weights evicted to host RAM, VRAM freed
#   3. wake        : back to serving, and latency of the FIRST request after wake
#   4. concurrency : four DIFFERENT models on four cards, all serving at once
B=/workspace/bench; R=/workspace/results; S=$R/smoke; MD=/workspace/models
mkdir -p "$S" "$R"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 6
}
now(){ date +%s.%N; }
el(){ python3 -c "print(f'{$2-$1:.1f}')"; }
vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

serve1(){ # gpu port model alias [extra...]
  local g=$1 p=$2 m=$3 a=$4; shift 4
  CUDA_VISIBLE_DEVICES=$g VLLM_SERVER_DEV_MODE=1 \
  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1 \
  vllm serve "$m" --served-model-name "$a" --host 0.0.0.0 --port "$p" \
    --enable-sleep-mode --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 --no-enable-flashinfer-autotune \
    --enable-prefix-caching --trust-remote-code --disable-uvicorn-access-log "$@" \
    > "$S/od_${a}_$p.log" 2>&1 &
}
wait_health(){ local p=$1 lim=${2:-600} t=0
  while [ "$t" -lt "$lim" ]; do
    curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && return 0
    sleep 2; t=$((t+2))
  done; return 1; }
ping1(){ curl -sS -m 300 "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}" >/dev/null 2>&1; }

kill_all
log "=== PART 1: cold start, four different models on four cards, concurrently ==="
T0=$(now)
serve1 0 8000 "$MD/gpt-oss-120b"            gptoss   --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
serve1 1 8001 "$MD/Qwen3.8-27B-FP8"         qwen27b  --kernel-config.linear_backend b12x
serve1 2 8002 "$MD/gpt-oss-120b"            gptoss2  --moe-backend marlin
serve1 3 8003 "$MD/Qwen3.8-Flash-Next-NVFP4" qfn     --kernel-config.linear_backend b12x
for p in 8000 8001 8002 8003; do
  if wait_health "$p" 900; then
    T=$(now); log "  :$p healthy after $(el "$T0" "$T") s | VRAM $(vram $((p-8000))) MiB"
  else
    log "  :$p FAILED"
    grep -iE "error|not supported|no such|Traceback" "$S"/od_*_"$p".log 2>/dev/null | head -2 | cut -c1-170
  fi
done
log "  four distinct endpoints live simultaneously:"
for p in 8000 8001 8002 8003; do
  n=$(curl -fsS -m 5 "http://127.0.0.1:$p/v1/models" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "-")
  echo "      :$p -> $n"
done

log "=== PART 2: sleep / wake cycles on :8000 (the on-demand primitive) ==="
for i in 1 2 3; do
  ping1 8000 gptoss
  v0=$(vram 0)
  t0=$(now); curl -sS -m 300 -X POST "http://127.0.0.1:8000/sleep?level=1" >/dev/null 2>&1; t1=$(now)
  sleep 2; v1=$(vram 0)
  t2=$(now); curl -sS -m 300 -X POST "http://127.0.0.1:8000/wake_up" >/dev/null 2>&1; t3=$(now)
  t4=$(now); ping1 8000 gptoss; t5=$(now)
  log "  cycle $i: sleep $(el "$t0" "$t1")s  wake $(el "$t2" "$t3")s  first-request $(el "$t4" "$t5")s"
  log "           VRAM ${v0} -> ${v1} MiB while asleep"
done

log "=== PART 3: cold start for comparison (same model, full restart) ==="
kill_all
T0=$(now)
serve1 0 8000 "$MD/gpt-oss-120b" gptoss --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
if wait_health 8000 900; then T=$(now); log "  cold start: $(el "$T0" "$T") s"; fi
t0=$(now); ping1 8000 gptoss; t1=$(now); log "  first request after cold start: $(el "$t0" "$t1") s"

log "ONDEMAND DONE"
kill_all
