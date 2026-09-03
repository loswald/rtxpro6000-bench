#!/usr/bin/env bash
# Round 4: (a) push the concurrency ladder until it plateaus, recording full latency;
#          (b) prove the fp8 KV cache we switched on for throughput costs no quality.
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

log "PART A: extend the ladder to find the real plateau (full TTFT/TPOT/ITL/E2E recorded)"
if launch_x4 sat2 "$MD/gpt-oss-120b" gptoss --moe-backend marlin; then
  point sat2 short     256   64 0    512
  point sat2 router   1024  128 0    512
  point sat2 promptopt 512  256 3072 256
  point sat2 promptopt 512  256 3072 512
  point sat2 judge    4096  512 0    256
  point sat2 rollout  8192 2048 0    128
fi

log "PART B: fp8 KV vs bf16 KV, greedy, identical prompts, both live at once"
kill_all
cat > "$B/l_kv.sh" <<'LKV'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
  --host 0.0.0.0 --port 8000 --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
  --trust-remote-code --disable-uvicorn-access-log > /workspace/results/smoke/kv_fp8.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
  --host 0.0.0.0 --port 8001 --kv-cache-dtype auto --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
  --trust-remote-code --disable-uvicorn-access-log > /workspace/results/smoke/kv_bf16.log 2>&1 &
wait
LKV
chmod +x "$B/l_kv.sh"
tmux new-session -d -s srv "bash $B/l_kv.sh"
t=0; ok=0
while [ "$t" -lt 720 ]; do
  ok=1
  for p in 8000 8001; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
  [ "$ok" = 1 ] && break
  sleep 10; t=$((t+10))
done
if [ "$ok" = 1 ]; then
  log "both KV servers up in ${t}s (8000 = fp8, 8001 = bf16)"
  grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/kv_fp8.log"  | sed 's/^/  fp8  /'
  grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/kv_bf16.log" | sed 's/^/  bf16 /'
  python3 "$B/kvdiff.py" http://127.0.0.1:8000 http://127.0.0.1:8001 gptoss "$P/kv_fp8_vs_bf16.json"
else
  log "KV comparison servers failed to start"
  grep -iE "error|not supported" "$S/kv_fp8.log" 2>/dev/null | head -3 | cut -c1-180
fi

log "ROUND4 DONE"
kill_all
