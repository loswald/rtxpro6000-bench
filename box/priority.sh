#!/usr/bin/env bash
# Credit-constrained priority run. Order = most decision-critical first.
#   1. KV quality gate: is the fp8 cache that bought us +110% actually free?  (~12 min)
#   2. FlashInfer CUTLASS MoE, correctly invoked, at saturation.             (~10 min)
#   3. Batching knob at saturation, where it can finally bind.               (~8 min)
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

log "PRIORITY 1: fp8 KV vs bf16 KV, greedy, identical prompts, both live simultaneously"
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
while [ "$t" -lt 600 ]; do
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
  log "KV servers failed to start"
  grep -iE "error|not supported" "$S/kv_fp8.log" 2>/dev/null | head -3 | cut -c1-180
fi

log "PRIORITY 2: FlashInfer CUTLASS MoE with mxfp8 activations, at saturation"
if launch_x4 pr_ficutlass "$MD/gpt-oss-120b" gptoss --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8; then
  point pr_ficutlass short  256  64 0 256
  point pr_ficutlass router 1024 128 0 256
fi

log "PRIORITY 3: batching budget at saturation (the knob that did nothing when we were unsaturated)"
if launch_x4 pr_mnbt16k "$MD/gpt-oss-120b" gptoss --moe-backend marlin --max-num-batched-tokens 16384; then
  point pr_mnbt16k short  256  64 0 256
  point pr_mnbt16k router 1024 128 0 256
fi

log "PRIORITY 4: in-session control so the two arms above are readable"
if launch_x4 pr_control "$MD/gpt-oss-120b" gptoss --moe-backend marlin; then
  point pr_control short  256  64 0 256
  point pr_control router 1024 128 0 256
fi

log "PRIORITY DONE"
kill_all
