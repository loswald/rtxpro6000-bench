#!/usr/bin/env bash
# Noise floor for the KV quality gate.
# Two servers with IDENTICAL settings (both fp8) on separate GPUs, same prompts, same diff.
# Whatever divergence this produces is the floor; the fp8-vs-bf16 number only means
# something measured against it.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 8
}
kill_all
cat > "$B/l_floor.sh" <<'LF'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
    --host 0.0.0.0 --port $((8000+i)) --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
    --trust-remote-code --disable-uvicorn-access-log \
    > /workspace/results/smoke/kv_floor_$i.log 2>&1 &
  sleep 2
done
wait
LF
chmod +x "$B/l_floor.sh"
log "launching two IDENTICAL fp8 servers for the noise floor"
tmux new-session -d -s srv "bash $B/l_floor.sh"
t=0; ok=0
while [ "$t" -lt 600 ]; do
  ok=1
  for p in 8000 8001; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
  [ "$ok" = 1 ] && break
  sleep 10; t=$((t+10))
done
if [ "$ok" = 1 ]; then
  log "both fp8 servers up in ${t}s"
  python3 "$B/kvdiff.py" http://127.0.0.1:8000 http://127.0.0.1:8001 gptoss "$P/kv_floor_fp8_vs_fp8.json"
  log "interpretation:"
  python3 - <<'PY'
import json
try:
    fl = json.load(open("/workspace/results/probe/kv_floor_fp8_vs_fp8.json"))["summary"]
    tr = json.load(open("/workspace/results/probe/kv_fp8_vs_bf16.json"))["summary"]
    print(f"  floor  (fp8 vs fp8 ): exact {fl['exact_rate']:.2f}  mean_ned {fl['mean_norm_edit_distance']:.4f}")
    print(f"  treatment (fp8 vs bf16): exact {tr['exact_rate']:.2f}  mean_ned {tr['mean_norm_edit_distance']:.4f}")
    excess = tr['mean_norm_edit_distance'] - fl['mean_norm_edit_distance']
    print(f"  EXCESS divergence attributable to the KV dtype: {excess:+.4f}")
    if excess <= 0.05:
        print("  VERDICT: fp8 KV is indistinguishable from bf16 at this sample size. The throughput gain is free.")
    elif excess <= 0.15:
        print("  VERDICT: small excess divergence. Needs an accuracy benchmark before trusting fp8.")
    else:
        print("  VERDICT: fp8 KV diverges materially beyond the noise floor. Re-report headline numbers on bf16.")
except Exception as e:
    print("  could not compare:", type(e).__name__, e)
PY
else
  log "floor servers failed to start"
fi
log "KVFLOOR DONE"
kill_all
