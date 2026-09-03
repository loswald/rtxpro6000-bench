#!/usr/bin/env bash
# SGLang vs vLLM on the champion config: gpt-oss-120b as 4 single-GPU replicas.
set +e
VENV=/workspace/venv-sglang; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =sgl 2>/dev/null
  for pid in $(pgrep -f "sglang.launch_serve[r]"); do kill "$pid" 2>/dev/null; done; sleep 8
  for pid in $(pgrep -f "sglang.launch_serve[r]"); do kill -9 "$pid" 2>/dev/null; done; sleep 3; }
kill_all
mkdir -p $P $S
cat > /workspace/bench/l_sgl_gptoss.sh <<'L'
#!/usr/bin/env bash
source /workspace/venv-sglang/bin/activate
export SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0
export NCCL_IB_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
    --model-path /workspace/models/gpt-oss-120b --served-model-name gptoss \
    --host 0.0.0.0 --port $((8000+i)) --tp-size 1 \
    --context-length 40960 --max-running-requests 256 --chunked-prefill-size 8192 \
    --mem-fraction-static 0.92 --trust-remote-code \
    > /workspace/results/smoke/sgl_gptoss_p$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
chmod +x /workspace/bench/l_sgl_gptoss.sh
log "launching SGLang gpt-oss x4"
tmux new-session -d -s sgl "bash /workspace/bench/l_sgl_gptoss.sh"
t=0; ok=0
while [ $t -lt 900 ]; do
  ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
  [ "$ok" = 1 ] && break; sleep 10; t=$((t+10))
done
if [ "$ok" = 1 ]; then
  log "SGLang healthy after ${t}s"
  python3 /workspace/bench/quality20.py gptoss http://127.0.0.1:8000 $P/sgl_gptoss_x4_quality20.json
  bash /workspace/bench/probe4.sh sgl_gptoss_x4 gptoss $MD/gpt-oss-120b auto full > $P/sgl_gptoss_x4.log 2>&1
  log "SGLang sweep done"
else
  log "SGLang FAILED to start"
  grep -iE "error|not supported|assert|Traceback" /workspace/results/smoke/sgl_gptoss_p8000.log 2>/dev/null | tail -6 | cut -c1-200
fi
kill_all
log "SGLANG-AB-DONE"
