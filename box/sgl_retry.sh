#!/usr/bin/env bash
# SGLang retry for DeepSeek-V4-Flash.
# First attempt died at kernel 96/97 of the FlashInfer sm_120 JIT compile, scheduler
# exit -3, while simultaneously loading a 156 GB model. Fix: compile the kernels FIRST
# against a tiny model with bounded nvcc parallelism, then launch the real server against
# a warm cache.
B=/workspace/bench; R=/workspace/results; S=$R/smoke; P=$R/probe; MD=/workspace/models
DS=$MD/DeepSeek-V4-Flash-0731
V=/workspace/venv-sgl2
mkdir -p "$S" "$P"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =sgls 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 8
}
kill_all
free -g | head -2

log "STEP 1: warm the FlashInfer sm_120 JIT cache with bounded nvcc parallelism"
cat > "$B/sgl_warm.sh" <<'W'
#!/usr/bin/env bash
source /workspace/venv-sgl2/bin/activate
export MAX_JOBS=6 NVCC_THREADS=2                 # was unbounded: 97 parallel nvcc jobs
export SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
timeout 2400 python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.8-27B-FP8 --served-model-name warm \
  --host 127.0.0.1 --port 8009 --tp-size 1 \
  --context-length 4096 --max-running-requests 8 --mem-fraction-static 0.60 \
  --kv-cache-dtype fp8_e4m3 --trust-remote-code &
SP=$!
for i in $(seq 1 160); do
  curl -fsS -m 3 http://127.0.0.1:8009/health >/dev/null 2>&1 && { echo "WARM-OK"; break; }
  kill -0 $SP 2>/dev/null || { echo "WARM-DIED"; break; }
  sleep 15
done
kill -9 $SP 2>/dev/null
W
chmod +x "$B/sgl_warm.sh"
tmux new-session -d -s sgls "bash $B/sgl_warm.sh > $S/sgl_warm.log 2>&1"
for i in $(seq 1 170); do
  grep -qE "WARM-OK|WARM-DIED" "$S/sgl_warm.log" 2>/dev/null && break
  sleep 15
done
log "  warm-up: $(grep -oE 'WARM-OK|WARM-DIED' "$S/sgl_warm.log" | tail -1)"
log "  cached kernels: $(find /root/.cache/sglang -name '*.so' 2>/dev/null | wc -l)"
kill_all

log "STEP 2: launch DeepSeek on SGLang with DP-attention, warm cache, bounded jobs"
cat > "$B/l_sgl2.sh" <<'L'
#!/usr/bin/env bash
source /workspace/venv-sgl2/bin/activate
export MAX_JOBS=6 NVCC_THREADS=2
export SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS
export SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS=1024
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_OPT_USE_TILELANG_INDEXER=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
export SGLANG_RAGGED_VERIFY_MODE=compact
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
exec python -m sglang.launch_server \
  --model-path /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4 \
  --host 0.0.0.0 --port 8000 \
  --tp-size 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head --ep-size 4 \
  --mem-fraction-static 0.82 --context-length 40960 \
  --max-running-requests 256 --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 4096 --disable-custom-all-reduce --trust-remote-code
L
chmod +x "$B/l_sgl2.sh"
tmux new-session -d -s sgls "bash $B/l_sgl2.sh > $S/sgl2.log 2>&1; echo EXIT=\$? >> $S/sgl2.log"
t=0; ok=0
while [ "$t" -lt 2100 ]; do
  curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  grep -q "^EXIT=" "$S/sgl2.log" 2>/dev/null && break
  sleep 15; t=$((t+15))
done
if [ "$ok" = 1 ]; then
  log "  SGLang healthy in ${t}s"
  for c in 128 256; do
    vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
      --model ds4 --tokenizer "$DS" --tokenizer-mode deepseek_v4 --trust-remote-code \
      --dataset-name random --random-input-len 1024 --random-output-len 128 --random-range-ratio 0 \
      --request-rate inf --max-concurrency $c --num-prompts $((c*6)) --ignore-eos --seed 321 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/sgl2_ds4" --result-filename "sgl2_ds4__router__c${c}__p8000.json" \
      > "$P/sgl2_ds4_router_c${c}.log" 2>&1
    mkdir -p "$P/sgl2_ds4"
    python3 "$B/agg.py" "$P/sgl2_ds4" "sgl2_ds4__router__c${c}__p" router "$c" sgl2_ds4
  done
  python3 "$B/quality20.py" ds4 http://127.0.0.1:8000 "$P/sgl2_ds4_quality20.json" 2>&1 | tail -1
else
  log "  SGLang failed again"
  grep -iE "error|assert|no kernel|died|OutOfMemory|Killed" "$S/sgl2.log" | tail -6 | cut -c1-190
  dmesg -T 2>/dev/null | grep -i "oom" | tail -3 | cut -c1-150
fi
log "SGL-RETRY DONE"
kill_all
