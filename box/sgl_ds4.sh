#!/usr/bin/env bash
# SGLang with DP-attention for DeepSeek-V4-Flash: the untested lever.
# Layout comes from the ombori 4x RTX PRO 6000 config, verified against its
# docker-compose.example.yml rather than its README.
# Also runs the vLLM DP4+EP TP1 layout that vLLM's OWN recipe recommends and
# that we never tried (we ran TP4 throughout).
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
DS=$MD/DeepSeek-V4-Flash-0731
V=/workspace/venv-sgl2
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 10
}

bench(){ # tag label in out prefix concurrency  (vllm bench serve against either engine)
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  local np=$(( c * 6 ))
  mkdir -p "$P/$tag"
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model ds4 --tokenizer "$DS" --tokenizer-mode deepseek_v4 --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((5000+c)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}

# ---------- A: vLLM, the layout vLLM's own recipe recommends (never tried) ----------
log "########## A: vLLM DP4 + EP, TP1 -- vLLM's own recommended layout ##########"
kill_all
cat > "$B/l_dp4.sh" <<'LA'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4 \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --block-size 256 --moe-backend marlin \
  --attention_config.use_fp4_indexer_cache False \
  --kv-cache-dtype fp8 --max-model-len 40960 \
  --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x \
  --disable-custom-all-reduce --enable-prefix-caching --trust-remote-code \
  --disable-uvicorn-access-log
LA
chmod +x "$B/l_dp4.sh"
tmux new-session -d -s srv "bash $B/l_dp4.sh > $S/vllm_dp4.log 2>&1; echo EXIT=\$? >> $S/vllm_dp4.log"
t=0; ok=0
while [ "$t" -lt 900 ]; do
  curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  grep -q "^EXIT=" "$S/vllm_dp4.log" 2>/dev/null && break
  sleep 10; t=$((t+10))
done
if [ "$ok" = 1 ]; then
  log "  vLLM DP4 healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' $S/vllm_dp4.log)"
  grep -m1 -oE "Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB" "$S/vllm_dp4.log" | sed 's/^/  CUDA graphs: /'
  bench vllm_dp4 router 1024 128 0 256
  bench vllm_dp4 router 1024 128 0 512
  bench vllm_dp4 promptopt 512 256 3072 512
else
  log "  vLLM DP4 FAILED"
  grep -iE "does not support|invalid|Error" "$S/vllm_dp4.log" | grep -vE "import_utils|deep_ep" | head -3 | cut -c1-190
fi

# ---------- B: SGLang with DP-attention ----------
log "########## B: SGLang 0.5.18, DP-attention + EP4 (the ombori layout) ##########"
kill_all
cat > "$B/l_sgl.sh" <<'LB'
#!/usr/bin/env bash
source /workspace/venv-sgl2/bin/activate
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
  --mem-fraction-static 0.85 --context-length 40960 \
  --max-running-requests 256 --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 4096 \
  --disable-custom-all-reduce \
  --trust-remote-code
LB
chmod +x "$B/l_sgl.sh"
tmux new-session -d -s srv "bash $B/l_sgl.sh > $S/sgl_ds4.log 2>&1; echo EXIT=\$? >> $S/sgl_ds4.log"
t=0; ok=0
while [ "$t" -lt 1500 ]; do
  curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  grep -q "^EXIT=" "$S/sgl_ds4.log" 2>/dev/null && break
  sleep 15; t=$((t+15))
done
if [ "$ok" = 1 ]; then
  log "  SGLang healthy ${t}s"
  bench sgl_ds4 router 1024 128 0 128
  bench sgl_ds4 router 1024 128 0 256
  bench sgl_ds4 promptopt 512 256 3072 256
  python3 "$B/quality20.py" ds4 http://127.0.0.1:8000 "$P/sgl_ds4_quality20.json" 2>&1 | tail -1
else
  log "  SGLang FAILED to start -- capturing why"
  grep -iE "error|not supported|assert|Traceback|CUDA|no kernel|instantiat" "$S/sgl_ds4.log" 2>/dev/null | tail -8 | cut -c1-200
fi

log "SGL-DS4 DONE"
kill_all
