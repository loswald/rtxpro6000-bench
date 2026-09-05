#!/usr/bin/env bash
# GLM-5.3-Flash: how fast can it go on this node? Every GLM throughput row so far was taken at 256 sequences
# (router C64/C256, promptopt C256, judge C128) with the MTP head off, while the server holds 2.1M KV tokens.
# This raises the sequence budget to 1,024 and runs the router at C256 and C1024, the shared-prefix shape at
# C1024 and the judge shape at C256, without and then with the MTP head (lossless on paired items; 67% of drafted
# tokens accepted) - the two levers that cost nothing to pull. Same vendor build and sm_120 port as glm_eval.sh.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=${MD:-/workspace/models/GLM-5.3-Flash-NVFP4}
TOK=/workspace/models/glm53f_tok
mkdir -p "$P" "$S" /workspace/glmvllm
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$B/hardkill.sh"
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
VEND=/workspace/glmimg/usr/local/lib/python3.12/dist-packages/vllm
[ -d "$VEND" ] || { log "no vendor vLLM tree at $VEND"; exit 1; }
[ -e /workspace/glmvllm/vllm ] || ln -s "$VEND" /workspace/glmvllm/vllm
python3 "$B/vllm_sm120_nope.py" "$VEND" 2>&1 | sed 's/^/  /'

launch(){ # tag [extra...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_gp.sh" <<L
#!/usr/bin/env bash
export PYTHONPATH=/workspace/glmvllm
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \\
  --model $MD --served-model-name m --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size 4 --attention-backend FLASHINFER_MLA_SPARSE_SM90 \\
  --kv-cache-dtype auto --block-size 1024 --max-model-len 40960 --max-num-seqs ${SEQS:-1024} \\
  --max-num-batched-tokens ${MNBT:-16384} --gpu-memory-utilization 0.90 \\
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \\
  --no-enable-flashinfer-autotune \\
  ${SPEC:+--speculative-config '$SPEC'} \\
  --disable-uvicorn-access-log $*
L
  chmod +x "$B/l_gp.sh"
  log "  launch $tag (max-num-seqs ${SEQS:-1024}, batched tokens ${MNBT:-16384}${SPEC:+, MTP})"
  tmux new-session -d -s srv "bash $B/l_gp.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0 ok=0
  while [ "$t" -lt 2700 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && break
    tmux has-session -t =srv 2>/dev/null || break
    sleep 15; t=$((t+15))
  done
  [ "$ok" = 1 ] || { log "  $tag FAILED after ${t}s"; grep -ohE "ValueError: [^\"]{0,140}|RuntimeError: [^\"]{0,140}|CUDA out of memory[^\"]{0,60}" "$S/${tag}.log" | sort -u | head -3 | sed 's/^/    /'; return 1; }
  log "  $tag healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}.log")"
}
pt(){ # tag label in out prefix conc
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  mkdir -p "$P/$tag"
  $CLEAN vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model m --tokenizer "$TOK" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*4)) --ignore-eos --seed $((9300+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag" 2>&1 | tail -1
  curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode_num_(accepted|draft)_tokens_total" | awk '{s=s" "$NF} END{if(s!="") print "    spec draft/accepted:"s}'
}
log "===== GLM-5.3-Flash throughput ceiling: 1,024 sequences, then MTP ====="
if launch glm53f_s1024; then
  $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_s1024_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
  pt glm53f_s1024 router 1024 128 0 256
  pt glm53f_s1024 router 1024 128 0 1024
  pt glm53f_s1024 promptopt 512 256 3072 1024
  pt glm53f_s1024 judge 4096 512 0 256
fi
if SPEC='{"method":"glm5_next_mtp","num_speculative_tokens":3}' launch glm53f_s1024_mtp; then
  pt glm53f_s1024_mtp router 1024 128 0 1024
  pt glm53f_s1024_mtp promptopt 512 256 3072 1024
  pt glm53f_s1024_mtp router 1024 128 0 256
fi
log "GLMPERF DONE"
kill_all
