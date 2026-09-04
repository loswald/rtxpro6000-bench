#!/usr/bin/env bash
# GLM-5.3-Flash on the vLLM route: Z.AI's vendor vLLM build (lifted from its image) with the
# sm_120 NoPE port applied, run against the BOX environment's FlashInfer 0.6.18 (the vendor
# tree only has 0.6.17, and head_dim_kpe=0 support arrived in 0.6.18). Only the vendor
# `vllm` package is put on PYTHONPATH, via a directory that contains nothing else, so every
# other library resolves from the working environment.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=/workspace/models/GLM-5.3-Flash-NVFP4; TOK=/workspace/models/glm53f_tok
mkdir -p "$P" "$S" /workspace/glmvllm
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"

VEND=/workspace/glmimg/usr/local/lib/python3.12/dist-packages/vllm
[ -e /workspace/glmvllm/vllm ] || ln -s "$VEND" /workspace/glmvllm/vllm
python3 /workspace/bench/vllm_sm120_nope.py "$VEND" | sed 's/^/  /'

launch(){ # tag [extra vllm args...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_gv.sh" <<L
#!/usr/bin/env bash
export PYTHONPATH=/workspace/glmvllm
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \\
  --model $MD --served-model-name glm53f --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size 4 --attention-backend FLASHINFER_MLA_SPARSE_SM90 \\
  --kv-cache-dtype ${KV_DTYPE:-auto} --max-model-len 40960 --max-num-seqs 256 \\
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \\
  --reasoning-parser glm47 --tool-call-parser glm47 \\
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \\
  --no-enable-flashinfer-autotune \\
  ${SPEC:+--speculative-config '$SPEC'} \\
  --disable-uvicorn-access-log ${EXTRA_ARGS:-} $*
L
  chmod +x "$B/l_gv.sh"
  log "  launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_gv.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0 ok=0
  while [ "$t" -lt 2700 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && break
    sleep 15; t=$((t+15))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "error|not supported|no kernel|Traceback|ValueError|out of memory|Unknown|NotImplemented" "$S/${tag}.log" \
      | grep -vE "import_utils|deep_ep|min_frames|max_frames" | tail -8 | cut -c1-200
    return 1
  fi
  log "  $tag healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}.log")"
  grep -m2 -oE "Using [A-Z0-9_]+ attention backend|Using .* MoE backend[^,]*" "$S/${tag}.log" | sed 's/^/    /'
  return 0
}

pt(){ # tag label in out prefix conc
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  mkdir -p "$P/$tag"
  $CLEAN vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model glm53f --tokenizer "$TOK" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*6)) --ignore-eos --seed $((9300+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}

sweep(){ # tag [extra...]
  local tag="$1"; shift
  if launch "$tag" "$@"; then
    $CLEAN python3 "$B/quality20.py" glm53f http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens "${QMAX:-2048}" 2>&1 | tail -1
    if [ "${QUALITY_ONLY:-0}" = 1 ]; then log "  quality only"; return; fi
    for c in 64 256; do pt "$tag" router 1024 128 0 "$c"; done
    pt "$tag" promptopt 512 256 3072 256
    pt "$tag" judge 4096 512 0 128
    curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode.*(accepted|draft)" | head -3 | sed 's/^/    spec: /'
  fi
}

log "===== GLM-5.3-Flash on vLLM (vendor build + sm_120 NoPE port) ====="
# ARMS selects arms (default both). The MTP JSON goes through SPEC so the launcher heredoc keeps
# its quotes; passing it as a positional arg stripped them and vLLM saw method:glm5_next_mtp.
case "${ARMS:-base,mtp}" in *base*)
log "--- A: no speculation ---"
sweep vglm_base
;; esac
case "${ARMS:-base,mtp}" in *mtp*)
log "--- B: MTP, 3 draft tokens ---"
SPEC='{"method":"glm5_next_mtp","num_speculative_tokens":3}' sweep vglm_mtp
;; esac
log "VGLM DONE"
kill_all
