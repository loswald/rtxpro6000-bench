#!/usr/bin/env bash
# Task-accuracy pass for GLM-5.3-Flash, the highest-intelligence open model that fits 384 GB (AA 57 of a
# 60 ceiling). It cannot go through ksweep.sh: that launches the box's own vLLM, and this architecture
# (glm5_next) exists only in Z.AI's vendor build, lifted from their image and patched for sm_120 by
# vllm_sm120_nope.py. So this reuses glm_vllm.sh's exact serving path and points the eval suite at it.
#
# The suite talks OpenAI HTTP and asks the server to tokenise, so it needs nothing locally - which matters,
# because the box's transformers has no glm5_next and cannot build the tokenizer itself.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=${MD:-/workspace/models/GLM-5.3-Flash-NVFP4}
mkdir -p "$P" "$S" "$R/eval" /workspace/glmvllm
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
  cat > "$B/l_ge.sh" <<L
#!/usr/bin/env bash
export PYTHONPATH=/workspace/glmvllm
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \\
  --model $MD --served-model-name m --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size 4 --attention-backend FLASHINFER_MLA_SPARSE_SM90 \\
  --kv-cache-dtype auto --block-size 1024 --max-model-len 40960 --max-num-seqs 256 \\
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \\
  --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice \\
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \\
  --no-enable-flashinfer-autotune \\
  ${SPEC:+--speculative-config '$SPEC'} \\
  --disable-uvicorn-access-log ${EXTRA_ARGS:-} $*
L
  chmod +x "$B/l_ge.sh"
  log "  launch $tag"
  tmux new-session -d -s srv "bash $B/l_ge.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0 ok=0
  while [ "$t" -lt 2700 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && break
    tmux has-session -t =srv 2>/dev/null || break
    sleep 15; t=$((t+15))
  done
  [ "$ok" = 1 ] || {
    log "  $tag FAILED after ${t}s"
    grep -ohE "ValueError: [^\"]{0,140}|RuntimeError: [^\"]{0,140}|NotImplementedError: [^\"]{0,140}" "$S/${tag}.log" | sort -u | head -3 | sed 's/^/    /'
    return 1
  }
  log "  $tag healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}.log")"
  return 0
}

run_eval_for(){ # tag
  local tag="$1"
  [ -f "$R/eval/$tag.json" ] && { log "  $tag already evaluated"; return 0; }
  # This model reasons, and the server strips it into the reasoning field via deepseek_r1, so the scorers
  # see only the answer. No meaningful token cap: at the old budget half the maths items finished on it,
  # which measured the cap rather than the model. Time is the only limit, and running out of time marks an
  # item skipped (excluded from accuracy) where running out of tokens marked it wrong.
  $CLEAN python3 "$B/evalsuite/run_eval.py" --tag "$tag" --base-urls http://127.0.0.1:8000 --model m \
    --out "$R/eval" --gpus 4 --time-budget "${EVAL_BUDGET:-5400}" --concurrency "${EVAL_CONC:-32}" \
    --reasoning --max-tokens "${EVAL_MAXTOK:-24576}" \
    --max-tokens-family "${EVAL_CAPS:-math=24576,code=16384,knowledge=16384,ifeval=12288,tools=8192,longctx=2048}" \
    ${EVAL_ARGS:-} 2>&1 | tail -12 | sed 's/^/    eval: /'
}

log "===== GLM-5.3-Flash (AA 57): quality ====="
for arm in ${ARMS:-base mtp}; do
  case "$arm" in
    base) if launch glm53f_base; then
            $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_base_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
            run_eval_for glm53f_base
          fi;;
    long) # kept as a named arm so the capped and uncapped runs can be compared directly; the defaults above
          # are already uncapped, so this differs from `base` only in giving the run more wall-clock time.
          if launch glm53f_long; then EVAL_BUDGET=7200 run_eval_for glm53f_long; fi;;
    fp8)  # Fidelity arm. Everything else here serves RedHatAI's NVFP4 build, which is a post-training
          # quantisation of this model; Z.AI's own FP8 release is the precision it was trained at, and at
          # 330 GB it fits 4x96 GB with ~13 GB of KV left. That is a poor throughput configuration and the
          # right quality reference: the gap between this arm and the NVFP4 one IS the cost of our
          # quantisation on the highest-intelligence model that fits the node.
          MD=${MD_FP8:-/workspace/models/GLM-5.3-Flash-FP8}
          if [ ! -f "$MD/.dl_complete" ]; then log "SKIP glm53f_fp8 (native FP8 checkpoint not downloaded)"; continue; fi
          if EXTRA_ARGS="--max-model-len 16384 --max-num-seqs 32" launch glm53f_fp8; then
            $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_fp8_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
            EVAL_CONC=8 EVAL_BUDGET=9000 EVAL_CAPS="math=12288,code=10240,knowledge=10240,ifeval=8192,tools=6144,longctx=1024" \
              run_eval_for glm53f_fp8
          fi;;
    mtp)  if SPEC='{"method":"glm5_next_mtp","num_speculative_tokens":3}' launch glm53f_mtp; then
            $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_mtp_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
            # speculation verifies against the same model, so this arm must score the same as the base one;
            # a gap is a bug in the speculator, not a trade-off
            run_eval_for glm53f_mtp
          fi;;
  esac
done
log "GLMEVAL DONE"
kill_all
