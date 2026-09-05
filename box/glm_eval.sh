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
TOK=/workspace/models/glm53f_tok
mkdir -p "$P" "$S" "$R/eval" /workspace/glmvllm
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$B/hardkill.sh"
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"

VEND=/workspace/glmimg/usr/local/lib/python3.12/dist-packages/vllm
[ -d "$VEND" ] || { log "no vendor vLLM tree at $VEND"; exit 1; }
[ -e /workspace/glmvllm/vllm ] || ln -s "$VEND" /workspace/glmvllm/vllm
python3 "$B/vllm_sm120_nope.py" "$VEND" 2>&1 | sed 's/^/  /'

launch(){ # tag [extra...]   - EXTRA_ARGS come after the defaults, so a flag repeated there wins
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
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \\
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
    grep -ohE "ValueError: [^\"]{0,140}|RuntimeError: [^\"]{0,140}|NotImplementedError: [^\"]{0,140}|CUDA out of memory[^\"]{0,60}" "$S/${tag}.log" | sort -u | head -3 | sed 's/^/    /'
    return 1
  }
  log "  $tag healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}.log")"
  return 0
}

run_eval_for(){ # tag
  local tag="$1"
  [ -z "${FORCE:-}" ] && [ -f "$R/eval/$tag.json" ] && { log "  $tag already evaluated"; return 0; }
  # The server separates thinking into the reasoning field via --reasoning-parser glm45, so the scorers see
  # only the answer. No meaningful token cap: at the old budget half the maths items finished on it, which
  # measured the cap rather than the model. Time is the only limit, and running out of time marks an item
  # skipped (excluded from accuracy) where running out of tokens marked it wrong.
  #
  # One hour per request, not the runner's ten-minute default. At 96 concurrent streams a 32k-token
  # reasoning trace takes longer than ten minutes here, and the first pass lost 36 of 403 items - the
  # hardest ones, since they are the ones that run long - to "error: timeout". Each was then retried three
  # times, which is where most of that pass's wall-clock went.
  #
  # Sampling is Z.AI's own recipe (T=0.95, top_p=0.95, min_p=0) with reasoning_effort at its default max,
  # not the suite's house T=0.6 - the same class of error that cost every other model on this roster.
  $CLEAN python3 "$B/evalsuite/run_eval.py" --tag "$tag" --base-urls http://127.0.0.1:8000 --model m \
    --out "$R/eval" --gpus 4 --time-budget "${EVAL_BUDGET:-5400}" --concurrency "${EVAL_CONC:-96}" \
    --reasoning --request-timeout "${EVAL_REQ_TIMEOUT:-3600}" --max-tokens "${EVAL_MAXTOK:-32768}" \
    --max-tokens-family "${EVAL_CAPS:-math=32768,code=20480,knowledge=20480,ifeval=16384,tools=8192,longctx=6144}" \
    --temperature "${GLM_T:-0.95}" --top-p "${GLM_TOPP:-0.95}" --extra-body '{"min_p":0.0}' \
    --chat-template-kwargs '{"reasoning_effort":"max"}' \
    ${EVAL_ARGS:-} 2>&1 | tail -12 | sed 's/^/    eval: /'
}

pt(){ # tag label in out prefix conc   - the same shapes every other model gets, against this server
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  mkdir -p "$P/$tag"
  $CLEAN vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model m --tokenizer "$TOK" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*6)) --ignore-eos --seed $((9300+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}
bench_shapes(){ # tag maxconc   - concurrency clamped to what the server was launched with
  local tag="$1" mc="$2" c
  for c in 64 256; do [ "$c" -le "$mc" ] && pt "$tag" router 1024 128 0 "$c"; done
  [ "$mc" -lt 64 ] && pt "$tag" router 1024 128 0 "$mc"
  c=256; [ "$mc" -lt "$c" ] && c="$mc"; pt "$tag" promptopt 512 256 3072 "$c"
  c=128; [ "$mc" -lt "$c" ] && c="$mc"; pt "$tag" judge 4096 512 0 "$c"
}

log "===== GLM-5.3-Flash (AA 57): quality ====="
for arm in ${ARMS:-base mtp}; do
  case "$arm" in
    base) if launch glm53f_base; then
            $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_base_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
            run_eval_for glm53f_base
          fi;;
    resume) # Finish glm53f_base. The first pass scored 367 of 403; the other 36 (24 maths, 9 code, 2
          # instruction-following, 1 knowledge) hit the runner's 600 s request timeout, and they are the
          # hardest items because they are the ones that reason longest. Excluding them flatters the model
          # (0.861 on 367, where the MTP arm that finished them all scored 0.809 on 403). --resume re-runs
          # only the timed-out records and rewrites the summary over all 403, same caps, so the number is
          # like for like with every other row.
          if launch glm53f_base_resume; then FORCE=1 EVAL_ARGS="--resume" run_eval_for glm53f_base; fi;;
    long) if launch glm53f_long; then EVAL_BUDGET=7200 run_eval_for glm53f_long; fi;;
    best) # the fastest layout from glm_perf.sh, evaluated in full so throughput and quality are measured on the
          # same configuration; BEST_FLAGS carries its launch flags (layout, MoE kernel, sequence budget)
          if EXTRA_ARGS="${BEST_FLAGS:-}" launch glm53f_best; then EVAL_BUDGET="${BEST_BUDGET:-3600}" run_eval_for glm53f_best; fi;;
    mtp)  if SPEC='{"method":"glm5_next_mtp","num_speculative_tokens":3}' launch glm53f_mtp; then
            $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_mtp_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
            # speculation verifies against the same model, so this arm must score the same as the base one
            # on the items both scored - and it did: 0.872 vs 0.864 on 367, paired 13 vs 10, noise
            run_eval_for glm53f_mtp
          fi;;
    mtp64k) # 21 of 403 items ran past 32,768 output tokens and were marked wrong as truncated. A model
          # that needed more room, or one that loops? Same MTP server (lossless, and faster), the context
          # window raised to 98k, every family cap doubled. Whatever fraction of the truncated items resolve
          # is the cost the 32k cap was imposing on the best model here.
          if SPEC='{"method":"glm5_next_mtp","num_speculative_tokens":3}' EXTRA_ARGS="--max-model-len 98304" launch glm53f_mtp64k; then
            EVAL_MAXTOK=65536 EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
              EVAL_BUDGET="${MTP64K_BUDGET:-10800}" run_eval_for glm53f_mtp64k
          fi;;
    fp8)  # Fidelity arm. Everything else here serves RedHatAI's NVFP4 build, which is a post-training
          # quantisation of this model; Z.AI's own FP8 release is the precision it was trained at, and at
          # ~330 GB it fits 4x96 GB with a few tens of GB left for KV. The gap between this arm and the NVFP4
          # one IS the cost of quantisation on the highest-intelligence model that fits the node. Same context
          # window and the same caps as every other arm - a reduced cap would measure the cap, not the model;
          # fewer concurrent sequences is the only concession the memory forces. Its throughput is measured
          # too, on the shapes every other model gets, so the price of native precision is a number.
          MD=${MD_FP8:-/workspace/models/GLM-5.3-Flash-FP8}
          if [ ! -f "$MD/.dl_complete" ]; then log "SKIP glm53f_fp8 (native FP8 checkpoint not downloaded)"; continue; fi
          if EXTRA_ARGS="--gpu-memory-utilization 0.94 --max-num-seqs 32" launch glm53f_fp8; then
            $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/glm53f_fp8_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
            EVAL_CONC=32 EVAL_BUDGET="${FP8_BUDGET:-10800}" run_eval_for glm53f_fp8
            bench_shapes glm53f_fp8 32
          fi;;
    spec) # Kept for the record: the greedy-sequence test and its control. The control (the same server
          # captured twice) matched on 4 of 12, so this test cannot attribute anything on this stack; the
          # paired task comparison and the logit pass are what settled the question.
          if launch glm53f_specbase; then
            $CLEAN python3 "$B/specdiff.py" capture http://127.0.0.1:8000 m "$P/specdiff_glm_base.json" 2>&1 | tail -14 | sed 's/^/    base: /'
            $CLEAN python3 "$B/specdiff.py" capture http://127.0.0.1:8000 m "$P/specdiff_glm_base2.json" 2>&1 | tail -3 | sed 's/^/    base2: /'
            log "  CONTROL: the same server, twice, greedy"
            $CLEAN python3 "$B/specdiff.py" compare "$P/specdiff_glm_base.json" "$P/specdiff_glm_base2.json" 2>&1 | tail -3 | sed 's/^/    control: /'
          fi
          if SPEC='{"method":"glm5_next_mtp","num_speculative_tokens":3}' launch glm53f_specmtp; then
            $CLEAN python3 "$B/specdiff.py" capture http://127.0.0.1:8000 m "$P/specdiff_glm_mtp.json" 2>&1 | tail -14 | sed 's/^/    mtp:  /'
          fi
          $CLEAN python3 "$B/specdiff.py" judge "$P/specdiff_glm_base.json" "$P/specdiff_glm_base2.json" "$P/specdiff_glm_mtp.json" 2>&1 | sed 's/^/    /'
          ;;
  esac
done
log "GLMEVAL DONE"
kill_all
