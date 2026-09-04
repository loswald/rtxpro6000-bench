#!/usr/bin/env bash
# Motif-3 (Motif Technologies, AA index 47, 186.9 GB NVFP4) on the vendor's vLLM fork, lifted from
# ghcr.io/motiftechnologies/vllm:v0.26.0-motif3-patch1 with pull_image.py (Motif's architecture was
# removed from upstream vLLM in Sep 2025; SGLang never had it). The image is a vLLM 0.26 line on
# CUDA 13.1 with an explicit sm_120 FP4 grouped-GEMM branch, so unlike the GLM image we take its own
# torch/triton too and run it as a self-contained tree: PYTHONPATH points at the image's dist-packages
# and nothing from the box environment is imported. Flags follow the vendor recipe (DP4 x TP1 + EP).
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=${MD:-/workspace/models/Motif-3-NVFP4}
IMG=${IMG:-/workspace/motifimg}
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh

DEPS=${DEPS:-/workspace/motifdeps}   # pure-python packages the image forgot (pydivsufsort); never installed INTO the lifted tree
SITE=$(find "$IMG" -maxdepth 6 -type d -name dist-packages -path "*python3*" 2>/dev/null | head -1)
[ -z "$SITE" ] && SITE=$(find "$IMG" -maxdepth 7 -type d -name site-packages -path "*python3*" 2>/dev/null | head -1)
[ -z "$SITE" ] && { log "no python tree under $IMG"; exit 1; }
PYBIN=$(ls "$IMG"/usr/bin/python3* 2>/dev/null | head -1); PYBIN=${PYBIN:-python3}
log "vendor tree: $SITE  (python: $PYBIN)"
log "  vendor torch: $(PYTHONPATH=$SITE:$DEPS $PYBIN -c 'import torch;print(torch.__version__)' 2>&1 | tail -1)"
log "  vendor vllm : $(PYTHONPATH=$SITE:$DEPS $PYBIN -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)"
PYTHONPATH=$SITE:$DEPS $PYBIN -c "from vllm.model_executor.models.registry import ModelRegistry as M; print('  motif archs:', [a for a in M.get_supported_archs() if 'otif' in a])" 2>&1 | tail -1

launch(){ # tag [extra...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_motif.sh" <<L
#!/usr/bin/env bash
export PYTHONPATH=$SITE:$DEPS
export VLLM_WORKER_MULTIPROC_METHOD=spawn EP_DISABLE_GIN=1
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec $PYBIN -m vllm.entrypoints.openai.api_server \\
  --model $MD --served-model-name motif3 --host 0.0.0.0 --port 8000 \\
  --trust-remote-code --quantization modelopt_nvfp4 --dtype bfloat16 \\
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \\
  --gpu-memory-utilization 0.85 --max-model-len 40960 --max-num-seqs 256 --block-size 128 \\
  --tool-call-parser motif --reasoning-parser motif --enable-auto-tool-choice \\
  ${SPEC:+--speculative-config '$SPEC'} \\
  --disable-log-requests ${EXTRA_ARGS:-} $*
L
  chmod +x "$B/l_motif.sh"
  log "  launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_motif.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0 ok=0
  while [ "$t" -lt 2700 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && break
    sleep 15; t=$((t+15))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "error|not supported|no kernel|Traceback|ValueError|out of memory|Unknown|NotImplemented|ImportError|undefined symbol" "$S/${tag}.log" \
      | grep -vE "import_utils|deep_ep" | tail -8 | cut -c1-200
    return 1
  fi
  log "  $tag healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}.log")"
  grep -m3 -oE "Using [A-Za-z0-9_]+ (attention backend|for NVFP4 GEMM)|Using .[A-Z0-9_]+. [A-Za-z0-9]+ MoE backend|Selected [A-Za-z0-9]+Kernel" "$S/${tag}.log" | sort -u | sed 's/^/    kernel: /'
  return 0
}

pt(){ # tag label in out prefix conc
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  mkdir -p "$P/$tag"
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model motif3 --tokenizer "$MD" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*6)) --ignore-eos --seed $((9900+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}

sweep(){ # tag [extra...]
  local tag="$1"; shift
  if launch "$tag" "$@"; then
    python3 "$B/quality20.py" motif3 http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
    for c in 64 256; do pt "$tag" router 1024 128 0 "$c"; done
    pt "$tag" promptopt 512 256 3072 256
    pt "$tag" judge 4096 512 0 128
    curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode.*(accepted|draft)" | head -3 | sed 's/^/    spec: /'
  fi
}

[ -f "$MD/config.json" ] || { log "Motif-3 weights absent at $MD"; exit 1; }
log "===== Motif-3 on the vendor vLLM fork (DP4 x TP1 + EP) ====="
case "${ARMS:-base,mtp}" in *base*) log "--- A: no speculation ---"; sweep motif3_base ;; esac
case "${ARMS:-base,mtp}" in *mtp*)
  log "--- B: built-in MTP, 1 draft token (vendor recipe) ---"
  SPEC="{\"model\":\"$MD\",\"num_speculative_tokens\":1}" sweep motif3_mtp
;; esac
log "MOTIF DONE"
kill_all
