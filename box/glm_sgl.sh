#!/usr/bin/env bash
# GLM-5.3-Flash (Artificial Analysis index 57, the highest-scoring open-weight model that
# fits 4x96 GB) on SGLang's own glm-5.3-flash build, lifted out of its Docker image.
#
# Why SGLang and not the vLLM vendor build: vLLM's only sm_120 sparse-attention backend is
# hard-wired to DeepSeek's packed fp8 cache (pe_dim must be 64) and GLM-5.3-Flash has
# qk_rope_head_dim=0, with no bf16 or dense fallback on sm_120. SGLang's DSA backend has
# TileLang prefill/decode kernels that JIT-compile per architecture; LibertAI verified
# exactly this recipe on GB10 (sm_121), one step from our sm_120.
#
# The image is used as a self-contained runtime: its own python, its own torch, its own
# sgl-kernel, so nothing has to be ABI-compatible with the working vLLM environment.
set -u
IMG=${IMG:-/workspace/sglimg2}
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=/workspace/models/GLM-5.3-Flash-NVFP4
TOK=/workspace/models/glm53f_tok          # tokenizer-only copy; main-env transformers lacks glm5_next
# the bench client and aggregators run in the BOX env, not the lifted image env
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh

# ---- locate the image's runtime -------------------------------------------------
SITE=$(find "$IMG" -maxdepth 7 -type d -name sglang -path "*packages*" 2>/dev/null | head -1)
[ -z "$SITE" ] && { log "no sglang package in $IMG"; exit 1; }
SITE=$(dirname "$SITE")                                   # .../dist-packages
PYVER=$(basename "$(dirname "$SITE")")                    # python3.X
PYBIN=$(ls "$IMG"/usr/bin/"$PYVER" "$IMG"/usr/local/bin/"$PYVER" 2>/dev/null | head -1)
if [ -n "$PYBIN" ] && [ -d "$IMG/usr/lib/$PYVER" ]; then
  export PYTHONHOME="$IMG/usr"
  PY="$PYBIN"
else
  PY=$(command -v "$PYVER" || command -v python3)
fi
# sglang and transformers are editable installs in this image: the real trees live under
# /sgl-workspace and site-packages only holds stubs. Put the sources first.
SRC_SGL="$IMG/sgl-workspace/sglang/python"
SRC_TF="$IMG/sgl-workspace/transformers/src"
# .pth files are only honoured in real site directories, not PYTHONPATH entries, so the
# CUTLASS DSL (import name "cutlass", provided via nvidia_cutlass_dsl_packages.pth) has to
# be added by hand or FlashInfer's CUTLASS MoE path dies at graph capture.
export PYTHONPATH="$SRC_SGL:$SRC_TF:$SITE:$SITE/nvidia_cutlass_dsl/dsl_packages"
export LD_LIBRARY_PATH="$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
log "runtime: $PY  ($PYVER)  site=$SITE"

log "does the image's SGLang import and know glm5_next?"
"$PY" - <<'PY' 2>&1 | grep -vE "Ignore import error|^$" | tail -6
import torch, sglang
print("  torch", torch.__version__, "| cuda ok:", torch.cuda.is_available(), "| cap:", torch.cuda.get_device_capability())
import importlib.metadata as md
print("  sglang", md.version("sglang"))
try:
    from sglang.srt.models import registry as R
    ms = list(R.ModelRegistry.models.keys())
    print("  glm5 archs:", [m for m in ms if "Glm5" in m or "GLM5" in m] or "NONE")
except Exception as e:
    print("  registry FAILED:", type(e).__name__, str(e)[:140])
import importlib
for mod in ("sgl_kernel", "tilelang", "flashinfer"):
    try:
        m = importlib.import_module(mod); print(f"  {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod} FAILED: {type(e).__name__} {str(e)[:120]}")
PY
[ "${PROBE_ONLY:-0}" = 1 ] && { log "probe only, stopping here"; exit 0; }

# ---- launch ----------------------------------------------------------------------
launch(){ # tag [extra sglang args...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_sglglm.sh" <<L
#!/usr/bin/env bash
export PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
${PYTHONHOME:+export PYTHONHOME="$PYTHONHOME"}
export MAX_JOBS=6 NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
export SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0
# defaults to true and calls deep_gemm in the hyper-connection pre-norm: NameError on sm_120
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
# prompts up to this KV length take the dense one-shot prefill (patched in for sm_120 via
# dsa_sm120.py); upstream sets it to index_topk=2048, we cover the judge shape too
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=${DENSE_THR:-8192}
# discriminator knobs: arbitrary env (space-separated KEY=VAL) into the server process
${EXTRA_ENV:+export $EXTRA_ENV}
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "$PY" -m sglang.launch_server \\
  --model-path $MD --served-model-name glm53f \\
  --host 0.0.0.0 --port 8000 --tp-size 4 \\
  --attention-backend dsa \\
  --dsa-prefill-backend ${DSA_PREFILL:-triton} --dsa-decode-backend ${DSA_DECODE:-triton} \\
  --moe-runner-backend ${MOE_BACKEND:-flashinfer_cutlass} \\
  --kv-cache-dtype ${KV_DTYPE:-bfloat16} ${SHARED_FUSION_FLAG:---disable-shared-experts-fusion} \\
  --reasoning-parser glm45 --tool-call-parser glm47 \\
  --mem-fraction-static 0.85 --context-length 40960 \\
  --max-running-requests 256 --chunked-prefill-size 8192 \\
  --disable-custom-all-reduce --trust-remote-code ${EXTRA_ARGS:-} $*
L
  chmod +x "$B/l_sglglm.sh"
  log "  launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_sglglm.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0 ok=0
  while [ "$t" -lt 3000 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && break
    sleep 15; t=$((t+15))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "error|not supported|no kernel|Traceback|ValueError|out of memory|Unknown|died" "$S/${tag}.log" \
      | grep -vE "Ignore import error|import_utils" | tail -8 | cut -c1-200
    return 1
  fi
  log "  $tag healthy in ${t}s"
  grep -m2 -oiE "KV cache.*tokens|max_total_num_tokens[^,]*|#tokens[^,]*" "$S/${tag}.log" | sed 's/^/    /'
  return 0
}

pt(){ # tag label in out prefix conc
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  mkdir -p "$P/$tag"
  $CLEAN vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model glm53f --tokenizer "$TOK" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*6)) --ignore-eos --seed $((9100+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag"
}

sweep(){ # tag [extra...]
  local tag="$1"; shift
  if launch "$tag" "$@"; then
    # correctness first: a corrupted server is not worth benchmarking
    $CLEAN python3 "$B/quality20.py" glm53f http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode "${QMODE:-chat}" --max-tokens "${QMAX:-1024}" 2>&1 | tail -1
    if [ "${HOLD:-0}" = 1 ]; then
      log "  HOLD=1: server stays up for probing until /workspace/RELEASE exists"
      rm -f /workspace/RELEASE; while [ ! -e /workspace/RELEASE ]; do sleep 10; done
    fi
    if [ "${QUALITY_ONLY:-0}" = 1 ]; then log "  quality only, skipping throughput"; return; fi
    for c in 64 256; do pt "$tag" router 1024 128 0 "$c"; done
    pt "$tag" promptopt 512 256 3072 256
    pt "$tag" judge 4096 512 0 128
    curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -iE "spec|accept" | head -4 | sed 's/^/    spec: /'
  fi
}

log "===== GLM-5.3-Flash on SGLang, sm_120 ====="
log "--- A: baseline, no speculation ---"
sweep sglglm_base
case "${ARMS:-base,mtp}" in *mtp*)
log "--- B: native MTP head (NEXTN), 3 steps ---"
sweep sglglm_mtp --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
;; esac
log "SGLGLM DONE"
kill_all
