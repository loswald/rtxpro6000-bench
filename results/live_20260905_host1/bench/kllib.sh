#!/usr/bin/env bash
# Shared machinery for logit-level pairs: two servers side by side on two cards, one context set, twenty
# log-probabilities per position, and a JSON summary per pair. Sourced by kldiff.sh (the Qwen ladder on the
# 400 W box) and kldiff600w.sh (the quantiser pairs on the 600 W box). Every treatment is read against a
# control pair launched the same way, because two identical servers do not agree perfectly here.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
K=$R/kl; mkdir -p "$P" "$S" "$K"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$B/hardkill.sh"
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 40000 ]; then UTIL=0.88; MAXLEN=32768; else UTIL=0.90; MAXLEN=40960; fi

# one server on one card, named `m` so logit_diff.py can address both sides identically
one(){ # gpu port dir [extra...]
  local gpu="$1" port="$2" dir="$3"; shift 3
  {
    echo '#!/usr/bin/env bash'
    echo 'export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1'
    echo 'export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a'
    echo 'export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=4 NVCC_THREADS=2'
    printf 'CUDA_VISIBLE_DEVICES=%s vllm serve %q --served-model-name m --host 0.0.0.0 --port %d --max-model-len %d --max-num-seqs 32 --gpu-memory-utilization %s --trust-remote-code --no-enable-flashinfer-autotune --disable-uvicorn-access-log' \
      "$gpu" "$dir" "$port" "$MAXLEN" "$UTIL"
    for a in "$@"; do printf ' %q' "$a"; done
    printf ' > %s/kl_p%d.log 2>&1 &\n' "$S" "$port"
  } >> "$B/l_kl.sh"
}

pair(){ # tag  dirA nargsA argsA...  dirB nargsB argsB...
  local tag="$1"; shift
  local dirA="$1" nA="$2"; shift 2
  local argsA=(); local i; for i in $(seq 1 "$nA"); do argsA+=("$1"); shift; done
  local dirB="$1" nB="$2"; shift 2
  local argsB=(); for i in $(seq 1 "$nB"); do argsB+=("$1"); shift; done
  [ -f "$K/$tag.json" ] && { log "  $tag already measured"; return 0; }
  for d in "$dirA" "$dirB"; do
    [ -f "$d/config.json" ] || { log "SKIP $tag (missing $(basename "$d"))"; return 1; }
  done
  kill_all
  printf '#!/usr/bin/env bash\n' > "$B/l_kl.sh"
  one 0 8000 "$dirA" "${argsA[@]}"
  one 1 8001 "$dirB" "${argsB[@]}"
  echo 'wait' >> "$B/l_kl.sh"
  chmod +x "$B/l_kl.sh"
  log "########## $tag ##########"
  log "  A: $(basename "$dirA") ${argsA[*]}"
  log "  B: $(basename "$dirB") ${argsB[*]}"
  tmux new-session -d -s srv "bash $B/l_kl.sh"
  local t=0 ok=0
  while [ "$t" -lt 1500 ]; do
    ok=1; for p in 8000 8001; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "^(ValueError|RuntimeError|TypeError)|Engine core initialization failed|CUDA out of memory" "$S/kl_p8000.log" "$S/kl_p8001.log" 2>/dev/null && { sleep 15; break; }
    sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s: $(grep -ohE 'ValueError: [^\"]{0,110}|RuntimeError: [^\"]{0,110}|CUDA out of memory[^\"]{0,40}' "$S/kl_p8000.log" "$S/kl_p8001.log" 2>/dev/null | head -1)"
    kill_all; return 1
  fi
  log "  both healthy in ${t}s"
  $CLEAN python3 "$B/logit_diff.py" http://127.0.0.1:8000 http://127.0.0.1:8001 m "$K/$tag.json" "${POS:-16}" 2>&1 | tail -6 | sed 's/^/    /'
  kill_all
}
