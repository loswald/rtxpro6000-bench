#!/usr/bin/env bash
# Is each speculation method lossless? Test it, per model, the only way that admits a yes/no answer.
#
# Speculative decoding proposes tokens with a cheap head and verifies each against the full model, so at
# greedy sampling the output must be bit-identical to the same model without it. GLM-5.3-Flash's MTP head
# fails that test: 11 of 12 greedy completions diverged, and the divergence starts within ~100 characters.
# That makes its 0.060 task-accuracy gap real rather than sampling noise, and it means every speculation
# result in this campaign is suspect until checked - we have been reporting drafters as free latency wins.
#
# Usage: bash specsweep.sh <listfile>
#   listfile lines:  tag|dir|tp|linear:moe|SPEC_JSON|extra serve args...
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
mkdir -p "$P" "$S" "$R/spec"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$B/hardkill.sh"
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 40000 ]; then UTIL=0.88; MAXLEN=32768; else UTIL=0.92; MAXLEN=40960; fi
PROFILES=${PROFILES:-$B/lists/profiles.tsv}

prof_for(){ awk -F'\t' -v n="$(basename "$1")" '!/^#/ && NF>=3 { if (n ~ $1) { print $2; exit } }' "$PROFILES"; }

launch(){ # tag dir tp lin moe spec extra...
  local tag="$1" dir="$2" tp="$3" lin="$4" moe="$5" spec="$6"; shift 6
  kill_all
  local prof; prof=$(prof_for "$dir")
  {
    echo '#!/usr/bin/env bash'
    echo 'export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1'
    echo 'export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a'
    echo 'export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=4 NVCC_THREADS=2'
    # ONE server on the first GPU group: this measures correctness, not throughput, so a single replica is
    # enough and leaves the comparison free of any batching difference between the two arms.
    printf 'CUDA_VISIBLE_DEVICES=%s vllm serve %q --served-model-name m --host 0.0.0.0 --port 8000 --tensor-parallel-size %d --kv-cache-dtype fp8 --max-model-len %d --max-num-seqs 8 --gpu-memory-utilization %s --trust-remote-code --disable-custom-all-reduce --no-enable-flashinfer-autotune --disable-uvicorn-access-log' \
      "$(seq -s, 0 $((tp-1)))" "$dir" "$tp" "$MAXLEN" "$UTIL"
    [ "$lin" != "-" ] && printf ' --kernel-config.linear_backend %q' "$lin"
    [ "$moe" != "-" ] && printf ' --moe-backend %q' "$moe"
    [ -n "$spec" ] && printf " --speculative-config '%s'" "$spec"
    for a in "$@"; do printf ' %q' "$a"; done
    [ -n "$prof" ] && printf ' %s' "$prof"
    printf ' > %s/%s.log 2>&1\n' "$S" "$tag"
  } > "$B/l_sp.sh"
  chmod +x "$B/l_sp.sh"
  tmux new-session -d -s srv "bash $B/l_sp.sh"
  local t=0
  while [ "$t" -lt 1800 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { log "    $tag healthy ${t}s"; return 0; }
    tmux has-session -t =srv 2>/dev/null || break
    sleep 10; t=$((t+10))
  done
  log "    $tag FAILED after ${t}s: $(grep -ohE 'ValueError: [^\"]{0,90}|RuntimeError: [^\"]{0,90}' "$S/$tag.log" 2>/dev/null | head -1)"
  return 1
}

LIST="${1:-}"; [ -f "$LIST" ] || { echo "usage: specsweep.sh <listfile>"; exit 1; }
while IFS='|' read -r tag dir tp combo spec rest; do
  case "$tag" in ""|\#*) continue;; esac
  [ -d "$dir" ] || { log "SKIP $tag (no weights)"; continue; }
  lin="${combo%%:*}"; moe="${combo##*:}"
  IFS='|' read -ra EXTRA <<< "${rest:-}"
  log "########## $tag : is $(echo "$spec" | grep -oE '"method":"[a-z0-9_]+"' || echo speculation) lossless? ##########"
  if launch "${tag}_specbase" "$dir" "$tp" "$lin" "$moe" "" "${EXTRA[@]+"${EXTRA[@]}"}"; then
    $CLEAN python3 "$B/specdiff.py" capture http://127.0.0.1:8000 m "$R/spec/${tag}_base.json" >/dev/null 2>&1
  else continue; fi
  if launch "${tag}_specspec" "$dir" "$tp" "$lin" "$moe" "$spec" "${EXTRA[@]+"${EXTRA[@]}"}"; then
    $CLEAN python3 "$B/specdiff.py" capture http://127.0.0.1:8000 m "$R/spec/${tag}_spec.json" >/dev/null 2>&1
    curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null \
      | grep -E "^vllm:spec_decode_num_(draft_tokens|accepted_tokens)_total" | sed 's/^/    /'
  else continue; fi
  $CLEAN python3 "$B/specdiff.py" compare "$R/spec/${tag}_base.json" "$R/spec/${tag}_spec.json" 2>&1 | tail -4 | sed 's/^/    /'
done < "$LIST"
log "SPECSWEEP DONE"
kill_all
