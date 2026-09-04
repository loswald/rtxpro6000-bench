#!/usr/bin/env bash
# Kernel-sweep retry harness (host-agnostic: reads GPU count and VRAM from the driver).
#
# Written after a night in which 11 of ~30 arms died at engine init for three reasons, all of them a
# hard-coded kernel choice rather than a property of the model:
#
#   A. `--moe-backend flashinfer_cutlass` rejects NVFP4 MoE checkpoints whose expert scales are
#      u8/f8e4m3fn on GroupShape(row=1, col=16) — "does not support the deployment configuration since
#      kernel does not support quantization scheme QuantKey(u8,scale(f8e4m3fn,static,GroupShape(1,16)))".
#      Killed Ling-3.0-flash (both arms), Nemotron-3.5-Lightning, Qwen3.6-35B on the 5090 box. The same
#      backend also rejects Step-3.7-Flash for a different reason: "does not support MoEActivation.SW".
#   B. Forcing `--kernel-config.linear_backend b12x` on gemma-4-26B raises
#      "dense_gemm launch is unsupported with Float4E2M1FN, ... (128,128), 2816, 2112" — the CuTe DSL
#      dense GEMM has no tile for those shapes.
#   C. Fixed `--max-num-seqs 512` exceeds a hybrid model's Mamba cache blocks (Nemotron-3-Super: 333),
#      and 0.92 utilisation OOMs a 32 GB card for a 17 GB NVFP4 checkpoint at 32k context.
#
# So: never hard-code one backend. For each model, walk a candidate list, keep every combination that
# actually serves, benchmark all of them, and record the kernel line the engine selected. That is both
# the fix and the per-model kernel sweep the campaign owes.
#
# Usage:  bash ksweep.sh <listfile>     # listfile lines: tag|dir|tp|combos|extra args...
#         MODELS=... bash ksweep.sh     # or set MODELS to a here-doc string
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$B/hardkill.sh"
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# 32 GB cards need headroom the 96 GB cards do not: cudagraph capture OOMed at 0.92/32k/256 seqs.
if [ "$VRAM" -lt 40000 ]; then UTIL=0.88; MAXLEN=32768; SEQS=128; else UTIL=0.94; MAXLEN=40960; SEQS=512; fi
log "host: ${NGPU}x GPU, ${VRAM} MiB each -> util $UTIL, max-model-len $MAXLEN, max-num-seqs $SEQS"

# A full disk does not announce itself: the engine fails to write its compile cache and its own log, so
# every arm dies silently and looks like a model problem. Check before each launch, not once at the start.
FREE_GB(){ df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9'; }
disk_ok(){
  local f; f=$(FREE_GB)
  [ -n "$f" ] && [ "$f" -ge 15 ] && return 0
  log "  DISK ${f:-?} GB free: refusing to launch (an engine that cannot write its cache fails in ways that look like the model's fault)"
  return 1
}

# Trust our own downloader's sentinel where it exists; otherwise verify every shard the index names.
# A directory with a config and two small files passed the old check while holding 0.7 GB of a 171 GB model.
weights_ok(){ # dir
  local d="$1"
  [ -f "$d/config.json" ] || return 1
  find "$d" \( -name '*.incomplete' -o -name '*.part' \) 2>/dev/null | grep -q . && return 1
  [ -f "$d/.dl_complete" ] && return 0
  local idx="$d/model.safetensors.index.json"
  if [ -f "$idx" ]; then
    python3 - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]
try:
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
except Exception:
    sys.exit(1)
files = set(wm.values())
sys.exit(0 if files and all(os.path.exists(os.path.join(d, f)) for f in files) else 1)
PY
    return $?
  fi
  compgen -G "$d/*.safetensors" >/dev/null || compgen -G "$d/*.bin" >/dev/null || compgen -G "$d/*.gguf" >/dev/null
}

# Per-model serving profile from lists/profiles.tsv: the vendor's own parser, template kwargs and sampling.
# Serving a model outside its vendor recipe is not a small effect - it is how this campaign spent a day
# grading models on their own chain-of-thought.
PROFILES=${PROFILES:-$B/lists/profiles.tsv}
profile_field(){ # dir field(2=serve,3=eval)
  local name; name=$(basename "$1")
  [ -f "$PROFILES" ] || return 0
  awk -F'\t' -v n="$name" -v f="$2" '
    !/^#/ && NF>=3 { if (n ~ $1) { print $f; exit } }' "$PROFILES"
}

serve(){ # tag dir tp linear moe [extra...]
  local tag="$1" dir="$2" tp="$3" lin="$4" moe="$5"; shift 5
  local n=$(( NGPU / tp ))
  local prof; prof=$(profile_field "$dir" 2)
  kill_all
  {
    echo '#!/usr/bin/env bash'
    echo 'export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1'
    echo 'export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a'
    echo 'export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=4 NVCC_THREADS=2'
    [ -n "${EXTRA_ENV:-}" ] && echo "export $EXTRA_ENV"
    for i in $(seq 0 $((n-1))); do
      local gpus; gpus=$(seq -s, $((i*tp)) $((i*tp+tp-1)))
      printf 'CUDA_VISIBLE_DEVICES=%s vllm serve %q --served-model-name m --host 0.0.0.0 --port %d --tensor-parallel-size %d --kv-cache-dtype fp8 --max-model-len %d --max-num-seqs %d --max-num-batched-tokens 8192 --gpu-memory-utilization %s --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce --no-enable-flashinfer-autotune --disable-uvicorn-access-log' \
        "$gpus" "$dir" $((8000+i)) "$tp" "$MAXLEN" "$SEQS" "$UTIL"
      [ "$lin" != "-" ] && printf ' --kernel-config.linear_backend %q' "$lin"
      [ "$moe" != "-" ] && printf ' --moe-backend %q' "$moe"
      for a in "$@"; do printf ' %q' "$a"; done
      # the vendor profile goes last so a list row can still override any of it
      [ -n "$prof" ] && printf ' %s' "$prof"
      printf ' > %s/%s_p%d.log 2>&1 &\nsleep 2\n' "$S" "$tag" $((8000+i))
    done
    echo 'wait'
  } > "$B/l_ks.sh"
  chmod +x "$B/l_ks.sh"
  tmux new-session -d -s srv "bash $B/l_ks.sh"
  local t=0 ok=0
  # A dead launcher is the only reliable failure signal. On 4 Sept the box filled up, vLLM could not write
  # its own traceback into the server log, the log-scraping fast-fail below never matched, and eight hours
  # went into arms that had already died in the first minute. If the tmux session is gone, so is the run.
  while [ "$t" -lt 2100 ]; do
    ok=1; for i in $(seq 0 $((n-1))); do curl -fsS -m 3 "http://127.0.0.1:$((8000+i))/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "^(ValueError|RuntimeError|NotImplementedError|TypeError)|Engine core initialization failed|CUDA out of memory|does not support|Address already in use" "$S/${tag}_p8000.log" 2>/dev/null && { sleep 15; break; }
    tmux has-session -t =srv 2>/dev/null || { log "    (launcher exited after ${t}s)"; break; }
    sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    local why; why=$(grep -ohE "does not support [^\"]{0,110}|CUDA out of memory[^\"]{0,60}|dense_gemm launch is unsupported[^\"]{0,80}|exceeds available Mamba cache blocks[^\"]{0,50}|ValueError: [^\"]{0,110}|RuntimeError: [^\"]{0,110}" "$S/${tag}_p8000.log" 2>/dev/null | grep -viE "warning" | head -1)
    log "    x lin=$lin moe=$moe : ${why:-unknown (see ${tag}_p8000.log)}"
    return 1
  fi
  log "    v lin=$lin moe=$moe healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
  grep -m3 -ohE "Using [A-Za-z0-9_' ]+ (MoE|Mxfp4|NVFP4)[A-Za-z ]*backend[^,]*|Using [A-Za-z0-9]+ for [A-Z0-9]+ GEMM|Selected [A-Za-z0-9]+Kernel|Using [A-Z_]+ attention backend" "$S/${tag}_p8000.log" | sed 's/^/      kernel: /'
  return 0
}

pt(){ # tag label in out prefix c_per_port nports dir
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 n=$7 dir=$8
  local np=$(( c*8 )) tot=$(( c*n ))
  mkdir -p "$P/$tag"
  for i in $(seq 0 $((n-1))); do
    local p=$((8000+i))
    $CLEAN vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model m --tokenizer "$dir" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((8700+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "$P/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

evalrun(){ # tag n dir  - the six-family quality suite against the servers that are already up
  local tag=$1 n=$2 dir=$3 urls="" i
  local esamp; esamp=$(profile_field "$dir" 3)
  for i in $(seq 0 $((n-1))); do urls="${urls}${urls:+,}http://127.0.0.1:$((8000+i))"; done
  mkdir -p "$R/eval"
  # No meaningful token cap. A truncated answer is not a wrong answer, it is a broken measurement, and it
  # was corrupting these scores: GLM-5.3-Flash lost 51% of the maths items to the cap and Qwen3.8-27B 45%
  # of everything. Caps are now set just under what the context window allows, so the only limit is the
  # TIME budget - and running out of time marks an item skipped, which is excluded from the accuracy, where
  # running out of tokens marked it wrong. Long-context is the one exception: its prompts are up to 32k so
  # its output cap has to stay small, and its answers are a few tokens anyway.
  local caps="${EVAL_CAPS:-math=24576,code=16384,knowledge=16384,ifeval=12288,tools=8192,longctx=2048}"
  [ "$MAXLEN" -lt 40000 ] && caps="${EVAL_CAPS:-math=20480,code=14336,knowledge=14336,ifeval=10240,tools=8192,longctx=2048}"
  $CLEAN python3 "$B/evalsuite/run_eval.py" --tag "$tag" --base-urls "$urls" --model m \
    --out "$R/eval" --gpus "$NGPU" --time-budget "${EVAL_BUDGET:-900}" --concurrency $(( 16 * n )) \
    ${EVAL_REASONING:---reasoning} --max-tokens "${EVAL_MAXTOK:-24576}" --max-tokens-family "$caps" \
    $esamp ${EVAL_ARGS:-} 2>&1 | tail -8 | sed 's/^/    eval: /'
  # $esamp is the vendor's own sampling recipe from lists/profiles.tsv and it comes after --reasoning on
  # purpose: --reasoning sets a house default of T=0.6/top_p=0.95 for every model, and not one vendor here
  # recommends that. Qwen wants T=1.0/top_p=0.95/top_k=20/min_p=0, gemma T=0.0/top_k=64, Ling T=0.85,
  # Hy3 T=0.9/top_p=1.0. top_k and min_p have no slot in the OpenAI schema, so they ride in extra_body.
}

shapes(){ # tag dir n
  local tag=$1 dir=$2 n=$3
  $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
  case "${MODE:-bench}" in
    eval) evalrun "$tag" "$n" "$dir"; return;;
    both) evalrun "$tag" "$n" "$dir";;
  esac
  pt "$tag" router    1024 128 0    $(( 256 / n ))  "$n" "$dir"
  pt "$tag" router    1024 128 0    $(( 1024 / n )) "$n" "$dir"
  pt "$tag" promptopt  512 256 3072 $(( 1024 / n )) "$n" "$dir"
  pt "$tag" judge     4096 512 0    $(( 512 / n ))  "$n" "$dir"
  curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode.*(accepted|draft|emitted)" | head -4 | sed 's/^/    spec: /'
}

# sweep: try every candidate combo; benchmark each one that serves. combos are "linear:moe" pairs,
# comma-separated; "-" means "do not pass the flag" (engine auto-selection).
sweep(){ # tag dir tp combos [extra...]
  local tag="$1" dir="$2" tp="$3" combos="$4"; shift 4
  local n=$(( NGPU / tp ))
  # A directory is not a model: Hugging Face leaves partial blobs under .cache/huggingface/download as
  # *.incomplete, and benchmarking a half-downloaded checkpoint wastes an hour and produces a wrong number.
  weights_ok "$dir" || { log "SKIP $tag (weights absent or incomplete)"; return 1; }
  disk_ok || return 1
  log "########## $tag ($(du -sh "$dir" 2>/dev/null | cut -f1), tp=$tp x$n) : $combos ##########"
  local any=0 i=0
  IFS=',' read -ra CC <<< "$combos"
  for c in "${CC[@]}"; do
    local lin="${c%%:*}" moe="${c##*:}" atag
    atag="${tag}_$(echo "$c" | tr ':' '-' | tr -d '.')"
    case "${MODE:-bench}" in
      eval) [ -f "$R/eval/$atag.json" ] && { log "  $atag already evaluated"; any=1; continue; };;
      *)    [ "$(ls "$P/$atag"/*__judge__*.json 2>/dev/null | wc -l)" -ge "$n" ] && { log "  $atag already measured"; any=1; continue; };;
    esac
    if serve "$atag" "$dir" "$tp" "$lin" "$moe" "$@"; then
      shapes "$atag" "$dir" "$n"; any=1
      [ "${FIRST_ONLY:-0}" = 1 ] && break
    fi
    i=$((i+1))
  done
  kill_all
  [ "$any" = 1 ] || log "  $tag: NO combination served"
  return 0
}

LIST="${1:-}"
if [ -n "$LIST" ] && [ -f "$LIST" ]; then
  # Every field is pipe-separated, extra engine args included, so JSON specs keep their quotes and spaces
  # without passing through eval (the bug that killed the fleet tier: JSON in a heredoc loses its quoting).
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    IFS='|' read -ra F <<< "$line"
    [ "${#F[@]}" -ge 4 ] || { log "SKIP malformed list line: $line"; continue; }
    sweep "${F[0]}" "${F[1]}" "${F[2]}" "${F[3]}" "${F[@]:4}"
  done < "$LIST"
fi
log "KSWEEP DONE"
kill_all
