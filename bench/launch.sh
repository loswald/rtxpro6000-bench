#!/usr/bin/env bash
# =============================================================================
# bench/launch.sh <cell> [--no-wait] [--no-proxy] [--no-smoke] [--no-prefetch] [--dry-run]
#
# Starts the serving cell described by cells/<cell>.env inside a tmux session
# (bench_<cell>), one tmux window per server process, and blocks until every
# server answers GET /health (or HEALTH_TIMEOUT seconds elapse / a server dies).
#
# Writes to results/<cell>[__$RUN_TAG]/ :
#   server_p<port>.log/.sh/.pid/.exit  per-server log, exact command (runnable script), pid, exit code
#   server.env                      resolved environment given to the servers
#   gpus.txt versions.txt engine_version.txt
#   launch.json                     status, seconds_to_ready, KV-cache capacity lines, hardware
#                                   decisions (p2p_ok, custom_allreduce, nccl_p2p_disable,
#                                   acs_suspected, pessimistic_tp), model path, server argv
#   smoke.json                      one real completion after /health
#   loadtest.json                   (LOADTEST_ONLY=1 cells) ok|fail + error excerpt
#   proxy.log                       (REPLICAS>1) rr_proxy.py log
#
# Weights: $MODELS_DIR/<basename of MODEL> when that directory exists (plain
# `hf download --local-dir` layout, see bench/prefetch.sh), else the HF id.
#
# Env overrides (see bench/env.sh):  RUN_TAG, MAX_NUM_BATCHED_TOKENS, KV_CACHE_DTYPE,
#   GPU_MEM_UTIL, MAX_MODEL_LEN, ENABLE_EP, MOE_BACKEND, HEALTH_TIMEOUT, ENGINE=sglang (A/B),
#   CUSTOM_ALLREDUCE=1 (A/B; default OFF), FORCE_NCCL_P2P_DISABLE=1 (A/B only; never automatic),
#   ENABLE_SLEEP_MODE=1 (train co-tenancy cell), SPEC_CONFIG='{...}' (spec decoding is OFF
#   unless a cell or the command line sets it).
# --dry-run prints the resolved server command(s) and exits: no tmux, nothing under results/.
# Host RAM (decision 8): WARN only, never die.  VLLM_PLE_CPU_OFFLOAD (Qwen3.8-Flash-Next recipe env)
#   is NOT in vLLM main's envs.py -- UNVERIFIED (2026-09-02); the PLE (N-gram) cache is TP-replicated
#   (vllm/models/qwen4_exp/nvidia/ple_layer.py), so the estimate is PLE_TABLE_GB x TP x DP x replicas,
#   max'ed with the cell's HOST_RAM_NEEDED_GB hint.
# acs_suspected / pessimistic_tp are null in launch.json when results/hw/decisions.env is missing.
# All `vllm serve` flags verified against vllm main 2026-09-02 (see bench/env.sh).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

usage() {  # header block (line 3 .. the closing # ===== line)
  sed -n '3,/^# =\{5,\}/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac
CELL="$1"; shift
NO_WAIT=0; NO_SMOKE=0; NO_PREFETCH=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-wait)     NO_WAIT=1 ;;
    --no-proxy)    USE_PROXY=0 ;;
    --no-smoke)    NO_SMOKE=1 ;;
    --no-prefetch) NO_PREFETCH=1 ;;
    --dry-run)     DRY=1 ;;
    -h|--help)     usage ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

load_cell "$CELL"
export IMAGE_HINT="${IMAGE_HINT:-}" RUN_TAG="${RUN_TAG:-}" SPEC_CONFIG="${SPEC_CONFIG:-}" \
       ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE:-0}" COMPILATION_CONFIG
warn_if_no_decisions

# ---- advisory checks (never fatal) ----------------------------------------------------
# DP rank i gets the i-th TP-sized slice of CUDA_VISIBLE_DEVICES.  On this box GPUs 0-1 and
# 2-3 share a PCIe switch (PIX) and ACS makes the same-switch pair SLOWER than a cross-switch
# pair, so DP2xTP2 cells should pair cross-switch: GPU_IDS=0,2,1,3.
if [ "$REPLICAS" -eq 1 ] && [ "$DP" -gt 1 ] && [ "$TP" -gt 1 ] && [ "$ACS_SUSPECTED" = 1 ]; then
  case "$GPU_IDS" in
    0,1,2,3|1,0,3,2|2,3,0,1|3,2,1,0)
      log "WARN: DP${DP}xTP${TP} with GPU_IDS=$GPU_IDS pairs SAME-switch GPUs (0-1, 2-3 are PIX); with ACS on the host cross-switch pairs are faster -> set GPU_IDS=0,2,1,3 in cells/$CELL.env" ;;
  esac
fi
if [ -n "$SPEC_CONFIG" ]; then
  log "WARN: SPEC_CONFIG is set — the throughput campaign runs speculative decoding OFF; treating this launch as an explicit A/B (recorded in launch.json)"
fi
if [ "$CELL_PESSIMISTIC_TP" = 1 ]; then
  log "note: TP=$TP on a box with PESSIMISTIC_TP=1 (ACS suspected) -> results carry pessimistic_tp=1 (dagger in summarise.py)"
elif [ "$TP" -gt 1 ] && [ -z "$CELL_PESSIMISTIC_TP" ]; then
  log "note: TP=$TP but PESSIMISTIC_TP is unknown (no decisions.env) -> launch.json/meta carry pessimistic_tp=null, summaries show '?'"
fi

# Host RAM: warn (never die) when the cell needs more than `free -g` reports available.
RAM_NEED_GB=0; RAM_WHY=""; RAM_AVAIL_GB=""; RAM_WARNING=""
ram_check() {
  if [ "${VLLM_PLE_CPU_OFFLOAD:-0}" = 1 ]; then
    # The PLE cache is TP-replicated (vllm/models/qwen4_exp/nvidia/ple_layer.py): if the recipe env is
    # honoured, EVERY worker (TP x DP x replicas) parks its own ~PLE_TABLE_GB copy in host RAM.
    # VLLM_PLE_CPU_OFFLOAD itself is absent from vLLM main's envs.py -- UNVERIFIED (2026-09-02).
    RAM_NEED_GB=$(( PLE_TABLE_GB * TP * DP * REPLICAS ))
    RAM_WHY="VLLM_PLE_CPU_OFFLOAD=1: ~${PLE_TABLE_GB} GB N-gram table per TP rank x TP=${TP} x DP=${DP} x replicas=${REPLICAS}"
  fi
  if [ "${HOST_RAM_NEEDED_GB:-0}" -gt "$RAM_NEED_GB" ]; then
    RAM_NEED_GB=$HOST_RAM_NEEDED_GB; RAM_WHY="cell HOST_RAM_NEEDED_GB=$HOST_RAM_NEEDED_GB"
  fi
  if [ "$ENABLE_SLEEP_MODE" = 1 ] && [ "$MODEL_SOURCE" = local ]; then
    local w; w="$(du -sBG "$MODEL_PATH" 2>/dev/null | cut -f1 | tr -d 'G' || true)"
    if [ -n "${w:-}" ] && [ "$w" -gt "$RAM_NEED_GB" ] 2>/dev/null; then
      RAM_NEED_GB=$w; RAM_WHY="sleep mode level 1 offloads ~${w} GB of weights to host RAM"
    fi
  fi
  [ "$RAM_NEED_GB" -gt 0 ] || return 0
  if have free; then RAM_AVAIL_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')"; fi
  local need_hr=$(( RAM_NEED_GB + RAM_NEED_GB / 10 + 4 ))   # +10 % + 4 GB runtime headroom
  if [ -n "$RAM_AVAIL_GB" ] && [ "$RAM_AVAIL_GB" -lt "$need_hr" ] 2>/dev/null; then
    RAM_WARNING="need ~${need_hr} GB host RAM ($RAM_WHY) but only ${RAM_AVAIL_GB} GB available (total ${HOST_RAM_GB} GB)"
    log "WARN: $RAM_WARNING — launching anyway; watch for OOM-kills in the server log"
  else
    log "host RAM: need ~${need_hr} GB ($RAM_WHY); available ${RAM_AVAIL_GB:-?} GB of ${HOST_RAM_GB} GB total — OK"
  fi
}
ram_check
export RAM_NEED_GB RAM_WHY RAM_AVAIL_GB RAM_WARNING

# ---- build the server command --------------------------------------------------------
build_server_cmd() {  # $1 = port  -> SERVER_CMD array
  local port="$1"
  if [ "$ENGINE" = "sglang" ]; then
    local kv=auto
    case "$KV_CACHE_DTYPE" in fp8*) kv=fp8_e4m3 ;; esac
    SERVER_CMD=( python3 -m sglang.launch_server
      --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
      --host 0.0.0.0 --port "$port" --tp "$TP" --dp "$DP"
      --context-length "$MAX_MODEL_LEN" --max-running-requests "$MAX_NUM_SEQS"
      --chunked-prefill-size "$MAX_NUM_BATCHED_TOKENS" --mem-fraction-static "$GPU_MEM_UTIL"
      --kv-cache-dtype "$kv" --trust-remote-code )
    if [ "$ENABLE_EP" = 1 ]; then SERVER_CMD+=( --ep-size "$TP" ); fi
    SERVER_CMD+=( ${SGLANG_EXTRA_ARGS[@]+"${SGLANG_EXTRA_ARGS[@]}"} )
  else
    SERVER_CMD=( vllm serve "$MODEL_PATH"
      --port "$port" --served-model-name "$SERVED_MODEL_NAME"
      --tensor-parallel-size "$TP" --data-parallel-size "$DP"
      --kv-cache-dtype "$KV_CACHE_DTYPE"
      --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --compilation-config "$COMPILATION_CONFIG"
      "${VLLM_COMMON_ARGS[@]}" )
    # Decision 2: custom all-reduce OFF unless CUSTOM_ALLREDUCE=1 (A/B).
    if [ "$CUSTOM_ALLREDUCE" != 1 ]; then SERVER_CMD+=( --disable-custom-all-reduce ); fi
    if [ "$ENABLE_EP" = 1 ]; then SERVER_CMD+=( --enable-expert-parallel ); fi
    if [ -n "$SPEC_CONFIG" ]; then SERVER_CMD+=( --speculative-config "$SPEC_CONFIG" ); fi
    if [ "$ENABLE_SLEEP_MODE" = 1 ]; then SERVER_CMD+=( --enable-sleep-mode ); fi
    SERVER_CMD+=( ${CELL_EXTRA_ARGS[@]+"${CELL_EXTRA_ARGS[@]}"} )
  fi
}

log "cell=$CELL_NAME engine=$ENGINE model=$MODEL ($MODEL_SOURCE: $MODEL_PATH${MODEL_DIR_STATE:+, dir=$MODEL_DIR_STATE}) TP=$TP DP=$DP replicas=$REPLICAS gpus=$GPU_IDS"
log "kv=$KV_CACHE_DTYPE max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS mnbt=$MAX_NUM_BATCHED_TOKENS gpu_mem_util=$GPU_MEM_UTIL EP=$ENABLE_EP spec=${SPEC_CONFIG:-off}"
log "hw: P2P_OK=$P2P_OK NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} custom_allreduce=$CUSTOM_ALLREDUCE ACS_SUSPECTED=${ACS_SUSPECTED:-unknown} pessimistic_tp=${CELL_PESSIMISTIC_TP:-unknown} (decisions: $HW_DECISIONS_SOURCE)"

if [ "$DRY" = 1 ]; then
  for ((i = 0; i < REPLICAS; i++)); do
    build_server_cmd "${PORTS[$i]}"
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$(gpu_list_for_replica "$i")"
    printf '%q ' "${SERVER_CMD[@]}"; echo
  done
  if [ "$REPLICAS" -gt 1 ] && [ "$USE_PROXY" = 1 ]; then
    echo "python3 $HERE/rr_proxy.py --port $PROXY_PORT --backends $(printf '127.0.0.1:%s,' "${PORTS[@]}" | sed 's/,$//')"
  fi
  log "dry run: results dir would be $RESULTS_DIR"
  exit 0
fi

have tmux || die "tmux not found — run bench/setup_engine.sh (apt-get install -y tmux)"
have curl || die "curl not found"
mkdir -p "$RESULTS_DIR"

# ---- stop a previous instance of this cell and rotate its logs -------------------
"$HERE/stop.sh" "$CELL" --quiet || true
shopt -s nullglob
old=()
for f in "$RESULTS_DIR"/server_p*.log "$RESULTS_DIR"/server_p*.exit "$RESULTS_DIR"/server_p*.pid "$RESULTS_DIR"/server_p*.sh \
         "$RESULTS_DIR"/proxy.log "$RESULTS_DIR"/proxy.sh; do
  [ -e "$f" ] && old+=( "$f" )   # proxy.* are literal names: nullglob does not drop them when absent
done
if [ ${#old[@]} -gt 0 ]; then
  prev="$RESULTS_DIR/prev_$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$prev"; mv "${old[@]}" "$prev"/
  log "rotated previous server logs to $prev"
fi
shopt -u nullglob

# ---- snapshots ---------------------------------------------------------------------
ENGINE_VERSION="$(engine_version)"; export ENGINE_VERSION
echo "$ENGINE_VERSION" > "$RESULTS_DIR/engine_version.txt"
{ nvidia-smi -L; echo; nvidia-smi --query-gpu=index,name,memory.total,power.limit,driver_version,pstate --format=csv; } \
  > "$RESULTS_DIR/gpus.txt" 2>&1 || true
{ pip list 2>/dev/null || python3 -m pip list 2>/dev/null || uv pip list --system 2>/dev/null || true; } \
  | grep -iE '^(vllm|sglang|flashinfer|b12x|torch|transformers|triton|nvidia-nccl|nvidia-cudnn|xformers|flash|hf-transfer|huggingface)' \
  > "$RESULTS_DIR/versions.txt" || true
# Resolved environment handed to every server (shell-quoted KEY=VALUE; sourced by _run_server.sh).
# CUDA_VISIBLE_DEVICES is deliberately excluded: it is set per replica on the command line.
: > "$RESULTS_DIR/server.env"
for name in $(compgen -e | grep -E '^(VLLM_|NCCL_|FLASHINFER_|TORCH_CUDA|HF_|TOKENIZERS|PYTORCH_|TRITON_|SGLANG_|SGL_|OMP_|CUDA_)' | grep -v '^CUDA_VISIBLE_DEVICES$' || true); do
  printf '%s=%q\n' "$name" "${!name}" >> "$RESULTS_DIR/server.env"
done
log "engine version $ENGINE_VERSION; results=$RESULTS_DIR"
if [ -n "$IMAGE_HINT" ]; then log "image hint: $IMAGE_HINT"; fi

# ---- weights: local directory preferred; download into it only when missing/partial --------
if [ "$NO_PREFETCH" = 0 ] && { [ "$MODEL_SOURCE" = hub ] || [ "$MODEL_DIR_STATE" = partial ]; }; then
  log "weights not complete under $MODELS_DIR ($MODEL_DIR_STATE) -> bench/prefetch.sh $MODEL"
  if "$HERE/prefetch.sh" "$MODEL"; then
    resolve_model_path
    log "weights now: $MODEL_SOURCE $MODEL_PATH ($MODEL_DIR_STATE)"
  else
    log "WARN: prefetch failed; the server will try to load '$MODEL_PATH' itself"
  fi
fi
if [ "$MODEL_SOURCE" = local ] && [ "$MODEL_DIR_STATE" = partial ]; then
  log "WARN: $MODEL_PATH looks partial (no .complete marker / *.incomplete present); the server may fail to load"
fi

# ---- tmux session: one window per server (+ proxy) ---------------------------------------
tmux new-session -d -s "$SESSION" -n ctl bash
T0=$(date +%s)
for ((i = 0; i < REPLICAS; i++)); do
  port=${PORTS[$i]}
  gpus="$(gpu_list_for_replica "$i")"
  build_server_cmd "$port"
  logf="$RESULTS_DIR/server_p${port}.log"
  # The exact command goes into a small bash script (readable record + no quoting
  # ambiguity between bash %q and tmux's default shell).  BENCH_CELL marks the whole
  # process tree for stop.sh.
  runner="$RESULTS_DIR/server_p${port}.sh"
  {
    echo '#!/usr/bin/env bash'
    echo "# generated by launch.sh $(date -Is) -- cell $CELL_NAME replica $i (GPU $gpus, port $port)"
    printf 'export BENCH_CELL=%q BENCH_PORT=%q\n' "$CELL_NAME" "$port"
    printf 'exec bash %q %q %q env %q' "$HERE/_run_server.sh" "$logf" "$RESULTS_DIR/server.env" "CUDA_VISIBLE_DEVICES=$gpus"
    printf ' %q' "${SERVER_CMD[@]}"
    echo
  } > "$runner"
  chmod +x "$runner"
  tmux new-window -t "$SESSION" -n "p$port" "bash '$runner'"
  # dead panes stay visible for post-mortem (window option; set per window)
  tmux set-option -w -t "$SESSION:p$port" remain-on-exit on 2>/dev/null || true
  log "replica $i: GPU(s) $gpus port $port  log=$logf  cmd=$runner"
  if [ "$REPLICAS" -gt 1 ] && [ "$i" -lt $((REPLICAS - 1)) ]; then sleep "${REPLICA_STAGGER_S:-3}"; fi
done

if [ "$REPLICAS" -gt 1 ] && [ "$USE_PROXY" = 1 ]; then
  backends="$(printf '127.0.0.1:%s,' "${PORTS[@]}")"; backends="${backends%,}"
  {
    echo '#!/usr/bin/env bash'
    printf 'export BENCH_CELL=%q\n' "$CELL_NAME"
    printf 'exec python3 %q --port %q --backends %q --log %q\n' "$HERE/rr_proxy.py" "$PROXY_PORT" "$backends" "$RESULTS_DIR/proxy.log"
  } > "$RESULTS_DIR/proxy.sh"
  chmod +x "$RESULTS_DIR/proxy.sh"
  tmux new-window -t "$SESSION" -n proxy "bash '$RESULTS_DIR/proxy.sh'"
  tmux set-option -w -t "$SESSION:proxy" remain-on-exit on 2>/dev/null || true
  log "round-robin proxy on :$PROXY_PORT -> $backends (for gates/ad-hoc; sweep.sh defaults to per-port fan-out)"
fi

if [ "$NO_WAIT" = 1 ]; then
  log "started (--no-wait). Attach: tmux attach -t $SESSION"
  exit 0
fi

# ---- wait for /health on every port --------------------------------------------------------
error_excerpt() {  # $1 = log  (never fails: set -e + pipefail would otherwise abort on a grep miss)
  local out
  out="$( { tail -n 200 "$1" 2>/dev/null \
    | grep -iE 'error|exception|traceback|not supported|unsupported|sm_120|compute capability|out of memory|OOM|No module|No such file|not found|invalid choice|unrecognized|Killed' \
    | tail -n 15 | tr '\n' ' ' | cut -c1-2000; } || true )"
  if [ -z "$out" ]; then out="$( { tail -n 5 "$1" 2>/dev/null | tr '\n' ' ' | cut -c1-2000; } || true )"; fi
  printf '%s' "$out"
}

log "waiting for /health on ports ${PORTS[*]} (timeout ${HEALTH_TIMEOUT}s) ..."
STATUS=fail; DEAD_PORT=""
deadline=$((T0 + HEALTH_TIMEOUT))
while :; do
  all_ok=1
  for port in "${PORTS[@]}"; do
    if ! curl -fsS -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then all_ok=0; break; fi
  done
  if [ "$all_ok" = 1 ]; then STATUS=ok; break; fi
  for port in "${PORTS[@]}"; do
    if [ -f "$RESULTS_DIR/server_p${port}.exit" ]; then DEAD_PORT=$port; break; fi
  done
  if [ -n "$DEAD_PORT" ]; then STATUS=died; break; fi
  if [ "$(date +%s)" -ge "$deadline" ]; then STATUS=timeout; break; fi
  sleep 10
done
SECS=$(( $(date +%s) - T0 ))

KV_LINE="$(grep -hoE 'GPU KV cache size: [0-9,]+ tokens' "$RESULTS_DIR"/server_p*.log 2>/dev/null | head -n1 || true)"
MAXCONC_LINE="$(grep -hoE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$RESULTS_DIR"/server_p*.log 2>/dev/null | head -n1 || true)"
ERR=""
if [ "$STATUS" != ok ]; then
  ERR="$(error_excerpt "$RESULTS_DIR/server_p${DEAD_PORT:-${PORTS[0]}}.log")"
fi

write_launch_json() {  # $1 = output file; server argv of the last replica follows
  local out="$1"; shift
  LAUNCH_STATUS="$STATUS" LAUNCH_SECONDS="$SECS" LAUNCH_ERROR="$ERR" KV_LINE="$KV_LINE" MAXCONC_LINE="$MAXCONC_LINE" \
  PORTS_STR="${PORTS[*]}" NCCL_P2P_DISABLE_EFF="${NCCL_P2P_DISABLE:-0}" python3 - "$out" "$@" <<'PY'
import datetime, json, os, sys
e = os.environ
def num(v):
    for t in (int, float):
        try: return t(v)
        except (TypeError, ValueError): pass
    return v
def flag(v):
    """hardware decision flag: 0/1, or null when bench/env.sh left it empty (= unknown, no decisions.env)."""
    return None if v in (None, "") else num(v)
keys = ["CELL_NAME","ENGINE","ENGINE_VERSION","MODEL","MODEL_PATH","MODEL_SOURCE","SERVED_MODEL_NAME","TP","DP","REPLICAS","GPU_IDS",
        "KV_CACHE_DTYPE","MAX_MODEL_LEN","MAX_NUM_SEQS","MAX_NUM_BATCHED_TOKENS","GPU_MEM_UTIL","ENABLE_EP","ENABLE_SLEEP_MODE",
        "SPEC_CONFIG","COMPILATION_CONFIG","CUSTOM_ALLREDUCE","P2P_OK","P2P_DISABLED","ACS_SUSPECTED","HOST_RAM_GB",
        "RAM_NEED_GB","RAM_AVAIL_GB","RAM_WARNING","RUN_TAG","IMAGE_HINT","LOADTEST_ONLY","NCCL_P2P_LEVEL"]
# always strings: a single-GPU cell has GPU_IDS="0" (must not become the int 0), version strings, names, tags
STR_KEYS = {"GPU_IDS","CELL_NAME","ENGINE","ENGINE_VERSION","MODEL","MODEL_PATH","MODEL_SOURCE","SERVED_MODEL_NAME",
            "KV_CACHE_DTYPE","RUN_TAG","IMAGE_HINT","SPEC_CONFIG","COMPILATION_CONFIG","RAM_WARNING","NCCL_P2P_LEVEL"}
d = {k.lower(): (e.get(k, "") if k in STR_KEYS else num(e.get(k, ""))) for k in keys}
d.update({
    "status": e["LAUNCH_STATUS"],
    "seconds_to_ready": num(e["LAUNCH_SECONDS"]),
    "ports": [int(p) for p in e["PORTS_STR"].split()],
    "kv_cache_line": e.get("KV_LINE", ""),
    "max_concurrency_line": e.get("MAXCONC_LINE", ""),
    "error_excerpt": e.get("LAUNCH_ERROR", ""),
    # hardware decision contract (bench/env.sh <- results/hw/decisions.env); null = unknown (file missing)
    "nccl_p2p_disable": num(e.get("NCCL_P2P_DISABLE_EFF", "0")),
    "acs_suspected": flag(e.get("ACS_SUSPECTED")),
    "pessimistic_tp": flag(e.get("CELL_PESSIMISTIC_TP")),          # THIS cell: TP>1 and PESSIMISTIC_TP=1 (TP1 -> 0)
    "hw_pessimistic_tp": flag(e.get("PESSIMISTIC_TP")),            # raw box-level flag
    "hw_decisions_file": e.get("HW_DECISIONS_SOURCE", "missing"),
    "hw_notes": e.get("HW_NOTES", ""),
    "spec_decoding": "off" if not e.get("SPEC_CONFIG") else "on",
    "server_argv": sys.argv[2:],
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
})
json.dump(d, open(sys.argv[1], "w"), indent=2)
PY
}
write_launch_json "$RESULTS_DIR/launch.json" "${SERVER_CMD[@]}" || log "WARN: could not write launch.json"

if [ "$LOADTEST_ONLY" = 1 ]; then
  # Attempt-to-load cell: record the outcome, tear down, never fail the pipeline.
  cp "$RESULTS_DIR/launch.json" "$RESULTS_DIR/loadtest.json"
  tail -n 400 "$RESULTS_DIR/server_p${DEAD_PORT:-${PORTS[0]}}.log" > "$RESULTS_DIR/loadtest_log_tail.txt" 2>/dev/null || true
  if [ "$STATUS" = ok ]; then
    log "LOADTEST: model loaded in ${SECS}s. $KV_LINE"
    if [ "$NO_SMOKE" = 0 ]; then
      curl -sS -m 300 "http://127.0.0.1:${PORTS[0]}/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
        > "$RESULTS_DIR/smoke.json" 2>&1 || true
    fi
  else
    log "LOADTEST: status=$STATUS after ${SECS}s. Error excerpt: ${ERR:-<none captured; see loadtest_log_tail.txt>}"
  fi
  "$HERE/stop.sh" "$CELL" --quiet || true
  log "loadtest recorded in $RESULTS_DIR/loadtest.json"
  exit 0
fi

case "$STATUS" in
  ok) ;;
  died)
    log "server on port $DEAD_PORT exited during startup (exit=$(cat "$RESULTS_DIR/server_p${DEAD_PORT}.exit" 2>/dev/null || echo '?'))."
    log "excerpt: $ERR"
    log "full log: $RESULTS_DIR/server_p${DEAD_PORT}.log   (tmux attach -t $SESSION)"
    exit 1 ;;
  timeout)
    log "timed out after ${SECS}s waiting for /health; servers left running. tmux attach -t $SESSION"
    exit 1 ;;
esac

if [ "$REPLICAS" -gt 1 ] && [ "$USE_PROXY" = 1 ]; then
  for _ in 1 2 3 4 5 6; do
    if curl -fsS -m 5 "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1; then break; fi
    sleep 2
  done
fi

# ---- smoke request ------------------------------------------------------------------------------
if [ "$NO_SMOKE" = 0 ]; then
  if [ "$BENCH_ENDPOINT" = "/v1/chat/completions" ]; then
    body="{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32}"
  else
    body="{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"
  fi
  curl -sS -m 300 "http://127.0.0.1:${PORTS[0]}$BENCH_ENDPOINT" -H 'Content-Type: application/json' -d "$body" \
    > "$RESULTS_DIR/smoke.json" 2>&1 || log "WARN: smoke request failed (see smoke.json)"
fi

endpoints="$(printf 'http://127.0.0.1:%s ' "${PORTS[@]}")"
if [ "$REPLICAS" -gt 1 ] && [ "$USE_PROXY" = 1 ]; then endpoints+="proxy=http://127.0.0.1:$PROXY_PORT"; fi
log "READY in ${SECS}s. ${KV_LINE:+$KV_LINE. }${MAXCONC_LINE}"
log "endpoints: $endpoints"
log "next: bench/sweep.sh $CELL      stop: bench/stop.sh $CELL      logs: tmux attach -t $SESSION"
