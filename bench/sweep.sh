#!/usr/bin/env bash
# =============================================================================
# bench/sweep.sh <cell> [shapes] [concurrencies] [options]
#
#   shapes         space/comma list of  router | judge | agent   (default: all three)
#                  router = random 1024 in / 128 out, judge = 4096/512, agent = 32768/2048
#   concurrencies  space/comma list applied to EVERY listed shape.  Default = the per-shape
#                  lists from bench/env.sh (throughput-first, time-budgeted):
#                    router 1 4 8 16 32 64 128 256 | judge 1 4 16 64 128 256 | agent 1 4 16 64
#                  (ROUTER_CONCS / JUDGE_CONCS / AGENT_CONCS override them)
#
# options
#   --dataset-path FILE.jsonl  real prompts instead of random tokens (shape label "custom").
#                              One JSON object per line: {"prompt": "<text>", "output_tokens": <int>}
#   --output-len N             output tokens for --dataset-path runs (default 256).  -1 = honour each
#                              line's "output_tokens" (= `--custom-output-len -1`; every line must
#                              then carry the field, vllm raises otherwise)
#   --shape-label NAME         label for --dataset-path runs (default: custom)
#   --via-proxy                x4 cells only: drive the rr_proxy on :$PROXY_PORT with exact
#                              concurrency C (default is per-port fan-out, see below)
#   --no-warmup                skip the warm-up run
#   --dry-run                  print the plan + bench commands; touches nothing, needs no server
#
# Per run: `vllm bench serve` (fallback benchmark_serving.py) with --request-rate inf,
# --max-concurrency C, --ignore-eos and --num-prompts = max(4C,64) for router/judge,
# max(2C,16) for agent (ROUTER_NP_MULT/_MIN ..., NUM_PROMPTS_CAP), while
# `nvidia-smi dmon -s pucm` samples power/util/clocks/memory once per second.  The estimated
# minutes per run (EST_TOTAL_TOK_S, default 20k total tok/s) are logged up front and refined
# with the last measured throughput; MAX_RUN_MINUTES=N skips runs estimated longer than N
# (recorded as <run_id>.skipped.json).
#
# x4 replica cells (REPLICAS>1): concurrency C is split across min(C,REPLICAS) servers
# (one bench process per port, run concurrently; C=1 uses one replica, C=4 uses four at
# c=1 each, C=256 uses four at c=64). summarise.py sums req/s and tok/s across ports.
#
# Outputs in results/<cell>[__$RUN_TAG]/ :
#   <run_id>.meta.json  <run_id>[__pPORT].json  <run_id>[__pPORT].bench.log  <run_id>.dmon.csv
# run_id = <cell>[__tag]__<shape>__c<C>__<YYYYmmddTHHMMSS>
# meta and bench JSON carry p2p_ok, custom_allreduce, acs_suspected, pessimistic_tp (dagger
# in summarise.py for TP>1 rows on an ACS box; TP1 replica rows are never daggered; without
# results/hw/decisions.env the two flags are null / "unknown" and TP>1 rows show '?').
# All `vllm bench serve` flags verified against vllm main 2026-09-02
# (vllm/benchmarks/serve.py, vllm/benchmarks/datasets/datasets.py, docs.vllm.ai cli/bench/serve).
# BENCH_DROP_ARGS="--flag1 --flag2" removes a flag (and its value) from the client command line
# without editing this script, should the box's exact build reject one.
# Failure policy: a failed warm-up is FATAL (tail of warmup_p*.log printed) unless WARMUP_FAIL_OK=1;
# with STOP_ON_ERROR unset the FIRST failing point stops the sweep (a broken client would fail every
# point), later failures are logged; STOP_ON_ERROR=1 stops on any failure, STOP_ON_ERROR=0 never stops.
# The concurrency list is validated (positive integers only).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

usage() { sed -n '3,/^# =\{5,\}/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 1; }   # header block
[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac
CELL="$1"; shift

SHAPES=""; CONCS=""; DATASET_PATH=""; OUT_LEN_DS=256; SHAPE_LABEL=custom
VIA_PROXY=0; WARMUP=1; DRY=0
pos=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset-path) DATASET_PATH="$2"; shift ;;
    --output-len)   OUT_LEN_DS="$2"; shift ;;
    --shape-label)  SHAPE_LABEL="$2"; shift ;;
    --via-proxy)    VIA_PROXY=1 ;;
    --no-warmup)    WARMUP=0 ;;
    --dry-run)      DRY=1 ;;
    -h|--help)      usage ;;
    --*)            die "unknown option: $1" ;;
    *)
      pos=$((pos + 1))
      case $pos in
        1) SHAPES="$1" ;;
        2) CONCS="$1" ;;
        *) die "too many positional arguments" ;;
      esac ;;
  esac
  shift
done
SHAPES="${SHAPES:-$DEFAULT_SHAPES}"; SHAPES="${SHAPES//,/ }"
CONCS="${CONCS//,/ }"                      # empty -> per-shape defaults (shape_concs)
for c in $CONCS; do                        # validate: `abc` would die deep inside shape_num_prompts, `0` would run 0 requests
  case "$c" in ''|*[!0-9]*) die "concurrency '$c' is not a positive integer (list: $CONCS)" ;; esac
  [ "$c" -ge 1 ] || die "concurrency must be >= 1 (got $c)"
done

load_cell "$CELL"
# Reset before each measured point, after warm-up. Preserve caching WITHIN a
# workload; prevent reuse across warm-up, concurrency points and repeated A/Bs.
CACHE_POLICY="${CACHE_POLICY:-reset}"
case "$CACHE_POLICY" in reset|uncontrolled) ;; *) die "CACHE_POLICY must be reset or uncontrolled" ;; esac
DATASET_SHA256=""
if [ "$LOADTEST_ONLY" = 1 ]; then die "cell $CELL_NAME is LOADTEST_ONLY (attempt-to-load); nothing to sweep"; fi
SPEC_DECODING=off; if [ -n "${SPEC_CONFIG:-}" ]; then SPEC_DECODING=on; fi   # same spelling as launch.json
[ "$DRY" = 1 ] || mkdir -p "$RESULTS_DIR"
ENGINE_VERSION="$(cat "$RESULTS_DIR/engine_version.txt" 2>/dev/null || engine_version)"
warn_if_no_decisions

if [ -n "$DATASET_PATH" ]; then
  [ -f "$DATASET_PATH" ] || die "dataset not found: $DATASET_PATH"
  SHAPES="$SHAPE_LABEL"
  DATASET_SHA256="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$DATASET_PATH")"
fi

# ---- which bench client? -----------------------------------------------------------
detect_bench_cmd() {
  # shellcheck disable=SC2206
  if [ -n "${BENCH_CMD_OVERRIDE:-}" ]; then read -r -a BENCH_CMD <<< "$BENCH_CMD_OVERRIDE"; return; fi
  if vllm bench serve --help >/dev/null 2>&1; then BENCH_CMD=( vllm bench serve ); return; fi
  local f
  for f in "${BENCHMARK_SERVING_PY:-}" /vllm-workspace/benchmarks/benchmark_serving.py \
           /workspace/vllm/benchmarks/benchmark_serving.py "$BENCH_ROOT/benchmark_serving.py"; do
    if [ -n "$f" ] && [ -f "$f" ]; then BENCH_CMD=( python3 "$f" ); return; fi
  done
  die "neither 'vllm bench serve' nor benchmark_serving.py found. In an SGLang image: pip install vllm (client only), or set BENCHMARK_SERVING_PY=/path/benchmark_serving.py"
}
if [ "$DRY" = 1 ]; then
  # shellcheck disable=SC2206
  BENCH_CMD=( ${BENCH_CMD_OVERRIDE:-vllm bench serve} )
else
  detect_bench_cmd
fi
log "bench client: ${BENCH_CMD[*]}   engine=$ENGINE $ENGINE_VERSION   tokenizer=$MODEL_PATH ($MODEL_SOURCE)"
log "hw: P2P_OK=$P2P_OK NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} custom_allreduce=$CUSTOM_ALLREDUCE ACS_SUSPECTED=${ACS_SUSPECTED:-unknown} pessimistic_tp=${CELL_PESSIMISTIC_TP:-unknown} (decisions: $HW_DECISIONS_SOURCE)"

# ---- targets -------------------------------------------------------------------------
if [ "$REPLICAS" -gt 1 ] && [ "$VIA_PROXY" = 1 ]; then
  MODE=proxy; TARGET_PORTS=( "$PROXY_PORT" )
elif [ "$REPLICAS" -gt 1 ]; then
  MODE=per-port; TARGET_PORTS=( "${PORTS[@]}" )
else
  MODE=single; TARGET_PORTS=( "${PORTS[0]}" )
fi
if [ "$DRY" = 0 ]; then
  for p in "${TARGET_PORTS[@]}"; do
    curl -fsS -m 5 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || die "port $p not healthy — run bench/launch.sh $CELL first"
  done
fi

# ---- dmon (power/util/clock/mem sampler) --------------------------------------------------
DMON_PID=""
dmon_start() {
  local csv="$1"
  have nvidia-smi || return 0
  if have stdbuf; then
    stdbuf -oL nvidia-smi dmon -i "$GPU_IDS" -s pucm -d 1 -o DT > "$csv" 2>/dev/null &
  else
    nvidia-smi dmon -i "$GPU_IDS" -s pucm -d 1 -o DT > "$csv" 2>/dev/null &
  fi
  DMON_PID=$!
}
dmon_stop() {
  if [ -n "$DMON_PID" ]; then
    kill -INT "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=""
  fi
}
cleanup() { dmon_stop; pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

# ---- helpers -----------------------------------------------------------------------------------
write_json() {  # <file> key=value ...   (ints/floats coerced; empty value -> null; never fatal)
  python3 - "$@" <<'PY' || log "WARN: could not write $1"
import json, sys
out, d = sys.argv[1], {}
for kv in sys.argv[2:]:
    k, _, v = kv.partition("=")
    if v == "":
        d[k] = None          # e.g. pessimistic_tp / acs_suspected unknown (no decisions.env), empty run_tag
        continue
    for t in (int, float):
        try:
            v = t(v); break
        except ValueError:
            pass
    d[k] = v
json.dump(d, open(out, "w"), indent=2)
PY
}
gpu_mem_now() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';' || true; }
float_gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 > b + 0) }'; }   # float_gt A B -> true if A > B

# bench_common_args <in_len> <out_len> [shared_prefix_len] -> BENCH_ARGS
bench_common_args() {
  local in_len="$1" out_len="$2" shared_prefix_len="${3:-0}"
  BENCH_ARGS=( --backend "$BENCH_BACKEND" --endpoint "$BENCH_ENDPOINT"
    --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_PATH" --trust-remote-code
    --request-rate inf --ignore-eos
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
    --save-result --result-dir "$RESULTS_DIR" --disable-tqdm
    --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_S:-60}" )
  if [ "${BENCH_SAVE_DETAILED:-0}" = 1 ]; then BENCH_ARGS+=( --save-detailed ); fi
  if [ -n "$DATASET_PATH" ]; then
    # --custom-output-len -1 -> per-line "output_tokens" (vllm/benchmarks/datasets/datasets.py)
    BENCH_ARGS+=( --dataset-name custom --dataset-path "$DATASET_PATH" --custom-output-len "$out_len" )
    if [ "$BENCH_BACKEND" != "openai-chat" ]; then BENCH_ARGS+=( --skip-chat-template ); fi
  else
    BENCH_ARGS+=( --dataset-name random --random-input-len "$((in_len - shared_prefix_len))" --random-output-len "$out_len" --random-range-ratio 0 )
    if [ "$shared_prefix_len" -gt 0 ]; then BENCH_ARGS+=( --random-prefix-len "$shared_prefix_len" ); fi
  fi
  BENCH_ARGS+=( ${BENCH_EXTRA_ARGS[@]+"${BENCH_EXTRA_ARGS[@]}"} )
  # BENCH_DROP_ARGS="--flag1 --flag2": drop a flag (and its value, if the next token is not another
  # --flag) that this exact build rejects, without editing the script.  Flags are logged once.
  if [ -n "${BENCH_DROP_ARGS:-}" ]; then
    local kept=() skip=0 a
    for a in "${BENCH_ARGS[@]}"; do
      if [ "$skip" = 1 ]; then
        skip=0
        case "$a" in --*) ;; *) continue ;; esac        # the dropped flag's value
      fi
      case " $BENCH_DROP_ARGS " in *" $a "*) skip=1; continue ;; esac
      kept+=( "$a" )
    done
    BENCH_ARGS=( ${kept[@]+"${kept[@]}"} )
    if [ -z "${_DROP_LOGGED:-}" ]; then _DROP_LOGGED=1; log "BENCH_DROP_ARGS: removed [$BENCH_DROP_ARGS] from the bench client command line"; fi
  fi
}

# run_bench <port> <conc> <num_prompts> <result_filename> <seed> <run_id> <shape> <in> <out> <C_total> -> background
run_bench() {
  local port="$1" c="$2" np="$3" fname="$4" seed="$5" run_id="$6" shape="$7" in_len="$8" out_len="$9" ctot="${10}"
  local cmd=( "${BENCH_CMD[@]}" "${BENCH_ARGS[@]}"
    --base-url "http://127.0.0.1:$port" --max-concurrency "$c" --num-prompts "$np" --seed "$seed"
    --result-filename "$fname"
    --metadata "run_id=$run_id" "cell=$CELL_NAME" "run_tag=${RUN_TAG:-none}" "engine=$ENGINE" "engine_version=$ENGINE_VERSION"
      "model=$MODEL" "model_path=$MODEL_PATH" "model_source=$MODEL_SOURCE" "tp=$TP" "dp=$DP" "replicas=$REPLICAS"
      "kv_cache_dtype=$KV_CACHE_DTYPE" "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" "max_num_seqs=$MAX_NUM_SEQS"
      "max_model_len=$MAX_MODEL_LEN" "shape=$shape" "in_len=$in_len" "out_len=$out_len" "concurrency=$ctot"
      "replica_port=$port" "replica_concurrency=$c" "mode=$MODE" "gpu_ids=$GPU_IDS"
      "p2p_ok=$P2P_OK" "p2p_disabled=$P2P_DISABLED" "custom_allreduce=$CUSTOM_ALLREDUCE"
      "acs_suspected=${ACS_SUSPECTED:-unknown}" "pessimistic_tp=${CELL_PESSIMISTIC_TP:-unknown}" "spec_decoding=$SPEC_DECODING" )
  if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  "${cmd[@]}" > "$RESULTS_DIR/${fname%.json}.bench.log" 2>&1
}

LAST_TOTAL_TPS=""       # measured total tok/s of the last completed run -> refines the estimate
RUNS_ATTEMPTED=0        # points actually started (skipped-for-budget points do not count)

# run_one <shape> <C>
run_one() {
  local shape="$1" C="$2" in_len out_len dims shared_prefix_len=0
  if [ "$shape" = promptopt ] && [ -z "$DATASET_PATH" ]; then shared_prefix_len=3072; fi
  if [ -n "$DATASET_PATH" ]; then in_len=dataset; out_len="$OUT_LEN_DS"
  else dims="$(shape_dims "$shape")" || die "unknown shape '$shape' (router|judge|agent)"; read -r in_len out_len <<< "$dims"; fi
  local ts run_id
  ts="$(date +%Y%m%dT%H%M%S)"
  run_id="${CELL_NAME}${RUN_TAG:+__$RUN_TAG}__${shape}__c${C}__${ts}"
  # Stable across configurations for paired inputs, distinct across shapes/C.
  local point_seed cache_verified=0
  point_seed="$(python3 -c 'import hashlib,sys; print(int.from_bytes(hashlib.sha256("|".join(sys.argv[1:]).encode()).digest()[:4], "big") % 2147483647)' "$BENCH_SEED" "$shape" "$C")"
  bench_common_args "$in_len" "$out_len" "$shared_prefix_len"

  # concurrency split across replicas
  local active=1 base="$C" rem=0
  if [ "$MODE" = per-port ]; then
    active=$(( C < REPLICAS ? C : REPLICAS )); base=$(( C / active )); rem=$(( C % active ))
  fi
  local total_np; total_np="$(shape_num_prompts "$shape" "$C")"
  [ "$total_np" -ge "$C" ] || die "num_prompts=$total_np cannot reach requested concurrency C=$C"
  local est est_meas=""
  est="$(est_run_minutes "$total_np" "$in_len" "$out_len")"
  if [ -n "$LAST_TOTAL_TPS" ]; then est_meas="$(est_run_minutes "$total_np" "$in_len" "$out_len" "$LAST_TOTAL_TPS")"; fi

  log "=== $run_id  shape=$shape in=$in_len out=$out_len C=$C prompts=$total_np mode=$MODE active_replicas=$active  est ~${est} min @${EST_TOTAL_TOK_S} tok/s${est_meas:+ (~${est_meas} min @ last measured ${LAST_TOTAL_TPS} tok/s)} ==="
  if [ "$MAX_RUN_MINUTES" != 0 ] && [ "$est" != "?" ] && float_gt "$est" "$MAX_RUN_MINUTES"; then
    log "SKIP: estimated ${est} min > MAX_RUN_MINUTES=$MAX_RUN_MINUTES"
    [ "$DRY" = 1 ] || write_json "$RESULTS_DIR/$run_id.skipped.json" "run_id=$run_id" "cell=$CELL_NAME" "shape=$shape" \
        "concurrency=$C" "num_prompts=$total_np" "est_minutes=$est" "max_run_minutes=$MAX_RUN_MINUTES" "reason=time_budget"
    return 0
  fi
  if [ "$DRY" = 0 ] && [ "$CACHE_POLICY" = reset ]; then
    python3 "$HERE/cache_control.py" --engine "$ENGINE" --ports "${PORTS[@]}" \
      --out "$RESULTS_DIR/$run_id.cache.json" || die "cache reset failed; inspect $run_id.cache.json"
    cache_verified=1
  fi
  # High-concurrency historical runs failed with EMFILE. Raise only this process's
  # soft descriptor limit, within the existing hard limit; the server does likewise.
  local fd_limit; fd_limit="$(ulimit -Hn)"
  if [ "$fd_limit" = unlimited ] || [ "$fd_limit" -gt 65536 ]; then fd_limit=65536; fi
  ulimit -Sn "$fd_limit" || die "could not set benchmark file descriptor limit"
  if [ "$fd_limit" -lt $((C * 2 + 128)) ]; then die "descriptor limit $fd_limit too small for C=$C"; fi
  local started; started="$(date -Is)"
  RUNS_ATTEMPTED=$((RUNS_ATTEMPTED + 1))
  [ "$DRY" = 1 ] || dmon_start "$RESULTS_DIR/$run_id.dmon.csv"
  local pids=() ports_used=() concs_used=() i port ci npi np_sum=0 fname rc=0
  for ((i = 0; i < active; i++)); do
    port=${TARGET_PORTS[$i]}
    ci=$(( base + (i < rem ? 1 : 0) ))
    npi=$(( total_np / active + (i < total_np % active ? 1 : 0) ))
    np_sum=$(( np_sum + npi ))
    if [ "$MODE" = per-port ]; then fname="${run_id}__p${port}.json"; else fname="${run_id}.json"; fi
    ports_used+=( "$port" ); concs_used+=( "$ci" )
    run_bench "$port" "$ci" "$npi" "$fname" "$((point_seed + i))" "$run_id" "$shape" "$in_len" "$out_len" "$C" &
    pids+=( $! )
  done
  local p
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  [ "$DRY" = 1 ] || dmon_stop
  if [ "$DRY" = 1 ]; then return 0; fi
  local ended; ended="$(date -Is)"
  write_json "$RESULTS_DIR/$run_id.meta.json" \
    "run_id=$run_id" "cell=$CELL_NAME" "run_tag=${RUN_TAG:-}" "engine=$ENGINE" "engine_version=$ENGINE_VERSION" \
    "model=$MODEL" "model_path=$MODEL_PATH" "model_source=$MODEL_SOURCE" "tp=$TP" "dp=$DP" "replicas=$REPLICAS" "enable_ep=$ENABLE_EP" \
    "kv_cache_dtype=$KV_CACHE_DTYPE" "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" "max_num_seqs=$MAX_NUM_SEQS" \
    "max_model_len=$MAX_MODEL_LEN" "shape=$shape" "in_len=$in_len" "out_len=$out_len" "concurrency=$C" "num_prompts=$np_sum" \
    "mode=$MODE" "ports=$(IFS=,; echo "${ports_used[*]}")" "replica_concurrencies=$(IFS=,; echo "${concs_used[*]}")" \
    "dataset_path=${DATASET_PATH:-}" "dataset_sha256=$DATASET_SHA256" "gpu_ids=$GPU_IDS" "shared_prefix_len=$shared_prefix_len" \
    "seed=$point_seed" "cache_policy=$CACHE_POLICY" "cache_reset_verified=$cache_verified" \
    "ignore_eos=1" "output_length_mode=$([ "$out_len" = -1 ] && echo variable || echo fixed)" \
    "expected_total_output_tokens=${EXPECTED_TOTAL_OUTPUT_TOKENS:-}" \
    "p2p_ok=$P2P_OK" "p2p_disabled=$P2P_DISABLED" "custom_allreduce=$CUSTOM_ALLREDUCE" \
    "acs_suspected=$ACS_SUSPECTED" "pessimistic_tp=$CELL_PESSIMISTIC_TP" "hw_pessimistic_tp=$PESSIMISTIC_TP" \
    "hw_decisions_file=$HW_DECISIONS_SOURCE" "spec_decoding=$SPEC_DECODING" \
    "bench_backend=$BENCH_BACKEND" "bench_endpoint=$BENCH_ENDPOINT" "bench_client=${BENCH_CMD[*]}" \
    "est_minutes=$est" "est_total_tok_s=$EST_TOTAL_TOK_S" \
    "started=$started" "ended=$ended" "bench_exit_code=$rc" "gpu_mem_used_mib_after=$(gpu_mem_now)"
  local result_files=()
  for port in "${ports_used[@]}"; do
    if [ "$MODE" = per-port ]; then result_files+=( "$RESULTS_DIR/${run_id}__p${port}.json" )
    else result_files+=( "$RESULTS_DIR/${run_id}.json" ); fi
  done
  if ! python3 "$HERE/run_integrity.py" --meta "$RESULTS_DIR/$run_id.meta.json" \
      --out "$RESULTS_DIR/$run_id.integrity.json" "${result_files[@]}" > /dev/null; then
    log "WARN: run is ineligible for headline/cost claims; see $run_id.integrity.json"
    rc=1
  fi
  if [ "$rc" -ne 0 ]; then
    log "WARN: bench exited $rc for $run_id (see $RESULTS_DIR/${run_id}*.bench.log)"
    local bl; bl="$(ls "$RESULTS_DIR/${run_id}"*.bench.log 2>/dev/null | head -n1 || true)"
    if [ -n "$bl" ]; then tail -n 8 "$bl" >&2 || true; fi
    if [ "${STOP_ON_ERROR:-}" = 1 ]; then die "stopping (STOP_ON_ERROR=1)"; fi
    # STOP_ON_ERROR unset: the FIRST point failing means the client itself is broken (every later point
    # would fail the same way) -> stop; a later, isolated failure is logged and the sweep continues.
    if [ -z "${STOP_ON_ERROR:-}" ] && [ "$RUNS_ATTEMPTED" -eq 1 ]; then
      die "the first point of the sweep failed -> stopping (STOP_ON_ERROR=0 keeps going after any failure, STOP_ON_ERROR=1 stops on every failure)"
    fi
  else
    # one-line readout from the JSON(s); also feeds the running time estimate
    local readout tps
    readout="$(python3 - "${result_files[@]}" <<'PY' 2>/dev/null || true
import json, sys
files = sys.argv[1:]
rq = ot = tt = 0.0; c = n = 0; dur = 0.0
for f in files:
    j = json.load(open(f))
    rq += j.get("request_throughput", 0) or 0; ot += j.get("output_throughput", 0) or 0
    tt += j.get("total_token_throughput", 0) or 0; c += j.get("completed", 0) or 0; n += j.get("num_prompts", 0) or 0
    dur = max(dur, j.get("duration", 0) or 0)
print(f"    -> {c}/{n} done in {dur:.0f}s  {rq:.2f} req/s  {ot:,.0f} out tok/s  {tt:,.0f} total tok/s  ({len(files)} bench file(s))")
print(f"TOTAL_TPS {tt:.0f}")
PY
)"
    printf '%s\n' "$readout" | grep -v '^TOTAL_TPS' >&2 || true
    tps="$(printf '%s\n' "$readout" | awk '/^TOTAL_TPS/{print $2}')"
    if [ -n "$tps" ] && [ "$tps" != 0 ]; then LAST_TOTAL_TPS="$tps"; fi
  fi
}

# ---- plan (logged before anything runs) ----------------------------------------------------------
PLAN_TOTAL=0
for shape in $SHAPES; do
  if [ -n "$DATASET_PATH" ]; then p_in=dataset; p_out="$OUT_LEN_DS"
  else p_dims="$(shape_dims "$shape")" || die "unknown shape '$shape' (router|judge|agent)"; read -r p_in p_out <<< "$p_dims"; fi
  p_concs="${CONCS:-$(shape_concs "$shape")}"
  for C in $p_concs; do
    p_np="$(shape_num_prompts "$shape" "$C")"
    p_est="$(est_run_minutes "$p_np" "$p_in" "$p_out")"
    log "plan: $shape C=$C prompts=$p_np ~${p_est} min"
    if [ "$p_est" != "?" ]; then PLAN_TOTAL="$(awk -v a="$PLAN_TOTAL" -v b="$p_est" 'BEGIN{printf "%.1f", a + b}')"; fi
  done
done
log "plan total ~${PLAN_TOTAL} min at ${EST_TOTAL_TOK_S} total tok/s (EST_TOTAL_TOK_S; MAX_RUN_MINUTES=$MAX_RUN_MINUTES, NUM_PROMPTS_CAP=$NUM_PROMPTS_CAP)"

# ---- warm-up (cudagraph capture / JIT / prefix-cache priming; results discarded) -------------
if [ "$WARMUP" = 1 ] && [ "$DRY" = 0 ]; then
  log "warm-up: router shape, C=8, 32 prompts on port(s) ${TARGET_PORTS[*]}"
  mkdir -p "$RESULTS_DIR/warmup"
  read -r win wout <<< "$(shape_dims router)"
  bench_common_args "$win" "$wout"
  wpids=()
  for p in "${TARGET_PORTS[@]}"; do
    "${BENCH_CMD[@]}" "${BENCH_ARGS[@]}" --base-url "http://127.0.0.1:$p" \
      --max-concurrency 8 --num-prompts 32 --seed "$BENCH_SEED" --result-dir "$RESULTS_DIR/warmup" \
      --result-filename "warmup_p${p}_$(date +%Y%m%dT%H%M%S).json" > "$RESULTS_DIR/warmup/warmup_p${p}.log" 2>&1 &
    wpids+=( $! )
  done
  wfail=0
  for wp in "${wpids[@]}"; do wait "$wp" || wfail=$((wfail + 1)); done
  if [ "$wfail" -gt 0 ]; then
    # A client that cannot even warm up (tokenizer load failure, wrong alias, a flag this build rejects)
    # would otherwise fail all 18 points one by one and leave a completed-looking sweep with no numbers.
    for wl in "$RESULTS_DIR"/warmup/warmup_p*.log; do
      [ -f "$wl" ] || continue
      log "--- tail of $wl ---"; tail -n 15 "$wl" >&2 || true
    done
    if [ "${WARMUP_FAIL_OK:-0}" = 1 ]; then
      log "WARN: $wfail warm-up run(s) failed; continuing because WARMUP_FAIL_OK=1"
    else
      die "$wfail warm-up run(s) failed (see above, $RESULTS_DIR/warmup/). Fix the client first (BENCH_DROP_ARGS, DS4_BENCH_TOKENIZER_MODE / DS4_BENCH_TOKENIZER, ...) or WARMUP_FAIL_OK=1 to sweep anyway"
    fi
  fi
fi

# ---- the sweep -----------------------------------------------------------------------------------
SWEEP_T0=$(date +%s)
for shape in $SHAPES; do
  s_concs="${CONCS:-$(shape_concs "$shape")}"
  for C in $s_concs; do
    run_one "$shape" "$C"
  done
done
log "sweep finished in $(( $(date +%s) - SWEEP_T0 ))s -> $RESULTS_DIR"
if [ "$DRY" = 0 ]; then
  python3 "$HERE/summarise.py" --quiet || log "WARN: summarise.py failed"
  log "summary: $RESULTS_ROOT/summary.md"
fi
