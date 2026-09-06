#!/usr/bin/env bash
# =============================================================================
# bench/env.sh — shared environment for the 4x RTX PRO 6000 Blackwell (sm_120)
# throughput bench.  Sourced by launch.sh / sweep.sh / stop.sh / prefetch.sh /
# collect_env.sh and by the gates/ and train/ tracks (load_cell contract).
#
# Everything here is a DEFAULT.  Override by exporting before the call, e.g.
#   MAX_NUM_BATCHED_TOKENS=16384 RUN_TAG=mnbt16k bench/launch.sh ds4flash_tp4
#
# Runs INSIDE the Vast.ai container (root, GPUs exposed, no nested Docker).
#
# ---- hardware / decision contract:  $HW_DIR/decisions.env  (HW_DIR=results/hw) ----
# Written by vast/hardware_truth.sh (plain KEY=VALUE lines, '#' comments allowed):
#   P2P_OK=1            peer access supported on ALL GPU pairs (torch can_device_access_peer).
#                       Bandwidth/latency are RECORDED by hardware_truth, never used to
#                       disable P2P.
#   CUSTOM_ALLREDUCE=0  vLLM custom all-reduce (0 -> --disable-custom-all-reduce).  Default OFF;
#                       CUSTOM_ALLREDUCE=1 on the command line forces it on for an A/B.
#   NCCL_P2P_DISABLE=0  1 ONLY by explicit human decision when peer access is unsupported.
#                       bench/ never sets it on its own (host staging is slower than P2P here).
#   ACS_SUSPECTED=1     PCIe ACS on the host redirects switch-local P2P through the root
#                       complex (same-switch pair slower than cross-switch pair).
#   PESSIMISTIC_TP=1    TP>1 numbers are a lower bound for a node without ACS / with NVLink.
#   HOST_RAM_GB=1500    total host RAM (free -g) for the host-RAM warnings in launch.sh.
#   NOTES=free text
# Precedence: command-line environment > decisions.env > default.  P2P_OK is read from the file
# only.  When the file is MISSING, ACS_SUSPECTED / PESSIMISTIC_TP stay EMPTY (= unknown, never a
# silent 0): launch.json / *.meta.json then carry null, summarise.py and gates_summary.py mark
# TP>1 rows with '?' instead of the dagger, and warn_if_no_decisions() says so.
# Measured on the campaign box (2026-09-02): 4x RTX PRO 6000 Blackwell Server Edition, 96 GB,
# PCIe Gen5 x16, no NVLink; topo 0-1 PIX, 2-3 PIX, cross pairs NODE; peer access OK on all
# pairs; NCCL transport P2P/CUMEM; all_reduce busbw ~21 GB/s same-switch, ~38 GB/s
# cross-switch, ~19 GB/s 4-GPU ring regardless of NCCL tuning -> ACS enabled on the host,
# not changeable from the container.  Consequences implemented here: P2P stays ON, custom
# all-reduce OFF by default, DP2xTP2 cells pair cross-switch (GPU_IDS=0,2,1,3), TP2/TP4
# rows carry pessimistic_tp=1 (dagger in summarise.py), TP1 replica rows do not.
#
# ---- models -------------------------------------------------------------------------
# Weights live as plain directories $MODELS_DIR/<basename of HF id> (downloaded with
# `hf download <repo> --local-dir ...`, NOT the HF cache layout).  load_cell derives
# MODEL_PATH from MODEL and uses it for `vllm serve` and the bench client's --tokenizer when
# the directory exists, otherwise falls back to the HF id.  Disk is ~390 GB: models are
# benchmarked sequentially and deleted (bench/prefetch.sh --delete <cell>).
# =============================================================================

# Idempotent guard (safe to source several times in one shell).
if [ -n "${_BENCH_ENV_LOADED:-}" ]; then return 0 2>/dev/null || exit 0; fi
_BENCH_ENV_LOADED=1

# ---- helpers (defined first: used by the contract code below) -----------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
# norm01 <value> <default>: 1/true/yes/on -> 1, 0/false/no/off -> 0, anything else -> default
norm01() {
  local v; v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|on)   echo 1 ;;
    0|false|no|off)  echo 0 ;;
    *)               echo "${2:-0}" ;;
  esac
}
# norm01u <value>: like norm01 but anything unrecognised (incl. empty) -> "" = UNKNOWN, never a silent 0
norm01u() {
  local v; v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|on)   echo 1 ;;
    0|false|no|off)  echo 0 ;;
    *)               echo "" ;;
  esac
}

# ---- paths ------------------------------------------------------------------
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
CELLS_DIR="${CELLS_DIR:-$BENCH_ROOT/cells}"                # override only for tests / ad-hoc cell variants
RESULTS_ROOT="${RESULTS_ROOT:-$BENCH_ROOT/results}"
HW_DIR="${HW_DIR:-$RESULTS_ROOT/hw}"                       # hardware-truth outputs (vast/ track)
HW_DECISIONS_FILE="${HW_DECISIONS_FILE:-$HW_DIR/decisions.env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"              # plain model directories (see header)
export BENCH_DIR BENCH_ROOT CELLS_DIR RESULTS_ROOT HW_DIR HW_DECISIONS_FILE MODELS_DIR

# ---- legacy machine env ------------------------------------------------------------
# Older vast/hardware_truth.sh wrote $BENCH_ROOT/env.sh (P2P_OK=true/false, NCCL_P2P_DISABLE=1,
# VLLM_TP_EXTRA_ARGS, later also CUSTOM_ALLREDUCE/ACS_SUSPECTED/PESSIMISTIC_TP) and onstart
# sourced it in every login shell.  Neither happens any more (hardware_truth.sh writes only
# results/hw/{decisions.env,hardware.json,machine.env}; onstart's profile.d exports just the
# sm_120/NCCL baseline).  A stale copy is still read here, in a SUBSHELL, and only a whitelist
# of machine facts / extras is imported (never a decision key, never NCCL_*): decisions.env
# below is the single source of truth and only the caller's own environment may override it.
# BENCH_SKIP_ROOT_ENV=1 skips the file entirely.
if [ "${BENCH_SKIP_ROOT_ENV:-0}" != 1 ] && [ -f "$BENCH_ROOT/env.sh" ]; then
  _legacy_kv="$( ( set +eu
      # shellcheck disable=SC1091
      . "$BENCH_ROOT/env.sh" >/dev/null 2>&1
      for _k in COST_PER_HOUR GPU_COUNT GPU_NAME GPU_EDITION GPU_POWER_LIMIT_W GPU_VBIOS NVIDIA_DRIVER \
                SAME_SWITCH_PAIRS TP2_CROSS_SWITCH_GPU_IDS HW_JSON; do
        [ -n "${!_k+x}" ] && printf '%s=%q\n' "$_k" "${!_k}"
      done ) 2>/dev/null || true )"
  while IFS= read -r _line; do
    [ -n "$_line" ] || continue
    _k="${_line%%=*}"
    [ -n "${!_k+x}" ] || eval "export $_line"        # environment wins over the stale file
  done <<< "$_legacy_kv"
  unset _legacy_kv _line _k
fi

# ---- Hugging Face (all models are public; no token needed) -------------------
# HF_HOME defaults to /workspace/hf, the same value vast/onstart.sh exports, so tokenizer/config lookups by
# HF id land in one place.  It shares the ~390 GB container disk with $MODELS_DIR; weights never go here
# (hf download --local-dir), and launch.sh refuses to let `vllm serve <HF id>` fill it after a failed
# prefetch unless ALLOW_HUB_FALLBACK=1.
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM=false
# NOTE: HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT forced here — with it set and the
# hf_transfer package missing every download fails.  prefetch.sh enables it only when
# `python3 -c 'import hf_transfer'` works.

# ---- sm_120 (compute capability 12.0) vLLM baseline --------------------------
# Env names verified against vllm/envs.py on main (2026-09-02).
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"                # DeepGEMM has no sm_120 kernels
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"        # any JIT (triton/flashinfer) builds
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"            # exposes /sleep, /wake_up, /is_sleeping
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"   # default 600 s is too short for 170 GB loads
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-10}"

# ---- NCCL for PCIe Gen5, no NVLink -------------------------------------------
# NCCL_P2P_LEVEL=PHB keeps P2P for PIX and NODE pairs (this box: measured transport
# P2P/CUMEM on every pair under exactly this env).  Never set NCCL_P2P_DISABLE here.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ---- decision contract: parse $HW_DECISIONS_FILE -----------------------------------------
HW_P2P_OK=""; HW_CUSTOM_ALLREDUCE=""; HW_NCCL_P2P_DISABLE=""; HW_ACS_SUSPECTED=""
HW_PESSIMISTIC_TP=""; HW_HOST_RAM_GB=""; HW_NOTES=""
HW_DECISIONS_SOURCE="missing"
if [ -f "$HW_DECISIONS_FILE" ]; then
  HW_DECISIONS_SOURCE="$HW_DECISIONS_FILE"
  while IFS= read -r _line || [ -n "$_line" ]; do
    _line="${_line#"${_line%%[![:space:]]*}"}"                       # ltrim
    case "$_line" in ''|'#'*) continue ;; esac
    _line="${_line#export }"
    case "$_line" in *=*) ;; *) continue ;; esac
    _key="${_line%%=*}"; _val="${_line#*=}"
    _key="${_key%"${_key##*[![:space:]]}"}"                          # rtrim key
    _val="${_val#"${_val%%[![:space:]]*}"}"; _val="${_val%"${_val##*[![:space:]]}"}"
    case "$_val" in
      \"*\") _val="${_val#\"}"; _val="${_val%\"}" ;;
      \'*\') _val="${_val#\'}"; _val="${_val%\'}" ;;
    esac
    case "$_key" in
      P2P_OK)           HW_P2P_OK="$_val" ;;
      CUSTOM_ALLREDUCE) HW_CUSTOM_ALLREDUCE="$_val" ;;
      NCCL_P2P_DISABLE) HW_NCCL_P2P_DISABLE="$_val" ;;
      ACS_SUSPECTED)    HW_ACS_SUSPECTED="$_val" ;;
      PESSIMISTIC_TP)   HW_PESSIMISTIC_TP="$_val" ;;
      HOST_RAM_GB)      HW_HOST_RAM_GB="$_val" ;;
      NOTES)            HW_NOTES="$_val" ;;
      *) ;;                                                          # unknown keys ignored
    esac
  done < "$HW_DECISIONS_FILE"
  unset _line _key _val
fi

# P2P_OK comes from the file only (the legacy root env.sh exported P2P_OK=true/false into
# every login shell, so an environment value cannot be trusted).  Default 1: P2P is never
# disabled by guesswork.
P2P_OK="$(norm01 "$HW_P2P_OK" 1)"
# Explicit command-line values win over the file for the A/B-able knobs ("auto" -> 0: decision 2).
CUSTOM_ALLREDUCE="$(norm01 "${CUSTOM_ALLREDUCE:-$HW_CUSTOM_ALLREDUCE}" 0)"
# ACS_SUSPECTED / PESSIMISTIC_TP: command line > file > UNKNOWN ("" -- no silent 0 when the file
# is missing; consumers write null / '?').  A file with ACS but without PESSIMISTIC_TP derives it.
ACS_SUSPECTED="$(norm01u "${ACS_SUSPECTED:-$HW_ACS_SUSPECTED}")"
PESSIMISTIC_TP="$(norm01u "${PESSIMISTIC_TP:-$HW_PESSIMISTIC_TP}")"
if [ -z "$PESSIMISTIC_TP" ] && [ -n "$ACS_SUSPECTED" ]; then
  PESSIMISTIC_TP="$ACS_SUSPECTED"; [ "$P2P_OK" = 0 ] && PESSIMISTIC_TP=1
fi
# NCCL_P2P_DISABLE: dropped unless decisions.env says 1 (documented human decision) or
# FORCE_NCCL_P2P_DISABLE=1 is given explicitly for an A/B.  Anything inherited from the
# legacy root env.sh / a stale login shell is discarded.
_hw_p2p_disable="$(norm01 "$HW_NCCL_P2P_DISABLE" 0)"
if [ "${NCCL_P2P_DISABLE:-0}" = 1 ] && [ "$_hw_p2p_disable" != 1 ] && [ "${FORCE_NCCL_P2P_DISABLE:-0}" != 1 ]; then
  log "note: dropping inherited NCCL_P2P_DISABLE=1 (P2P works on this box; FORCE_NCCL_P2P_DISABLE=1 for an explicit A/B)"
fi
unset NCCL_P2P_DISABLE
P2P_DISABLED=0
if [ "$_hw_p2p_disable" = 1 ] || [ "${FORCE_NCCL_P2P_DISABLE:-0}" = 1 ]; then
  export NCCL_P2P_DISABLE=1
  P2P_DISABLED=1
fi
unset _hw_p2p_disable
# Host RAM (GB): command line > decisions.env > free -g > 0 (unknown)
HOST_RAM_GB="${HOST_RAM_GB:-$HW_HOST_RAM_GB}"
if [ -z "$HOST_RAM_GB" ] && have free; then HOST_RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"; fi
HOST_RAM_GB="${HOST_RAM_GB:-0}"
export P2P_OK P2P_DISABLED CUSTOM_ALLREDUCE ACS_SUSPECTED PESSIMISTIC_TP HOST_RAM_GB HW_DECISIONS_SOURCE HW_NOTES

# warn_if_no_decisions: one-line warning used by launch.sh / sweep.sh
warn_if_no_decisions() {
  if [ "$HW_DECISIONS_SOURCE" = missing ]; then
    log "WARN: $HW_DECISIONS_FILE not found (run vast/hardware_truth.sh). Using P2P_OK=1 CUSTOM_ALLREDUCE=$CUSTOM_ALLREDUCE; ACS_SUSPECTED=${ACS_SUSPECTED:-unknown} PESSIMISTIC_TP=${PESSIMISTIC_TP:-unknown} — TP>1 rows get '?' (unknown) instead of the dagger until the file exists."
  elif [ -z "$PESSIMISTIC_TP" ]; then
    log "WARN: $HW_DECISIONS_SOURCE has no ACS_SUSPECTED/PESSIMISTIC_TP -> TP>1 rows are marked '?' (unknown)."
  fi
}

# ---- vLLM serve: common flags for every cell ---------------------------------
# All flags verified against vllm main (docs.vllm.ai/en/latest/cli/serve.html and
# vllm/engine/arg_utils.py, 2026-09-02).  cudagraph FULL_AND_PIECEWISE is the sm_120
# recommendation (DeepSeek-V4-Flash recipe).  Spec decoding is OFF everywhere: launch.sh
# adds --speculative-config only when a cell / the command line sets SPEC_CONFIG.
if [ -z "${COMPILATION_CONFIG:-}" ]; then
  COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
fi
VLLM_COMMON_ARGS=(
  --host 0.0.0.0
  --enable-prefix-caching
  --enable-chunked-prefill
  --no-enable-flashinfer-autotune
  --trust-remote-code
  --disable-uvicorn-access-log
)
DEFAULT_MAX_MODEL_LEN=40960          # >= 32768 in + 2048 out (agent shape)
DEFAULT_MAX_NUM_SEQS=256
DEFAULT_MAX_NUM_BATCHED_TOKENS=8192  # sweep 8192 / 16384 via MAX_NUM_BATCHED_TOKENS=16384 RUN_TAG=mnbt16k
DEFAULT_GPU_MEM_UTIL=0.92
DEFAULT_HEALTH_TIMEOUT=7200          # s; applied in load_cell so a cell's ${HEALTH_TIMEOUT:-N} can differ
PROXY_PORT="${PROXY_PORT:-8080}"             # rr_proxy.py for the x4 replica cells
PLE_TABLE_GB="${PLE_TABLE_GB:-51}"           # Legacy offload-warning estimate only; g798544433 GPU PLE is TP row-sharded and has no implemented PLE_CPU_OFFLOAD path.

# ---- sweep defaults (throughput-first, time-budgeted) ------------------------------------
# Per-shape concurrency lists and prompt counts.  The agent shape (32768 in / 2048 out)
# at C=256 with 4C prompts would prefill ~33M tokens (~30 min/point at ~20k tok/s), so it
# stops at C=64 with 2C prompts.  Everything is overridable from the environment; a
# concurrency list on the sweep.sh command line applies to every shape.
DEFAULT_SHAPES="router judge agent"
DEFAULT_CONCURRENCIES="${DEFAULT_CONCURRENCIES:-}"   # legacy single list for all shapes (empty = per-shape lists)
ROUTER_CONCS="${ROUTER_CONCS:-1 4 8 16 32 64 128 256}"
PROMPTOPT_CONCS="${PROMPTOPT_CONCS:-64 256 512 1024}"
JUDGE_CONCS="${JUDGE_CONCS:-1 4 16 64 128 256}"
AGENT_CONCS="${AGENT_CONCS:-1 4 16 64}"
CUSTOM_CONCS="${CUSTOM_CONCS:-$ROUTER_CONCS}"        # --dataset-path runs
# prompts per run = max(MULT * C, MIN), then capped by NUM_PROMPTS_CAP (0 = no cap)
ROUTER_NP_MULT="${ROUTER_NP_MULT:-4}"; ROUTER_NP_MIN="${ROUTER_NP_MIN:-64}"
PROMPTOPT_NP_MULT="${PROMPTOPT_NP_MULT:-8}"; PROMPTOPT_NP_MIN="${PROMPTOPT_NP_MIN:-64}"
JUDGE_NP_MULT="${JUDGE_NP_MULT:-4}";   JUDGE_NP_MIN="${JUDGE_NP_MIN:-64}"
AGENT_NP_MULT="${AGENT_NP_MULT:-2}";   AGENT_NP_MIN="${AGENT_NP_MIN:-16}"
CUSTOM_NP_MULT="${CUSTOM_NP_MULT:-4}"; CUSTOM_NP_MIN="${CUSTOM_NP_MIN:-64}"
NUM_PROMPTS_CAP="${NUM_PROMPTS_CAP:-0}"
EST_TOTAL_TOK_S="${EST_TOTAL_TOK_S:-20000}"          # planning assumption for the "est. minutes" log line
MAX_RUN_MINUTES="${MAX_RUN_MINUTES:-0}"              # >0: sweep.sh skips runs whose estimate exceeds it
BENCH_SEED="${BENCH_SEED:-1234}"
COST_PER_HOUR="${COST_PER_HOUR:-}"                   # USD/hr for the whole machine -> cost column
export COST_PER_HOUR

# shape -> "input_len output_len"
shape_dims() {
  case "$1" in
    router) echo "1024 128" ;;
    promptopt) echo "3584 256" ;; # 3072 shared prefix + 512 unique suffix
    judge)  echo "4096 512" ;;
    agent)  echo "32768 2048" ;;
    *) return 1 ;;
  esac
}
# shape -> default concurrency list
shape_concs() {
  if [ -n "$DEFAULT_CONCURRENCIES" ]; then echo "$DEFAULT_CONCURRENCIES"; return 0; fi
  case "$1" in
    router) echo "$ROUTER_CONCS" ;;
    promptopt) echo "$PROMPTOPT_CONCS" ;;
    judge)  echo "$JUDGE_CONCS" ;;
    agent)  echo "$AGENT_CONCS" ;;
    *)      echo "$CUSTOM_CONCS" ;;
  esac
}
# shape_num_prompts <shape> <C> -> prompts for the run (max(MULT*C, MIN), capped)
shape_num_prompts() {
  local shape="$1" C="$2" mult min n
  case "$shape" in
    router) mult=$ROUTER_NP_MULT; min=$ROUTER_NP_MIN ;;
    promptopt) mult=$PROMPTOPT_NP_MULT; min=$PROMPTOPT_NP_MIN ;;
    judge)  mult=$JUDGE_NP_MULT;  min=$JUDGE_NP_MIN ;;
    agent)  mult=$AGENT_NP_MULT;  min=$AGENT_NP_MIN ;;
    *)      mult=$CUSTOM_NP_MULT; min=$CUSTOM_NP_MIN ;;
  esac
  n=$(( mult * C )); [ "$n" -ge "$min" ] || n=$min
  if [ "$NUM_PROMPTS_CAP" -gt 0 ] && [ "$n" -gt "$NUM_PROMPTS_CAP" ]; then n=$NUM_PROMPTS_CAP; fi
  echo "$n"
}
# est_run_minutes <num_prompts> <in_len> <out_len> [total_tok_s] -> "12.3" or "?" (non-numeric lengths)
est_run_minutes() {
  local np="$1" il="$2" ol="$3" tps="${4:-$EST_TOTAL_TOK_S}"
  case "$il$ol$tps" in *[!0-9.]*) echo "?"; return 0 ;; esac
  awk -v np="$np" -v il="$il" -v ol="$ol" -v tps="$tps" 'BEGIN { if (tps <= 0) { print "?"; exit } printf "%.1f", np * (il + ol) / tps / 60 }'
}

# ---- models on disk ------------------------------------------------------------------
# model_dir_state <dir> -> complete | partial | missing
#   complete: config.json present, weights present, no *.incomplete under .cache/huggingface
#             (hf download --local-dir keeps in-flight files there) or a .complete marker.
model_dir_state() {
  local d="$1"
  [ -d "$d" ] || { echo missing; return 0; }
  if [ -f "$d/.complete" ]; then echo complete; return 0; fi
  [ -f "$d/config.json" ] || { echo partial; return 0; }
  if [ -d "$d/.cache/huggingface" ] && find "$d/.cache/huggingface" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
    echo partial; return 0
  fi
  if find "$d" -maxdepth 2 \( -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' -o -name '*.pt' \) -print -quit 2>/dev/null | grep -q .; then
    echo complete
  else
    echo partial
  fi
}
# resolve_model_path: MODEL (HF id or path) -> MODEL_PATH / MODEL_SOURCE (local|hub) / MODEL_DIR_STATE
resolve_model_path() {
  MODEL_DIR="$MODELS_DIR/$(basename "$MODEL")"
  if [ -d "$MODEL" ]; then MODEL_DIR="$MODEL"; fi
  MODEL_DIR_STATE="$(model_dir_state "$MODEL_DIR")"
  if [ "$MODEL_DIR_STATE" != missing ] && [ -f "$MODEL_DIR/config.json" ]; then
    MODEL_PATH="$MODEL_DIR"; MODEL_SOURCE=local
  else
    MODEL_PATH="$MODEL"; MODEL_SOURCE=hub
  fi
  export MODEL_DIR MODEL_DIR_STATE MODEL_PATH MODEL_SOURCE
}

# load_cell <cell>: sources cells/<cell>.env, validates, derives PORTS/SESSION/RESULTS_DIR/MODEL_PATH.
# One cell per process: load_cell EXPORTS the cell's values, and cells written as ${REPLICAS:-4} /
# ${GPU_IDS:-0,1,2,3} (the x4 cells, so `REPLICAS=1 GPU_IDS=0 bench/launch.sh <x4 cell>` and
# gates/run_kv_diff.sh can shrink them) would inherit them from a cell loaded earlier in the SAME shell.
# Every harness script loads exactly one cell (prefetch.sh resolves cells in a subshell).
load_cell() {
  local cell="$1"
  local f="$CELLS_DIR/$cell.env"
  [ -f "$f" ] || die "no such cell: $f  (available: $(for c in "$CELLS_DIR"/*.env; do c=$(basename "$c" .env); case "$c" in _*) ;; *) printf '%s ' "$c" ;; esac; done))"
  # shellcheck disable=SC1090
  . "$f"
  : "${CELL_NAME:=$cell}"
  : "${ENGINE:=vllm}"
  : "${MODEL:?cell must set MODEL}"
  : "${SERVED_MODEL_NAME:=$CELL_NAME}"
  : "${TP:=1}"; : "${DP:=1}"; : "${REPLICAS:=1}"
  : "${GPU_IDS:=0,1,2,3}"
  : "${BASE_PORT:=8000}"
  : "${KV_CACHE_DTYPE:=auto}"
  : "${MAX_MODEL_LEN:=$DEFAULT_MAX_MODEL_LEN}"
  : "${MAX_NUM_SEQS:=$DEFAULT_MAX_NUM_SEQS}"
  : "${MAX_NUM_BATCHED_TOKENS:=$DEFAULT_MAX_NUM_BATCHED_TOKENS}"
  : "${GPU_MEM_UTIL:=$DEFAULT_GPU_MEM_UTIL}"
  : "${ENABLE_EP:=0}"
  : "${BENCH_BACKEND:=openai}"
  : "${BENCH_ENDPOINT:=/v1/completions}"
  : "${LOADTEST_ONLY:=0}"
  : "${USE_PROXY:=1}"
  : "${HOST_RAM_NEEDED_GB:=0}"        # optional cell hint for launch.sh's host-RAM warning
  : "${HEALTH_TIMEOUT:=$DEFAULT_HEALTH_TIMEOUT}"
  [ "$REPLICAS" -ge 1 ] || die "REPLICAS must be >= 1"
  if [ "$REPLICAS" -gt 1 ] && { [ "$TP" -ne 1 ] || [ "$DP" -ne 1 ]; }; then
    die "REPLICAS>1 cells must use TP=1 DP=1 (one independent server per GPU)"
  fi
  RESULTS_DIR="$RESULTS_ROOT/$CELL_NAME${RUN_TAG:+__$RUN_TAG}"
  SESSION="bench_${CELL_NAME}"
  IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
  PORTS=()
  local i
  for ((i = 0; i < REPLICAS; i++)); do PORTS+=( $((BASE_PORT + i)) ); done
  if [ "$REPLICAS" -gt 1 ] && [ "${#GPU_ARR[@]}" -lt "$REPLICAS" ]; then
    die "GPU_IDS='$GPU_IDS' has fewer GPUs than REPLICAS=$REPLICAS"
  fi
  if [ "$REPLICAS" -eq 1 ] && [ "${#GPU_ARR[@]}" -lt $((TP * DP)) ]; then
    die "GPU_IDS='$GPU_IDS' has fewer GPUs than TP*DP=$((TP * DP))"
  fi
  # pessimistic_tp for THIS cell: TP>1 on a box whose decisions.env says PESSIMISTIC_TP=1.
  # TP1 cells (x4 replicas) are never pessimistic (0); TP>1 with no decisions.env stays "" (unknown).
  if [ "$TP" -gt 1 ]; then CELL_PESSIMISTIC_TP="$PESSIMISTIC_TP"; else CELL_PESSIMISTIC_TP=0; fi
  resolve_model_path
  export CELL_NAME ENGINE MODEL SERVED_MODEL_NAME TP DP REPLICAS GPU_IDS BASE_PORT \
         KV_CACHE_DTYPE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEM_UTIL \
         ENABLE_EP BENCH_BACKEND BENCH_ENDPOINT LOADTEST_ONLY USE_PROXY RESULTS_DIR SESSION \
         HOST_RAM_NEEDED_GB HEALTH_TIMEOUT CELL_PESSIMISTIC_TP
}

# engine_version: prints the installed engine version (or "unknown").
engine_version() {
  case "${ENGINE:-vllm}" in
    sglang) python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo unknown ;;
    *)      python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown ;;
  esac
}

# gpu_list_for_replica <i>: CUDA_VISIBLE_DEVICES for replica i (single server -> all GPU_IDS,
# in the cell's order: vLLM gives DP rank i the i-th TP-sized slice of that list).
gpu_list_for_replica() {
  if [ "$REPLICAS" -gt 1 ]; then echo "${GPU_ARR[$1]}"; else echo "$GPU_IDS"; fi
}
