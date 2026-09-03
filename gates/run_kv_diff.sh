#!/usr/bin/env bash
# =============================================================================
# gates/run_kv_diff.sh <cell> [kv_dtype_a] [kv_dtype_b]
#
# Correctness gate 2: launch the SAME cell twice, sequentially, with a different
# --kv-cache-dtype (A = the cell's fp8* default, else 'fp8'; B = 'auto' = model
# dtype, i.e. bf16 KV), capture the fixed 50-prompt set from each side with
# gates/kv_diff.py (/v1/completions, temperature 0, seed 1234, max_tokens 512,
# thinking disabled) and compare.
#   bash gates/run_kv_diff.sh qwen38_27b_fp8_x4                 # fp8 vs auto
#   bash gates/run_kv_diff.sh ds4flash_tp4 fp8_ds_mla auto      # DeepSeek MLA layout
#   SELF_CHECK=1 bash gates/run_kv_diff.sh gptoss120b_x4_marlin # + same-server noise floor
#
# Launch path = the REAL launcher (decision 7); this script never starts vllm itself:
#   KV_CACHE_DTYPE=<kv> RUN_TAG=kvdiff_<kv> bench/launch.sh <cell> --no-smoke [--no-proxy]
#     - cells/<cell>.env declares KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-...}, so the env wins
#     - target port = the cell's BASE_PORT (first replica); x4 cells with VIA_PROXY=1
#       use the rr_proxy on :$PROXY_PORT (then the proxy is started too)
#     - KVDIFF_SINGLE_REPLICA=1 (default): an x4 replica cell is launched as ONE server on the
#       first GPU of its GPU_IDS (the cells declare REPLICAS=${REPLICAS:-4}, GPU_IDS=${GPU_IDS:-...})
#       -- replicas are identical independent servers and the capture only talks to BASE_PORT, so
#       this saves three model loads per side.  KVDIFF_SINGLE_REPLICA=0 keeps the full cell.
#     - blocks until /health on every port; exit 1 + launch.json.error_excerpt in
#       results/<cell>__kvdiff_<kv>/ when a server dies or times out (unsupported dtype)
#   bench/stop.sh <cell> --quiet   tears the cell down and waits for GPU memory release
# Fallback ONLY for an ad-hoc engine with no cell file (e.g. SGLang on a 2nd box):
#   SERVE_CMD='python3 -m sglang.launch_server ...' WITHOUT --port/--kv-cache-dtype
#   (both are appended), PORT (default 8000).
#
# Output (results/<cell>/): kv_diff.json, kv_capture_<A>.json, kv_capture_<B>.json,
#   [kv_capture_<B>_rerun.json + kv_diff_selfcheck.json with SELF_CHECK=1]
#   per-side server logs + launch.json in results/<cell>__kvdiff_<kv>/
# kv_diff.json carries acs_suspected / pessimistic_tp (TP>1 cells only) from
# results/hw/decisions.env (gates/hwdecisions.py), plus both sides' launch.json excerpts.
#
# Env: CONCURRENCY=4 MAX_TOKENS=512 SEED=1234 ENDPOINT=completions|chat
#      (chat sends chat_template_kwargs {"enable_thinking": false, "thinking": false}
#       -- Qwen3.x / DeepSeek-V4 template kwargs -- unless ALLOW_THINKING=1)
#      SELF_CHECK=0|1 KEEP_B_RUNNING=0|1 MAX_NED=0.30 PROMPTS=<custom.jsonl> VIA_PROXY=0|1
#      KV_A / KV_B (same as the positional args)  HEALTH_TIMEOUT=7200 (read by launch.sh)
#      KVDIFF_SINGLE_REPLICA=1|0 (x4 cells: one server on the first GPU, see above)
# Exit: 0 pass, 1 gate failed, 2 not applicable, 6 launch failed (details in kv_diff.json)
# Two model loads + 100 requests: run it inside tmux.
# =============================================================================
set -euo pipefail

CELL_ARG="${1:?usage: gates/run_kv_diff.sh <cell> [kv_dtype_a] [kv_dtype_b]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
say() { printf '[kv_diff %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
command -v python3 >/dev/null 2>&1 || { say "python3 not found"; exit 2; }
command -v curl >/dev/null 2>&1 || { say "curl not found"; exit 2; }

CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SEED="${SEED:-1234}"
ENDPOINT="${ENDPOINT:-completions}"
SELF_CHECK="${SELF_CHECK:-0}"
KEEP_B_RUNNING="${KEEP_B_RUNNING:-0}"
MAX_NED="${MAX_NED:-0.30}"
ALLOW_THINKING="${ALLOW_THINKING:-0}"
PROMPTS="${PROMPTS:-}"
VIA_PROXY="${VIA_PROXY:-0}"

# ---- cell contract: bench/env.sh + cells/<cell>.env via load_cell ------------------
USE_LAUNCH_SH=0; CELL="$CELL_ARG"; SERVED=""; CELL_KV_DEFAULT=""; TARGET_PORT="${PORT:-8000}"
TP=""; DP=""; REPLICAS=""
unset KV_CACHE_DTYPE RUN_TAG   # set per launch below; the cell's own default must be visible first
if [[ -f "$ROOT/bench/env.sh" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/bench/env.sh"
fi
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
HW_DIR="${HW_DIR:-$RESULTS_ROOT/hw}"
export RESULTS_ROOT HW_DIR GATES_DIR="$SCRIPT_DIR"
if [[ -z "${SERVE_CMD:-}" && -f "$ROOT/cells/$CELL_ARG.env" && -f "$ROOT/bench/launch.sh" ]] && declare -F load_cell >/dev/null; then
  load_cell "$CELL_ARG"
  if [[ "$REPLICAS" -gt 1 && "${KVDIFF_SINGLE_REPLICA:-1}" == 1 ]]; then
    n_full="${#GPU_ARR[@]}"
    export REPLICAS=1 GPU_IDS="${GPU_ARR[0]}"     # inherited by bench/launch.sh and bench/stop.sh below
    load_cell "$CELL_ARG"                          # recompute PORTS / SESSION / RESULTS_DIR with the override
    if [[ "$REPLICAS" -eq 1 ]]; then
      say "KVDIFF_SINGLE_REPLICA=1: launching one replica on GPU $GPU_IDS (port ${PORTS[0]}) instead of $n_full servers"
    else
      say "note: cells/$CELL_ARG.env hard-codes REPLICAS=$REPLICAS (not \${REPLICAS:-N}); launching the full cell"
    fi
  fi
  USE_LAUNCH_SH=1; CELL="$CELL_NAME"; SERVED="$SERVED_MODEL_NAME"; CELL_KV_DEFAULT="$KV_CACHE_DTYPE"
  # PORTS[0] is the cell's BASE_PORT (first replica): one deterministic server, no load balancing.
  if [[ "$REPLICAS" -gt 1 && "$VIA_PROXY" == 1 ]]; then TARGET_PORT="$PROXY_PORT"; else TARGET_PORT="${PORTS[0]}"; fi
  if [[ "${LOADTEST_ONLY:-0}" == 1 ]]; then say "cell $CELL is LOADTEST_ONLY; kv_diff not applicable"; exit 2; fi
elif [[ -z "${SERVE_CMD:-}" ]]; then
  say "no cells/$CELL_ARG.env + bench/launch.sh and no SERVE_CMD; nothing to launch"; exit 2
fi
case "$CELL_KV_DEFAULT" in fp8*) DEFAULT_A="$CELL_KV_DEFAULT" ;; *) DEFAULT_A="fp8" ;; esac
KV_A="${2:-${KV_A:-$DEFAULT_A}}"
KV_B="${3:-${KV_B:-auto}}"
if [[ "$KV_A" == "$KV_B" ]]; then say "KV_A == KV_B ($KV_A): nothing to diff (SELF_CHECK=1 gives the same-server noise floor)"; exit 2; fi
if [[ -n "$CELL_KV_DEFAULT" && "$CELL_KV_DEFAULT" != fp8* ]]; then
  say "note: the cell's own KV dtype is '$CELL_KV_DEFAULT' (no FP8 KV in the sweep); side A=$KV_A is an explicit override"
fi
OUT_DIR="$RESULTS_ROOT/$CELL"
mkdir -p "$OUT_DIR"
BASE="http://127.0.0.1:$TARGET_PORT"
LAUNCH_FLAGS=(--no-smoke)
# x4 cells: the rr_proxy is only needed when we capture through it.
if [[ "$USE_LAUNCH_SH" == 1 && "${REPLICAS:-1}" -gt 1 && "$VIA_PROXY" != 1 ]]; then LAUNCH_FLAGS+=(--no-proxy); fi
say "cell=$CELL A=$KV_A B=$KV_B target=$BASE endpoint=$ENDPOINT concurrency=$CONCURRENCY tp=${TP:-?} dp=${DP:-?} replicas=${REPLICAS:-?} launcher=$([[ $USE_LAUNCH_SH == 1 ]] && echo "bench/launch.sh ${LAUNCH_FLAGS[*]}" || echo SERVE_CMD)"

# ---- launch / stop -----------------------------------------------------------------
SERVER_PID=""; LAST_LAUNCH_DIR=""
launch() {   # $1 kv -> 0 healthy, 1 failed (LAUNCH_ERR set)
  local kv="$1"; LAUNCH_ERR=""
  if [[ "$USE_LAUNCH_SH" == 1 ]]; then
    LAST_LAUNCH_DIR="$RESULTS_ROOT/${CELL}__kvdiff_${kv}"
    say "KV_CACHE_DTYPE=$kv RUN_TAG=kvdiff_$kv bench/launch.sh $CELL_ARG ${LAUNCH_FLAGS[*]}"
    if KV_CACHE_DTYPE="$kv" RUN_TAG="kvdiff_$kv" bash "$ROOT/bench/launch.sh" "$CELL_ARG" "${LAUNCH_FLAGS[@]}"; then return 0; fi
    LAUNCH_ERR="$(python3 - "$LAST_LAUNCH_DIR/launch.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(f"launch status={d.get('status')} after {d.get('seconds_to_ready')}s: {d.get('error_excerpt') or '<no excerpt; see server_p*.log>'}")
except Exception as e:  # noqa: BLE001
    print(f"bench/launch.sh failed before writing launch.json ({e})")
PY
)"
    return 1
  fi
  local logf="$OUT_DIR/kv_diff_server_${kv}.log"; LAST_LAUNCH_DIR="$OUT_DIR"
  say "SERVE_CMD ... --port $TARGET_PORT --kv-cache-dtype $kv"
  setsid bash -c "exec $SERVE_CMD --port $TARGET_PORT --kv-cache-dtype $kv" >"$logf" 2>&1 &
  SERVER_PID=$!
  local deadline=$(( $(date +%s) + ${HEALTH_TIMEOUT:-7200} ))
  while (( $(date +%s) < deadline )); do
    curl -fsS -m 5 "$BASE/health" >/dev/null 2>&1 && return 0
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      LAUNCH_ERR="process exited before /health: $(tail -n 30 "$logf" | tr '\n' ' ' | cut -c1-1500)"; return 1
    fi
    sleep 10
  done
  LAUNCH_ERR="no /health within ${HEALTH_TIMEOUT:-7200}s"; return 1
}
stop() {
  if [[ "$USE_LAUNCH_SH" == 1 ]]; then bash "$ROOT/bench/stop.sh" "$CELL_ARG" --quiet || true; return; fi
  [[ -n "$SERVER_PID" ]] || return 0
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 45); do
    local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 || echo 0)"
    [[ -z "$used" || "$used" -lt 2000 ]] && break; sleep 2
  done
  SERVER_PID=""
}
record_failure() {   # $1 kv, $2 reason  -> kv_diff.json with status launch_failed (same pattern as the LOADTEST_ONLY cell)
  KVD_TP="$TP" KVD_DP="$DP" KVD_REPLICAS="$REPLICAS" python3 - "$OUT_DIR/kv_diff.json" "$CELL" "$1" "$2" "$KV_A" "$KV_B" "$LAST_LAUNCH_DIR" <<'PY'
import json, os, sys, datetime as dt
sys.path.insert(0, os.environ["GATES_DIR"])
from hwdecisions import hw_decisions, pessimistic_flags
out, cell, kv, reason, kva, kvb, ldir = sys.argv[1:8]
E = os.environ
flags = pessimistic_flags(hw_decisions(), E.get("KVD_TP"), E.get("KVD_DP"), E.get("KVD_REPLICAS"))
launch = None
try:
    launch = json.load(open(os.path.join(ldir, "launch.json"), encoding="utf-8"))
except Exception:  # noqa: BLE001
    pass
rec = {"gate": "kv_diff", "cell": cell, "pass": False, "status": "launch_failed", "kv_dtype_failed": kv,
       "reason": reason, "kv_a": kva, "kv_b": kvb, "launch_dir": ldir,
       "launch_json": {k: launch.get(k) for k in ("status", "seconds_to_ready", "engine", "engine_version", "kv_cache_dtype",
                                                    "tp", "dp", "replicas", "gpu_ids", "custom_allreduce", "error_excerpt")} if launch else None,
       **flags,
       "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(rec, open(out, "w", encoding="utf-8"), indent=2)
print(f"wrote {out} (launch_failed: {reason[:200]})")
PY
}
trap 'stop' EXIT

CAPTURE_ARGS=(--cell "$CELL" --endpoint "$ENDPOINT" --max-tokens "$MAX_TOKENS" --seed "$SEED" --concurrency "$CONCURRENCY")
[[ -n "$SERVED" ]] && CAPTURE_ARGS+=(--model "$SERVED")
[[ "$ALLOW_THINKING" == 1 ]] && CAPTURE_ARGS+=(--allow-thinking)
[[ -n "$PROMPTS" ]] && CAPTURE_ARGS+=(--prompts "$PROMPTS")
COMPARE_ARGS=(--cell "$CELL" --max-ned "$MAX_NED")
[[ -n "$TP" ]] && COMPARE_ARGS+=(--tp "$TP")
[[ -n "$DP" ]] && COMPARE_ARGS+=(--dp "$DP")
[[ -n "$REPLICAS" ]] && COMPARE_ARGS+=(--replicas "$REPLICAS")
CAP_A="$OUT_DIR/kv_capture_${KV_A}.json"
CAP_B="$OUT_DIR/kv_capture_${KV_B}.json"
CAP_B2="$OUT_DIR/kv_capture_${KV_B}_rerun.json"
LAUNCH_A=""; LAUNCH_B=""

# ---- side A ------------------------------------------------------------------------------
if ! launch "$KV_A"; then record_failure "$KV_A" "$LAUNCH_ERR"; exit 6; fi
[[ "$USE_LAUNCH_SH" == 1 ]] && LAUNCH_A="$LAST_LAUNCH_DIR/launch.json"
say "side A ($KV_A) healthy on $BASE; capturing $CONCURRENCY-way"
python3 "$SCRIPT_DIR/kv_diff.py" capture --url "$BASE" --label "$KV_A" --out "$CAP_A" "${CAPTURE_ARGS[@]}" || say "side A capture had request errors (continuing; they count against the gate)"
stop

# ---- side B ------------------------------------------------------------------------------
if ! launch "$KV_B"; then record_failure "$KV_B" "$LAUNCH_ERR"; exit 6; fi
[[ "$USE_LAUNCH_SH" == 1 ]] && LAUNCH_B="$LAST_LAUNCH_DIR/launch.json"
say "side B ($KV_B) healthy on $BASE; capturing $CONCURRENCY-way"
python3 "$SCRIPT_DIR/kv_diff.py" capture --url "$BASE" --label "$KV_B" --out "$CAP_B" "${CAPTURE_ARGS[@]}" || say "side B capture had request errors (continuing; they count against the gate)"

NOISE_ARGS=()
if [[ "$SELF_CHECK" == 1 ]]; then
  say "SELF_CHECK: second capture of side B for the same-server noise floor"
  python3 "$SCRIPT_DIR/kv_diff.py" capture --url "$BASE" --label "${KV_B}_rerun" --out "$CAP_B2" "${CAPTURE_ARGS[@]}" || true
  python3 "$SCRIPT_DIR/kv_diff.py" compare --a "$CAP_B" --b "$CAP_B2" "${COMPARE_ARGS[@]}" \
      --out "$OUT_DIR/kv_diff_selfcheck.json" --no-pairs || true
  NOISE_ARGS=(--noise-floor "$OUT_DIR/kv_diff_selfcheck.json")
fi
if [[ "$KEEP_B_RUNNING" == 1 ]]; then
  say "KEEP_B_RUNNING=1: leaving the $KV_B server up on $BASE (bench/stop.sh $CELL_ARG when done)"; trap - EXIT; SERVER_PID=""
else
  stop
fi

# ---- compare ------------------------------------------------------------------------------
[[ -n "$LAUNCH_A" ]] && COMPARE_ARGS+=(--launch-a "$LAUNCH_A")
[[ -n "$LAUNCH_B" ]] && COMPARE_ARGS+=(--launch-b "$LAUNCH_B")
set +e
python3 "$SCRIPT_DIR/kv_diff.py" compare --a "$CAP_A" --b "$CAP_B" "${COMPARE_ARGS[@]}" \
    --out "$OUT_DIR/kv_diff.json" ${NOISE_ARGS[@]+"${NOISE_ARGS[@]}"}
RC=$?
set -e
say "results in $OUT_DIR/kv_diff.json (exit $RC: 0=pass 1=fail)"
exit $RC
