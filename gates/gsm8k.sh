#!/usr/bin/env bash
# =============================================================================
# gates/gsm8k.sh <cell> [port]
#
# Correctness gate 1: GSM8K (200-question subset) against the RUNNING server of a
# cell, via EleutherAI lm-evaluation-harness (`lm_eval[api]`, PyPI 0.4.13 on 2026-08-31)
# installed in an ISOLATED venv (--system-site-packages so the image's torch /
# transformers are reused; vLLM's pins are never modified).
#
# CLI verified against lm-eval docs/interface.md + lm_eval/models/api_models.py (2026-09-02):
#   --model local-completions | local-chat-completions
#   --model_args model=<served alias>,base_url=http://HOST:PORT/v1/completions,
#                num_concurrent=N,max_retries=3,tokenized_requests=False,
#                tokenizer=<local model dir or HF id>,max_gen_toks=N
#   --tasks gsm8k | gsm8k_cot   --limit 200   --seed 1234   --gen_kwargs temperature=0
#   --output_path <DIR> (lm_eval writes <DIR>/<model>/results_*.json)   --log_samples
#   chat: --apply_chat_template --fewshot_as_multiturn [--think_end_token </think>]
#   Unknown --gen_kwargs are forwarded into the request payload as-is; the parser
#   splits on commas, so a nested dict (chat_template_kwargs) cannot be passed that way.
#
# Cell integration: sources bench/env.sh and cells/<cell>.env (load_cell) so the
# served alias (SERVED_MODEL_NAME), the tokenizer (local /workspace/models/<basename>
# directory when present - decision 5 - else the HF id), the port (first replica =
# BASE_PORT, or the rr_proxy :8080 for x4 cells) and the results dir
# (results/<cell>[__$RUN_TAG]/) come from the cell definition.  Without a cell file
# the script falls back to /v1/models and the env overrides below.
#
# Output: results/<cell>[__RUN_TAG]/gsm8k.json (+ lm_eval_raw/, gsm8k_lm_eval.log),
#         carrying acs_suspected / pessimistic_tp (TP>1 only) from results/hw/decisions.env.
# Exit:   0 ok, 3 server unhealthy, 4 lm_eval failed / not installable, 5 gate failed vs REF_JSON
#
# Env: HOST=127.0.0.1  PORT=<override>  MODE=completions|chat (default from the cell's
#      BENCH_BACKEND)  TASK=gsm8k|gsm8k_cot  LIMIT=200  NUM_CONCURRENT=32 (floor: 32)
#      MAX_GEN_TOKS=512 (chat: 2048)  SEED=1234  SERVED_ALIAS=<served id override>
#      TOKENIZER=<dir or HF id>  TRUST_REMOTE_CODE=0|1  REF_JSON=<other gsm8k.json>
#      TOLERANCE=0.03  THINK_END_TOKEN='</think>' (chat mode; strips thinking before scoring)
#      EXTRA_GEN_KWARGS='key=val,...' (appended to --gen_kwargs)
#      LMEVAL_BIN=<path>  (default: /workspace/venv-eval/bin/lm_eval from onstart, else
#      $LMEVAL_VENV=/opt/lmeval-venv created here)  LMEVAL_SPEC='lm_eval[api]'
# =============================================================================
set -euo pipefail

CELL="${1:?usage: gates/gsm8k.sh <cell> [port]}"
PORT_ARG="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
say() { printf '[gsm8k %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
command -v python3 >/dev/null 2>&1 || { say "python3 not found"; exit 4; }
command -v curl >/dev/null 2>&1 || { say "curl not found"; exit 3; }

# The user's MODEL (served id, no-cell mode) must be read BEFORE load_cell overwrites MODEL with the HF id.
USER_SERVED="${SERVED_ALIAS:-${MODEL:-}}"

# ---- cell contract (bench/env.sh + cells/<cell>.env) -------------------------
HF_MODEL=""; SERVED=""; DEFAULT_PORT=8000; CELL_BACKEND="openai"; OUT_DIR=""; TP=""; DP=""; REPLICAS=""; MODEL_PATH="${MODEL_PATH:-}"
if [[ -f "$ROOT/bench/env.sh" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/bench/env.sh"
fi
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
HW_DIR="${HW_DIR:-$RESULTS_ROOT/hw}"
export RESULTS_ROOT HW_DIR GATES_DIR="$SCRIPT_DIR"
if [[ -f "$ROOT/cells/$CELL.env" ]] && declare -F load_cell >/dev/null; then
  load_cell "$CELL"
  HF_MODEL="$MODEL"; SERVED="$SERVED_MODEL_NAME"; CELL_BACKEND="$BENCH_BACKEND"; OUT_DIR="$RESULTS_DIR"
  if [[ "$REPLICAS" -gt 1 && "${USE_PROXY:-1}" == 1 ]]; then DEFAULT_PORT="$PROXY_PORT"; else DEFAULT_PORT="${PORTS[0]}"; fi
  CELL="$CELL_NAME"
else
  say "no cells/$CELL.env; using /v1/models + env overrides"
fi
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/$CELL}"
RAW_DIR="$OUT_DIR/lm_eval_raw"
mkdir -p "$RAW_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT_ARG:-${PORT:-$DEFAULT_PORT}}"
BASE="http://$HOST:$PORT"
if [[ -z "${MODE:-}" ]]; then MODE=completions; [[ "$CELL_BACKEND" == "openai-chat" ]] && MODE=chat; fi
TASK="${TASK:-gsm8k}"
LIMIT="${LIMIT:-200}"
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"
if (( NUM_CONCURRENT < 32 )); then say "NUM_CONCURRENT=$NUM_CONCURRENT is below the gate floor; using 32"; NUM_CONCURRENT=32; fi
SEED="${SEED:-1234}"
REF_JSON="${REF_JSON:-}"
TOLERANCE="${TOLERANCE:-0.03}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
THINK_END_TOKEN="${THINK_END_TOKEN:-}"
EXTRA_GEN_KWARGS="${EXTRA_GEN_KWARGS:-}"

# ---- server must be healthy ---------------------------------------------------
say "waiting for $BASE/health"
for _ in $(seq 1 60); do
  curl -fsS -m 5 "$BASE/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS -m 5 "$BASE/health" >/dev/null 2>&1 || { say "server not healthy at $BASE (bench/launch.sh $CELL first)"; exit 3; }

SERVED_LIST="$(curl -fsS -m 30 "$BASE/v1/models" 2>/dev/null | python3 -c 'import json,sys
try: print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))
except Exception: print("")' || echo "")"
MODEL_ID="${SERVED_ALIAS:-${SERVED:-$USER_SERVED}}"
if [[ -z "$MODEL_ID" ]]; then
  MODEL_ID="${SERVED_LIST%% *}"
  [[ -n "$MODEL_ID" ]] || { say "could not determine the served model id from $BASE/v1/models; set SERVED_ALIAS"; exit 3; }
fi
case " $SERVED_LIST " in
  *" $MODEL_ID "*) ;;
  *) say "WARN: served alias '$MODEL_ID' is not in $BASE/v1/models (${SERVED_LIST:-<empty>}); requests may 404" ;;
esac

# ---- tokenizer: local model directory (decision 5) beats the HF id -----------------
# bench/env.sh load_cell exports MODEL_PATH when ${MODELS_DIR}/<basename> exists; derive it otherwise.
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
LOCAL_DIR=""
if [[ -n "$HF_MODEL" ]]; then
  if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]]; then LOCAL_DIR="$MODEL_PATH"
  elif [[ -f "$MODELS_DIR/$(basename "$HF_MODEL")/config.json" ]]; then LOCAL_DIR="$MODELS_DIR/$(basename "$HF_MODEL")"
  fi
fi
TOKENIZER="${TOKENIZER:-${LOCAL_DIR:-${HF_MODEL:-$MODEL_ID}}}"
ENGINE_VERSION_JSON="$(curl -fsS -m 5 "$BASE/version" 2>/dev/null || echo '{}')"
say "cell=$CELL served=$MODEL_ID tokenizer=$TOKENIZER task=$TASK mode=$MODE limit=$LIMIT concurrency=$NUM_CONCURRENT port=$PORT tp=${TP:-?}"

# ---- lm-eval binary (isolated venv, never the image interpreter) -------------------
LMEVAL_VENV="${LMEVAL_VENV:-/opt/lmeval-venv}"
LMEVAL_SPEC="${LMEVAL_SPEC:-lm_eval[api]}"
LM_EVAL_BIN="${LMEVAL_BIN:-}"
if [[ -z "$LM_EVAL_BIN" ]]; then
  for c in /workspace/venv-eval/bin/lm_eval "$LMEVAL_VENV/bin/lm_eval"; do
    [[ -x "$c" ]] && { LM_EVAL_BIN="$c"; break; }
  done
fi
if [[ -z "$LM_EVAL_BIN" ]]; then
  say "installing $LMEVAL_SPEC into $LMEVAL_VENV (isolated venv with --system-site-packages: the image's torch/transformers are visible, vLLM's environment is not modified)"
  if [[ ! -x "$LMEVAL_VENV/bin/python" ]]; then
    if python3 -m venv --system-site-packages "$LMEVAL_VENV" 2>/dev/null; then :
    elif command -v uv >/dev/null 2>&1 && uv venv --system-site-packages --seed --python python3 "$LMEVAL_VENV" >/dev/null 2>&1; then :
    else say "cannot create a venv (apt-get install -y python3-venv, or install uv). Refusing to pip-install into the vLLM interpreter."; exit 4
    fi
  fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$LMEVAL_VENV/bin/python" -q "$LMEVAL_SPEC" \
      || "$LMEVAL_VENV/bin/python" -m pip install -q "$LMEVAL_SPEC" || { say "lm_eval install failed"; exit 4; }
  else
    "$LMEVAL_VENV/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
    "$LMEVAL_VENV/bin/python" -m pip install -q "$LMEVAL_SPEC" || { say "lm_eval install failed"; exit 4; }
  fi
  LM_EVAL_BIN="$LMEVAL_VENV/bin/lm_eval"
fi
[[ -x "$LM_EVAL_BIN" ]] || { say "lm_eval binary not found at $LM_EVAL_BIN"; exit 4; }
LM_EVAL_PY="$(dirname "$LM_EVAL_BIN")/python"
LM_EVAL_VERSION="$("$LM_EVAL_PY" -c 'import lm_eval;print(lm_eval.__version__)' 2>/dev/null || echo unknown)"

# ---- command --------------------------------------------------------------------
EXTRA_ARGS=()
if [[ "$MODE" == "chat" ]]; then
  MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"
  MODEL_TYPE="local-chat-completions"
  MODEL_ARGS="model=${MODEL_ID},base_url=${BASE}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,max_gen_toks=${MAX_GEN_TOKS}"
  EXTRA_ARGS+=(--apply_chat_template --fewshot_as_multiturn)
  # Reasoning models think in chat mode; --think_end_token discards everything up to the last
  # occurrence of the delimiter before scoring (lm-eval CLI flag, docs/interface.md).
  [[ -n "$THINK_END_TOKEN" ]] && EXTRA_ARGS+=(--think_end_token "$THINK_END_TOKEN")
else
  MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
  MODEL_TYPE="local-completions"
  MODEL_ARGS="model=${MODEL_ID},base_url=${BASE}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False,tokenizer=${TOKENIZER},max_gen_toks=${MAX_GEN_TOKS}"
fi
[[ "$TRUST_REMOTE_CODE" == "1" ]] && EXTRA_ARGS+=(--trust_remote_code)
GEN_KWARGS="temperature=0${EXTRA_GEN_KWARGS:+,$EXTRA_GEN_KWARGS}"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; T0=$(date +%s)
say "running: $LM_EVAL_BIN --model $MODEL_TYPE --model_args $MODEL_ARGS --tasks $TASK --limit $LIMIT --gen_kwargs $GEN_KWARGS ${EXTRA_ARGS[*]:-}"
set +e
"$LM_EVAL_BIN" \
  --model "$MODEL_TYPE" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --gen_kwargs "$GEN_KWARGS" \
  --output_path "$RAW_DIR" \
  --log_samples \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$OUT_DIR/gsm8k_lm_eval.log"
RC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
say "lm_eval exit $RC after $((T1 - T0)) s"

# ---- results/<cell>/gsm8k.json --------------------------------------------------
export GS_OUT_DIR="$OUT_DIR" GS_RAW_DIR="$RAW_DIR" GS_CELL="$CELL" GS_PORT="$PORT" GS_MODEL="$MODEL_ID" \
       GS_HF_MODEL="${HF_MODEL:-}" GS_TOKENIZER="$TOKENIZER" GS_MODEL_PATH="${LOCAL_DIR:-}" GS_MODE="$MODE" GS_TASK="$TASK" GS_LIMIT="$LIMIT" \
       GS_NUM_CONCURRENT="$NUM_CONCURRENT" GS_MAX_GEN_TOKS="$MAX_GEN_TOKS" GS_SEED="$SEED" GS_GEN_KWARGS="$GEN_KWARGS" \
       GS_START_TS="$START_TS" GS_DURATION="$((T1 - T0))" GS_RC="$RC" GS_ENGINE_VERSION_JSON="$ENGINE_VERSION_JSON" \
       GS_REF_JSON="$REF_JSON" GS_TOLERANCE="$TOLERANCE" \
       GS_LM_EVAL_VERSION="$LM_EVAL_VERSION" GS_LM_EVAL_BIN="$LM_EVAL_BIN" GS_MODEL_TYPE="$MODEL_TYPE" GS_MODEL_ARGS="$MODEL_ARGS" \
       GS_RUN_TAG="${RUN_TAG:-}" GS_TP="${TP:-}" GS_DP="${DP:-}" GS_REPLICAS="${REPLICAS:-}" GS_SERVED_LIST="$SERVED_LIST"
python3 - <<'PY'
import glob, json, os, sys, datetime as dt
sys.path.insert(0, os.environ["GATES_DIR"])
from hwdecisions import hw_decisions, pessimistic_flags
E = os.environ
out_dir, raw_dir, task = E["GS_OUT_DIR"], E["GS_RAW_DIR"], E["GS_TASK"]

def load(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return None  # noqa: BLE001

cands = sorted(glob.glob(os.path.join(raw_dir, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
raw_path = cands[-1] if cands else None
raw = load(raw_path) if raw_path else None
samples = sorted(glob.glob(os.path.join(raw_dir, "**", f"samples_{task}_*.json*"), recursive=True), key=os.path.getmtime)
metrics, num_fewshot, n_eff = {}, None, None
if raw:
    metrics = {k: v for k, v in ((raw.get("results") or {}).get(task) or {}).items() if k != "alias"}
    num_fewshot = ((raw.get("configs") or {}).get(task) or {}).get("num_fewshot")
    n_eff = ((raw.get("n-samples") or {}).get(task) or {}).get("effective")
strict = metrics.get("exact_match,strict-match")
flexible = metrics.get("exact_match,flexible-extract")
try: engine_version = json.loads(E["GS_ENGINE_VERSION_JSON"])
except Exception: engine_version = {"raw": E["GS_ENGINE_VERSION_JSON"]}  # noqa: BLE001
launch = load(os.path.join(out_dir, "launch.json")) or {}
tp = E.get("GS_TP") or launch.get("tp")
dec = hw_decisions()
flags = pessimistic_flags(dec, tp, E.get("GS_DP") or launch.get("dp"), E.get("GS_REPLICAS") or launch.get("replicas"))

status = "ok" if (E["GS_RC"] == "0" and raw is not None) else "lm_eval_failed"
passed, ref_info = None, None
if E["GS_REF_JSON"]:
    ref = load(E["GS_REF_JSON"])
    ref_strict = ((ref or {}).get("metrics") or {}).get("exact_match,strict-match")
    if isinstance(strict, (int, float)) and isinstance(ref_strict, (int, float)):
        tol = float(E["GS_TOLERANCE"]); passed = strict >= ref_strict - tol
        ref_info = {"path": E["GS_REF_JSON"], "cell": (ref or {}).get("cell"), "strict_match": ref_strict,
                    "delta": round(strict - ref_strict, 4), "tolerance": tol}
    else:
        ref_info = {"path": E["GS_REF_JSON"], "error": "reference has no exact_match,strict-match metric"}

summary = {
    "gate": "gsm8k", "status": status, "pass": passed,
    "cell": E["GS_CELL"], "run_tag": E["GS_RUN_TAG"] or None, "port": int(E["GS_PORT"]),
    "served_model": E["GS_MODEL"], "served_models_seen": (E["GS_SERVED_LIST"] or "").split(),
    "hf_model": E["GS_HF_MODEL"] or None, "model_path": E["GS_MODEL_PATH"] or None, "tokenizer": E["GS_TOKENIZER"],
    "engine_version": engine_version,
    "launch": {k: launch.get(k) for k in ("engine", "engine_version", "tp", "dp", "replicas", "gpu_ids", "kv_cache_dtype",
                                          "max_num_batched_tokens", "max_num_seqs", "custom_allreduce", "p2p_disabled", "p2p_ok",
                                          "acs_suspected", "pessimistic_tp", "status", "seconds_to_ready")},
    "lm_eval_version": E["GS_LM_EVAL_VERSION"], "lm_eval_bin": E["GS_LM_EVAL_BIN"],
    "lm_eval_model_type": E["GS_MODEL_TYPE"], "lm_eval_model_args": E["GS_MODEL_ARGS"], "gen_kwargs": E["GS_GEN_KWARGS"],
    "mode": E["GS_MODE"], "task": task, "limit": int(E["GS_LIMIT"]), "n_effective": n_eff, "num_fewshot": num_fewshot,
    "num_concurrent": int(E["GS_NUM_CONCURRENT"]), "max_gen_toks": int(E["GS_MAX_GEN_TOKS"]), "seed": int(E["GS_SEED"]), "temperature": 0,
    "metrics": metrics, "exact_match_strict": strict, "exact_match_flexible": flexible, "reference": ref_info,
    **flags,
    "hw_decisions": {k: v for k, v in dec.items() if not k.startswith("_")},
    "started_utc": E["GS_START_TS"], "duration_s": int(E["GS_DURATION"]), "lm_eval_exit_code": int(E["GS_RC"]),
    "raw_results_path": raw_path, "samples_paths": samples, "log_path": os.path.join(out_dir, "gsm8k_lm_eval.log"),
    "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(os.path.join(out_dir, "gsm8k.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: summary[k] for k in ("cell", "status", "pass", "exact_match_strict", "exact_match_flexible", "duration_s", "pessimistic_tp")}, indent=2))
print("wrote", os.path.join(out_dir, "gsm8k.json"))
if status != "ok": sys.exit(4)
if passed is False: sys.exit(5)
PY
