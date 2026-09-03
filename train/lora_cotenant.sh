#!/usr/bin/env bash
# =============================================================================
# train/lora_cotenant.sh -- training co-tenancy cell
#
# Serving: cells/cotenant_tp2_gpu01.env (TP2 on the cell's GPU_IDS -- default 0,2, the
# cross-switch pair of decision 3; COTENANT_GPU_IDS=0,1 for the same-switch A/B -- port
# 8000, sleep mode on), launched here through the REAL launcher (bench/launch.sh) if it is not
# already healthy, and stopped with bench/stop.sh <cell> --quiet at the end
# (STOP_AFTER=auto: only if we launched it; 1 always; 0 never).
# Training: train/lora_qwen8b.py on CUDA_VISIBLE_DEVICES=$TRAIN_GPU (Unsloth LoRA
# r=16, bf16, batch 8 x seq 2048, Qwen/Qwen3-8B, yahma/alpaca-cleaned) for
# TRAIN_MINUTES, in its own tmux session (train_lora) so it survives an ssh drop and
# is inspectable (tmux attach -t train_lora).  Run THIS script in tmux as well:
#     tmux new -s cotenant 'bash train/lora_cotenant.sh'
#
# Sequence
#   1. serving cell healthy (or launch it)          bench/launch.sh $CELL
#   2. training venv                                 $TRAIN_PYTHON | $UNSLOTH_VENV |
#                                                    /workspace/venv-train (onstart) |
#                                                    /opt/unsloth-venv (train/install_unsloth.sh)
#   3. BEFORE : ~BASELINE_SECONDS of `vllm bench serve`, one shape, concurrency CONC
#   4. training starts on GPU $TRAIN_GPU; wait for its first optimizer step
#   5. DURING : the identical benchmark while training runs
#   6. wait for training; AFTER=1 repeats the benchmark once more
#   7. results/cotenancy.json (+ results/<cell>/cotenancy.json copy)
#   8. RUN_SLEEP_WAKE=1 (default): train/sleep_wake.sh on the same server
#   9. STOP_AFTER: bench/stop.sh $CELL --quiet   (never stop.sh --all: it kills the trainer too)
#
# Benchmark driver (flags verified against `vllm bench serve` docs, 2026-09-02; the same
# set bench/sweep.sh uses):
#   SWEEP_MODE=direct (default): `vllm bench serve --backend openai --endpoint /v1/completions
#     --dataset-name random --random-input-len/--random-output-len --random-range-ratio 0
#     --request-rate inf --max-concurrency CONC --ignore-eos --num-prompts N` with N calibrated
#     so a window lasts about BASELINE_SECONDS (a short run measures req/s first).
#     --tokenizer = the local model directory (/workspace/models/<basename>, decision 5) when present.
#   SWEEP_MODE=sweep_sh: RUN_TAG=cotenant_<phase> bench/sweep.sh $CELL $SHAPE $CONC --no-warmup
#
# Env: CELL=cotenant_tp2_gpu01 SHAPE=judge (router|judge|agent) CONC=64
#      BASELINE_SECONDS=300 TRAIN_MINUTES=15 TRAIN_GPU=3 AFTER=0|1 AUTO_LAUNCH=1 STOP_AFTER=auto|0|1
#      RUN_SLEEP_WAKE=1 TRAIN_PYTHON=<python> UNSLOTH_VENV=<path> TRAIN_* (passed to lora_qwen8b.py)
#      COTENANT_MODEL / COTENANT_ALIAS (see the cell file)  TRAIN_SESSION=train_lora
# Exit: 0 ok, 2 bad args, 3 no server, 4 training env failed, 5 no bench client
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
say() { printf '[cotenant %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

CELL="${CELL:-cotenant_tp2_gpu01}"
SHAPE="${SHAPE:-judge}"
CONC="${CONC:-64}"
BASELINE_SECONDS="${BASELINE_SECONDS:-300}"
export TRAIN_MINUTES="${TRAIN_MINUTES:-15}"
TRAIN_GPU="${TRAIN_GPU:-3}"
AFTER="${AFTER:-0}"
AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
STOP_AFTER="${STOP_AFTER:-auto}"
RUN_SLEEP_WAKE="${RUN_SLEEP_WAKE:-1}"
SWEEP_MODE="${SWEEP_MODE:-direct}"
TRAIN_START_TIMEOUT="${TRAIN_START_TIMEOUT:-1800}"
TRAIN_SESSION="${TRAIN_SESSION:-train_lora}"

# ---- cell contract --------------------------------------------------------------
unset RUN_TAG
[[ -f "$ROOT/bench/env.sh" ]] || { say "bench/env.sh missing; this script relies on the bench track's cell contract"; exit 3; }
# shellcheck disable=SC1091
. "$ROOT/bench/env.sh"
load_cell "$CELL"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
HW_DIR="${HW_DIR:-$RESULTS_ROOT/hw}"
export RESULTS_ROOT HW_DIR GATES_DIR="$ROOT/gates"
export MODELS_DIR="${MODELS_DIR:-/workspace/models}" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
COT_DIR="$RESULTS_ROOT/cotenancy"
CELL_DIR="$RESULTS_DIR"
mkdir -p "$COT_DIR" "$CELL_DIR"
PORT="${PORTS[0]}"; BASE="http://127.0.0.1:$PORT"
case ",$GPU_IDS," in *",$TRAIN_GPU,"*) say "cell $CELL uses GPU $TRAIN_GPU (GPU_IDS=$GPU_IDS); pick another TRAIN_GPU"; exit 2 ;; esac
case "$SHAPE" in
  router) IN_LEN=1024;  OUT_LEN=128  ;;
  judge)  IN_LEN=4096;  OUT_LEN=512  ;;
  agent)  IN_LEN=32768; OUT_LEN=2048 ;;
  *) say "unknown SHAPE=$SHAPE"; exit 2 ;;
esac
# Local model directory (decision 5): load_cell exports MODEL_PATH when it exists; derive otherwise.
BENCH_TOKENIZER="${MODEL_PATH:-}"
[[ -n "$BENCH_TOKENIZER" && -f "$BENCH_TOKENIZER/config.json" ]] || BENCH_TOKENIZER="$MODELS_DIR/$(basename "$MODEL")"
[[ -f "$BENCH_TOKENIZER/config.json" ]] || BENCH_TOKENIZER="$MODEL"
# Hardware decisions (results/hw/decisions.env) -> HW_ACS_SUSPECTED, HW_PESSIMISTIC_THIS (TP>1 only), ...
eval "$(python3 "$ROOT/gates/hwdecisions.py" --shell --tp "$TP" --dp "$DP" --replicas "$REPLICAS")"

# ---- 1. serving cell -----------------------------------------------------------------
WE_LAUNCHED=0
if ! curl -fsS -m 5 "$BASE/health" >/dev/null 2>&1; then
  if [[ "$AUTO_LAUNCH" == 1 ]]; then
    say "no server on $BASE; launching $CELL via bench/launch.sh (ENABLE_SLEEP_MODE=${ENABLE_SLEEP_MODE:-1})"
    ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE:-1}" bash "$ROOT/bench/launch.sh" "$CELL" || { say "launch failed (see $CELL_DIR/launch.json)"; exit 3; }
    WE_LAUNCHED=1
  else
    say "no healthy server at $BASE; bench/launch.sh $CELL first (or AUTO_LAUNCH=1)"; exit 3
  fi
fi
ENGINE_VERSION_JSON="$(curl -fsS -m 5 "$BASE/version" 2>/dev/null || echo '{}')"
say "serving: cell=$CELL_NAME model=$MODEL served=$SERVED_MODEL_NAME TP=$TP gpus=$GPU_IDS port=$PORT tokenizer=$BENCH_TOKENIZER | train gpu=$TRAIN_GPU minutes=$TRAIN_MINUTES | shape=$SHAPE ($IN_LEN/$OUT_LEN) C=$CONC | acs_suspected=${HW_ACS_SUSPECTED:-?} pessimistic_tp=${HW_PESSIMISTIC_THIS:-?}"

# ---- 2. training venv (isolated from the vLLM interpreter) -------------------------------
venv_ok() {  # $1 python: unsloth/trl/datasets import + torch with sm_120
  [[ -x "$1" ]] && "$1" - <<'PY' >/dev/null 2>&1
import sys, torch, unsloth, trl, datasets  # noqa: F401
sys.exit(0 if torch.cuda.is_available() and any(a in ("sm_120", "compute_120", "sm_121") for a in torch.cuda.get_arch_list()) else 1)
PY
}
TRAIN_PY=""
if [[ -n "${TRAIN_PYTHON:-}" ]]; then
  venv_ok "$TRAIN_PYTHON" || { say "TRAIN_PYTHON=$TRAIN_PYTHON lacks unsloth/trl/datasets or an sm_120 torch"; exit 4; }
  TRAIN_PY="$TRAIN_PYTHON"
elif [[ -n "${UNSLOTH_VENV:-}" ]] && venv_ok "$UNSLOTH_VENV/bin/python"; then
  TRAIN_PY="$UNSLOTH_VENV/bin/python"
elif venv_ok /workspace/venv-train/bin/python; then
  TRAIN_PY=/workspace/venv-train/bin/python; say "using onstart's /workspace/venv-train"
else
  export UNSLOTH_VENV="${UNSLOTH_VENV:-/opt/unsloth-venv}"
  say "no usable training venv; running train/install_unsloth.sh -> $UNSLOTH_VENV"
  bash "$SCRIPT_DIR/install_unsloth.sh" || { say "unsloth install/verify failed"; exit 4; }
  TRAIN_PY="$UNSLOTH_VENV/bin/python"
fi
say "training python: $TRAIN_PY"

# ---- bench driver ---------------------------------------------------------------------------
BENCH=()
if [[ "$SWEEP_MODE" == "direct" ]]; then
  if [[ -n "${BENCH_CMD_OVERRIDE:-}" ]]; then read -r -a BENCH <<< "$BENCH_CMD_OVERRIDE"
  elif vllm bench serve --help >/dev/null 2>&1; then BENCH=(vllm bench serve)
  else
    for f in "${BENCHMARK_SERVING_PY:-}" /vllm-workspace/benchmarks/benchmark_serving.py /workspace/vllm/benchmarks/benchmark_serving.py; do
      [[ -n "$f" && -f "$f" ]] && { BENCH=(python3 "$f"); break; }
    done
  fi
  [[ ${#BENCH[@]} -gt 0 ]] || { say "no 'vllm bench serve' / benchmark_serving.py (pip install vllm for the client)"; exit 5; }
fi

bench_once() {   # $1 tag, $2 num_prompts -> $COT_DIR/serve_<tag>.json
  local tag="$1" n="$2"
  local out="$COT_DIR/serve_${tag}.json"
  rm -f "$out"
  if [[ "$SWEEP_MODE" == "sweep_sh" ]]; then
    RUN_TAG="cotenant_${tag}" bash "$ROOT/bench/sweep.sh" "$CELL" "$SHAPE" "$CONC" --no-warmup || true
    # newest <run_id>.json of that sweep (skip .meta.json); run_id embeds a timestamp so name order == time order
    local newest="" f
    for f in "$RESULTS_ROOT/${CELL_NAME}__cotenant_${tag}"/*__"${SHAPE}"__c"${CONC}"__*.json; do
      [[ -f "$f" && "$f" != *.meta.json ]] && newest="$f"
    done
    [[ -n "$newest" ]] && cp "$newest" "$out"
  else
    "${BENCH[@]}" \
      --backend "$BENCH_BACKEND" --endpoint "$BENCH_ENDPOINT" --base-url "$BASE" \
      --model "$SERVED_MODEL_NAME" --tokenizer "$BENCH_TOKENIZER" --trust-remote-code \
      --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" --random-range-ratio 0 \
      --num-prompts "$n" --request-rate inf --max-concurrency "$CONC" --ignore-eos --seed "${BENCH_SEED:-1234}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm --ready-check-timeout-sec 60 \
      --save-result --result-dir "$COT_DIR" --result-filename "serve_${tag}.json" \
      --metadata "cell=$CELL_NAME" "phase=$tag" "shape=$SHAPE" "in_len=$IN_LEN" "out_len=$OUT_LEN" "concurrency=$CONC" \
                 "tp=$TP" "dp=$DP" "kv_cache_dtype=$KV_CACHE_DTYPE" "gpu_ids=$GPU_IDS" "train_gpu=$TRAIN_GPU" \
                 "custom_allreduce=${CUSTOM_ALLREDUCE:-}" "p2p_disabled=${P2P_DISABLED:-0}" "p2p_ok=${HW_P2P_OK:-}" \
                 "acs_suspected=${HW_ACS_SUSPECTED:-}" "pessimistic_tp=${HW_PESSIMISTIC_THIS:-}" \
      ${BENCH_EXTRA_ARGS[@]+"${BENCH_EXTRA_ARGS[@]}"} > "$COT_DIR/serve_${tag}.bench.log" 2>&1 || say "bench ($tag) exited non-zero, see $COT_DIR/serve_${tag}.bench.log"
  fi
  [[ -f "$out" ]] || { say "no result for phase $tag"; return 1; }
}
DMON_PID=""
dmon_start() { nvidia-smi dmon -i "$GPU_IDS,$TRAIN_GPU" -s pucm -d 2 -o DT > "$COT_DIR/dmon_$1.txt" 2>&1 & DMON_PID=$!; }
dmon_stop()  { [[ -n "$DMON_PID" ]] && { kill -INT "$DMON_PID" 2>/dev/null || true; wait "$DMON_PID" 2>/dev/null || true; }; DMON_PID=""; }
stop_if_needed() {
  if [[ "$STOP_AFTER" == 1 ]] || { [[ "$STOP_AFTER" == auto ]] && [[ "$WE_LAUNCHED" == 1 ]]; }; then
    say "stopping $CELL (STOP_AFTER=$STOP_AFTER, launched here=$WE_LAUNCHED)"
    bash "$ROOT/bench/stop.sh" "$CELL" --quiet || true
  fi
}
trap 'dmon_stop' EXIT
jget() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' "$1" "$2"; }

# ---- 3. calibrate + BEFORE ---------------------------------------------------------------------
NUM_PROMPTS=$(( 4 * CONC > 64 ? 4 * CONC : 64 ))
if [[ "$SWEEP_MODE" == "direct" ]]; then
  say "calibration ($((CONC * 2)) prompts)"
  bench_once calib "$((CONC * 2))" || true
  RPS="$(jget "$COT_DIR/serve_calib.json" request_throughput 2>/dev/null || echo "")"
  NUM_PROMPTS="$(python3 -c "import math,sys;r=float(sys.argv[1] or 0);c=int(sys.argv[2]);s=float(sys.argv[3]);print(max(2*c, min(20000, math.ceil(r*s))) if r>0 else 4*c)" "$RPS" "$CONC" "$BASELINE_SECONDS")"
  say "request_throughput=${RPS:-?} req/s -> $NUM_PROMPTS prompts per ~${BASELINE_SECONDS}s window"
fi
say "BEFORE: serving alone"
dmon_start before; BEFORE_T0="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
bench_once before "$NUM_PROMPTS" || true
BEFORE_T1="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; dmon_stop

# ---- 4. training on GPU $TRAIN_GPU (own tmux session) ------------------------------------------------
FLAG="$COT_DIR/train_started.flag"; EXITF="$COT_DIR/train_lora.exit"; rm -f "$FLAG" "$EXITF"
TRAIN_OUT="$RESULTS_ROOT/train_lora.json"
export TRAIN_OUT TRAIN_STEP_LOG="$RESULTS_ROOT/train_lora_steps.jsonl" TRAIN_STARTED_FLAG="$FLAG" TRAIN_LABEL="${TRAIN_LABEL:-cotenant}"
# The runner script carries the environment explicitly (a tmux server does not inherit ours) --
# the same pattern bench/launch.sh uses for its servers.  CUDA_VISIBLE_DEVICES uses the same
# default enumeration as the serving processes, so "GPU 3" means the same device for both.
RUNNER="$COT_DIR/train_lora.sh"
{
  echo '#!/usr/bin/env bash'
  echo "# generated by lora_cotenant.sh $(date -Is): Unsloth LoRA on GPU $TRAIN_GPU with $TRAIN_PY"
  for v in $(compgen -e | grep -E '^(TRAIN_|HF_|MODELS_DIR$|TOKENIZERS_PARALLELISM$|HOME$|PATH$)' || true); do
    printf 'export %s=%q\n' "$v" "${!v}"
  done
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$TRAIN_GPU"
  printf 'cd %q\n' "$ROOT"
  printf '%q %q > %q 2>&1\n' "$TRAIN_PY" "$SCRIPT_DIR/lora_qwen8b.py" "$COT_DIR/train_lora.log"
  printf 'echo $? > %q\n' "$EXITF"
} > "$RUNNER"
chmod +x "$RUNNER"
TRAIN_PID=""
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$TRAIN_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$TRAIN_SESSION" "bash '$RUNNER'"
  say "training started in tmux session $TRAIN_SESSION (log $COT_DIR/train_lora.log)"
else
  nohup bash "$RUNNER" >/dev/null 2>&1 &
  TRAIN_PID=$!; say "tmux missing; training started with nohup (pid $TRAIN_PID)"
fi
TRAIN_LAUNCH_T="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
train_alive() { [[ ! -f "$EXITF" ]] && { [[ -z "$TRAIN_PID" ]] || kill -0 "$TRAIN_PID" 2>/dev/null; }; }
deadline=$(( $(date +%s) + TRAIN_START_TIMEOUT )); TRAIN_STARTED=0
while (( $(date +%s) < deadline )); do
  [[ -f "$FLAG" ]] && { TRAIN_STARTED=1; break; }
  train_alive || break
  sleep 5
done

DURING_T0=""; DURING_T1=""
if (( TRAIN_STARTED )); then
  say "training took step 1; DURING: serving while training"
  dmon_start during; DURING_T0="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bench_once during "$NUM_PROMPTS" || true
  DURING_T1="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; dmon_stop
else
  say "training never reached step 1 within ${TRAIN_START_TIMEOUT}s (see $COT_DIR/train_lora.log); DURING skipped"
fi
say "waiting for training to finish (tmux attach -t $TRAIN_SESSION to watch)"
# model load + TRAIN_MINUTES + save + margin; a hung trainer is killed, never waited on forever
train_deadline=$(( $(date +%s) + TRAIN_START_TIMEOUT + TRAIN_MINUTES * 60 + 900 ))
while train_alive && (( $(date +%s) < train_deadline )); do sleep 5; done
if [[ ! -f "$EXITF" ]]; then
  say "training did not finish by the deadline; killing it"
  tmux kill-session -t "$TRAIN_SESSION" 2>/dev/null || true
  [[ -n "$TRAIN_PID" ]] && kill "$TRAIN_PID" 2>/dev/null || true
  echo "timeout" > "$EXITF"
fi
TRAIN_RC="$(cat "$EXITF" 2>/dev/null || echo unknown)"
[[ "$TRAIN_RC" == 0 ]] || say "training exited with rc=$TRAIN_RC (see $COT_DIR/train_lora.log)"
TRAIN_END_T="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

AFTER_T0=""; AFTER_T1=""
if [[ "$AFTER" == 1 ]]; then
  say "AFTER: serving alone again"
  dmon_start after; AFTER_T0="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bench_once after "$NUM_PROMPTS" || true
  AFTER_T1="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; dmon_stop
fi

# ---- 5. results/cotenancy.json -----------------------------------------------------------------------
export COT_DIR RESULTS_ROOT CELL_DIR CELL_NAME MODEL SERVED_MODEL_NAME BENCH_TOKENIZER TP DP REPLICAS KV_CACHE_DTYPE GPU_IDS SHAPE CONC IN_LEN OUT_LEN \
       NUM_PROMPTS BASELINE_SECONDS TRAIN_GPU TRAIN_MINUTES ENGINE_VERSION_JSON BEFORE_T0 BEFORE_T1 DURING_T0 DURING_T1 \
       AFTER_T0 AFTER_T1 TRAIN_LAUNCH_T TRAIN_END_T TRAIN_STARTED TRAIN_RC TRAIN_OUT TRAIN_PY SWEEP_MODE WE_LAUNCHED TRAIN_SESSION
python3 - <<'PY'
import json, os, shutil, sys, datetime as dt
sys.path.insert(0, os.environ["GATES_DIR"])
from hwdecisions import hw_decisions, pessimistic_flags
E = os.environ
def load(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return None  # noqa: BLE001
KEYS = ["duration", "completed", "failed", "num_prompts", "max_concurrency", "request_throughput", "output_throughput",
        "total_token_throughput", "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "median_itl_ms", "p99_itl_ms",
        "median_e2el_ms", "p99_e2el_ms"]
def serve(tag):
    d = load(os.path.join(E["COT_DIR"], f"serve_{tag}.json"))
    return {k: d.get(k) for k in KEYS if k in d} if d else None
def pct(a, b):
    try: return round(100.0 * (b - a) / a, 2)
    except Exception: return None  # noqa: BLE001
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ") if s else None

before, during, after = serve("before"), serve("during"), serve("after")
train = load(E["TRAIN_OUT"]) or {"status": "missing", "exit_code": E["TRAIN_RC"], "log": os.path.join(E["COT_DIR"], "train_lora.log")}
try: engine = json.loads(E["ENGINE_VERSION_JSON"])
except Exception: engine = {"raw": E["ENGINE_VERSION_JSON"]}  # noqa: BLE001
launch = load(os.path.join(E["CELL_DIR"], "launch.json")) or {}
dec = hw_decisions()
flags = pessimistic_flags(dec, E["TP"], E["DP"], E["REPLICAS"])
overlap = None
d0, d1, t0, t1 = ts(E["DURING_T0"]), ts(E["DURING_T1"]), ts(train.get("started_utc")), ts(train.get("ended_utc"))
if d0 and d1 and t0 and t1 and d1 > d0:
    overlap = round(max(0.0, (min(d1, t1) - max(d0, t0)).total_seconds()) / (d1 - d0).total_seconds(), 3)
delta = None
if before and during:
    delta = {k: pct(before.get(k), during.get(k)) for k in ("request_throughput", "output_throughput", "total_token_throughput")}
    for k in ("p99_ttft_ms", "p99_tpot_ms", "median_ttft_ms", "median_tpot_ms"):
        delta[k] = pct(before.get(k), during.get(k))
out = {
  "cell": "cotenancy", "serving_cell": E["CELL_NAME"], "serving_model": E["MODEL"], "served_model_name": E["SERVED_MODEL_NAME"],
  "bench_tokenizer": E["BENCH_TOKENIZER"], "engine_version": engine,
  "launch": {k: launch.get(k) for k in ("engine_version", "seconds_to_ready", "status", "kv_cache_line", "custom_allreduce", "p2p_disabled", "gpu_ids")},
  "serving_tp": int(E["TP"]), "serving_dp": int(E["DP"]), "serving_kv_cache_dtype": E["KV_CACHE_DTYPE"],
  "serving_gpus": E["GPU_IDS"], "training_gpu": int(E["TRAIN_GPU"]), "training_python": E["TRAIN_PY"], "training_tmux_session": E["TRAIN_SESSION"],
  "shape": E["SHAPE"], "input_len": int(E["IN_LEN"]), "output_len": int(E["OUT_LEN"]), "concurrency": int(E["CONC"]),
  "num_prompts_per_window": int(E["NUM_PROMPTS"]), "target_window_s": int(E["BASELINE_SECONDS"]), "sweep_mode": E["SWEEP_MODE"],
  "train_minutes_requested": float(E["TRAIN_MINUTES"]),
  "serving": {"before": before, "during": during, "after": after},
  "serving_delta_during_vs_before_pct": delta,
  "training_overlap_fraction_of_during_window": overlap,
  "training": train, "training_exit_code": E["TRAIN_RC"],
  "training_tok_s": train.get("tok_s_overall"),
  "training_tok_s_steady": (train.get("tok_s_steady_state") or {}).get("tok_s") if isinstance(train.get("tok_s_steady_state"), dict) else None,
  "training_started": bool(int(E["TRAIN_STARTED"])),
  "serving_launched_here": E["WE_LAUNCHED"] == "1",
  "timestamps": {"before": [E["BEFORE_T0"], E["BEFORE_T1"]], "during": [E["DURING_T0"], E["DURING_T1"]], "after": [E["AFTER_T0"], E["AFTER_T1"]],
                 "train_launch": E["TRAIN_LAUNCH_T"], "train_end": E["TRAIN_END_T"]},
  "dmon_files": {t: os.path.join(E["COT_DIR"], f"dmon_{t}.txt") for t in ("before", "during", "after") if os.path.exists(os.path.join(E["COT_DIR"], f"dmon_{t}.txt"))},
  "raw_serving_files": {t: os.path.join(E["COT_DIR"], f"serve_{t}.json") for t in ("calib", "before", "during", "after") if os.path.exists(os.path.join(E["COT_DIR"], f"serve_{t}.json"))},
  **flags,
  "hw_decisions": {k: v for k, v in dec.items() if not k.startswith("_")},
  "notes": [f"serving (TP{E['TP']} over PCIe on GPUs {E['GPU_IDS']}) and training (GPU {E['TRAIN_GPU']}) share the host PCIe fabric, CPU and power budget; interference is fabric/CPU contention, not SM contention",
            f"pessimistic_tp={flags.get('pessimistic_tp')}: the TP{E['TP']} server's own numbers are a lower bound on an ACS-suspected host (same-switch pair 0,1 ~21 GB/s vs cross-switch 0,2 ~38 GB/s all_reduce); the before/during DELTA is still a fair interference measure",
            "training_overlap_fraction < 1.0 means training ended before the DURING window did: raise TRAIN_MINUTES or lower BASELINE_SECONDS"],
  "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
p = os.path.join(E["RESULTS_ROOT"], "cotenancy.json")
with open(p, "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
shutil.copyfile(p, os.path.join(E["CELL_DIR"], "cotenancy.json"))
print(json.dumps({"before_output_tok_s": (before or {}).get("output_throughput"), "during_output_tok_s": (during or {}).get("output_throughput"),
                  "delta_output_pct": (delta or {}).get("output_throughput"), "training_tok_s": out["training_tok_s"],
                  "training_tok_s_steady": out["training_tok_s_steady"], "overlap": overlap, "pessimistic_tp": out["pessimistic_tp"]}, indent=2))
print("wrote", p)
PY

# ---- 6. sleep / wake on the same server, then optional teardown ------------------------------
if [[ "$RUN_SLEEP_WAKE" == 1 ]]; then
  say "sleep/wake timing on $CELL"
  AUTO_LAUNCH=0 STOP_AFTER=0 bash "$SCRIPT_DIR/sleep_wake.sh" "$CELL" || say "sleep_wake.sh exited non-zero (server started without --enable-sleep-mode?)"
fi
stop_if_needed
say "done: $RESULTS_ROOT/cotenancy.json  $RESULTS_ROOT/train_lora.json  $COT_DIR/"
