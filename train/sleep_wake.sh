#!/usr/bin/env bash
# =============================================================================
# train/sleep_wake.sh [cell | port] [label]
#
# vLLM sleep-mode timing.  The server MUST run with
#     VLLM_SERVER_DEV_MODE=1  (bench/env.sh exports it)  and  --enable-sleep-mode
#     (cells: ENABLE_SLEEP_MODE=1, the default in cotenant_tp2_gpu01.env).
#
# Endpoints, verified against vLLM main (vllm/entrypoints/serve/dev/{sleep,rpc}/api_router.py,
# vllm/v1/worker/gpu_worker.py, 2026-09-02):
#   POST /sleep?level=1|2      (query param `level`, default 1)
#   POST /wake_up[?tags=weights][&tags=kv_cache]   (`tags` via query_params.getlist; the
#                                                    allocator tags are "weights" and "kv_cache")
#   GET  /is_sleeping          -> {"is_sleeping": bool}
#   POST /collective_rpc       body {"method": "reload_weights", "args": [], "kwargs": {}, "timeout": null}
#   level 1 = weights offloaded to CPU RAM + KV cache discarded (fast resume; needs host RAM >= weights)
#   level 2 = weights discarded too -> after wake_up the weights must be reloaded (LEVEL2=1 does one cycle)
#
# Argument: a cell name (port = the cell's BASE_PORT via load_cell; TP/DP known -> pessimistic_tp)
# or a port number.  With a cell and AUTO_LAUNCH=1 (default) an unhealthy server is started
# through the REAL launcher:  ENABLE_SLEEP_MODE=1 bench/launch.sh <cell>   and stopped again at
# the end with bench/stop.sh <cell> --quiet (STOP_AFTER=auto: only if we launched it; 1 always; 0 never).
# Writes results/sleep_wake.json (+ results/<cell>/sleep_wake.json).
#
# Env: HOST=127.0.0.1 CYCLES=3 SPLIT_WAKE=0|1 LEVEL2=0|1 AUTO_LAUNCH=1 STOP_AFTER=auto|0|1
# Exit: 0 ok, 1 error during cycles, 3 server unhealthy / launch failed, 7 sleep endpoints missing
# =============================================================================
set -euo pipefail

ARG1="${1:-}"
LABEL_ARG="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
STOP_AFTER="${STOP_AFTER:-auto}"
say() { printf '[sleep_wake %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

unset RUN_TAG
if [[ -f "$ROOT/bench/env.sh" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/bench/env.sh"
fi
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
HW_DIR="${HW_DIR:-$RESULTS_ROOT/hw}"
export RESULTS_ROOT HW_DIR GATES_DIR="$ROOT/gates"
PORT="${PORT:-8000}"; CELL="${CELL:-unknown}"; CELL_DIR=""; CELL_ARG=""; TP=""; DP=""; REPLICAS=""
if [[ -n "$ARG1" && "$ARG1" =~ ^[0-9]+$ ]]; then
  PORT="$ARG1"
elif [[ -n "$ARG1" && -f "$ROOT/cells/$ARG1.env" ]] && declare -F load_cell >/dev/null; then
  load_cell "$ARG1"; CELL_ARG="$ARG1"; PORT="${PORTS[0]}"; CELL="$CELL_NAME"; CELL_DIR="$RESULTS_DIR"
elif [[ -n "$ARG1" ]]; then
  CELL="$ARG1"
fi
[[ -n "$LABEL_ARG" ]] && CELL="$LABEL_ARG"
mkdir -p "$RESULTS_ROOT" ${CELL_DIR:+"$CELL_DIR"}
BASE="http://$HOST:$PORT"

# ---- server: healthy, or launched through bench/launch.sh --------------------------------
WE_LAUNCHED=0
if ! curl -fsS -m 5 "$BASE/health" >/dev/null 2>&1; then
  if [[ -n "$CELL_ARG" && "$AUTO_LAUNCH" == 1 && -f "$ROOT/bench/launch.sh" ]]; then
    say "no healthy server at $BASE; launching $CELL_ARG with ENABLE_SLEEP_MODE=1 via bench/launch.sh"
    ENABLE_SLEEP_MODE=1 bash "$ROOT/bench/launch.sh" "$CELL_ARG" --no-smoke || { say "launch failed (see $CELL_DIR/launch.json)"; exit 3; }
    WE_LAUNCHED=1
  else
    say "no healthy server at $BASE (ENABLE_SLEEP_MODE=1 bench/launch.sh <cell> first, or pass a cell name with AUTO_LAUNCH=1)"; exit 3
  fi
fi
stop_if_needed() {
  if [[ -n "$CELL_ARG" ]] && { [[ "$STOP_AFTER" == 1 ]] || { [[ "$STOP_AFTER" == auto ]] && [[ "$WE_LAUNCHED" == 1 ]]; }; }; then
    say "stopping $CELL_ARG (STOP_AFTER=$STOP_AFTER, launched here=$WE_LAUNCHED)"
    bash "$ROOT/bench/stop.sh" "$CELL_ARG" --quiet || true
  fi
}
trap stop_if_needed EXIT

export BASE CELL CYCLES="${CYCLES:-3}" SPLIT_WAKE="${SPLIT_WAKE:-0}" LEVEL2="${LEVEL2:-0}" \
       OUT="$RESULTS_ROOT/sleep_wake.json" OUT_CELL="${CELL_DIR:+$CELL_DIR/sleep_wake.json}" \
       SW_TP="$TP" SW_DP="$DP" SW_REPLICAS="$REPLICAS" SW_CELL_DIR="$CELL_DIR" SW_WE_LAUNCHED="$WE_LAUNCHED"

python3 - <<'PY'
import json, os, shutil, subprocess, sys, time, datetime as dt, urllib.request, urllib.error
sys.path.insert(0, os.environ["GATES_DIR"])
from hwdecisions import hw_decisions, pessimistic_flags
E = os.environ
base, out = E["BASE"], E["OUT"]
cycles, split_wake, level2 = int(E["CYCLES"]), E["SPLIT_WAKE"] == "1", E["LEVEL2"] == "1"

def call(path, method="POST", payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return time.perf_counter() - t0, body

def smi():
    try:
        txt = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=10)
        return [{"gpu": int(i), "used_mib": int(float(u)), "total_mib": int(float(t))} for i, u, t in (l.split(", ") for l in txt.strip().splitlines())]
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]
def host_ram():
    try:
        txt = subprocess.check_output(["free", "-g"], text=True, timeout=10).splitlines()[1].split()
        return {"total_gb": int(txt[1]), "used_gb": int(txt[2]), "available_gb": int(txt[-1])}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
def used_total(rows): return sum(r.get("used_mib", 0) for r in rows)
def is_sleeping():
    _, body = call("/is_sleeping", "GET", timeout=30); return bool(json.loads(body).get("is_sleeping"))
def poll(target, timeout=1800):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if is_sleeping() == target: return time.perf_counter() - t0
        time.sleep(0.2)
    raise TimeoutError(f"is_sleeping never became {target}")
def write(results):
    with open(out, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    if E.get("OUT_CELL"): shutil.copyfile(out, E["OUT_CELL"])

launch = None
if E.get("SW_CELL_DIR"):
    try: launch = json.load(open(os.path.join(E["SW_CELL_DIR"], "launch.json"), encoding="utf-8"))
    except Exception: pass  # noqa: BLE001
tp = E.get("SW_TP") or (launch or {}).get("tp")
dec = hw_decisions()
flags = pessimistic_flags(dec, tp, E.get("SW_DP") or (launch or {}).get("dp"), E.get("SW_REPLICAS") or (launch or {}).get("replicas"))
model = json.loads(call("/v1/models", "GET", timeout=30)[1])["data"][0]["id"]
try: version = json.loads(call("/version", "GET", timeout=10)[1])
except Exception: version = None  # noqa: BLE001
now = lambda: dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
common = {"cell": E["CELL"], "model": model, "engine_version": version, "base_url": base, "launched_here": E.get("SW_WE_LAUNCHED") == "1",
          "launch": {k: (launch or {}).get(k) for k in ("engine_version", "tp", "dp", "gpu_ids", "kv_cache_dtype", "seconds_to_ready", "status")} if launch else None,
          **flags, "host_ram": host_ram()}
try:
    asleep0 = is_sleeping()
except urllib.error.HTTPError as e:
    if e.code in (404, 405):
        write({**common, "status": "sleep_endpoints_unavailable", "http": e.code,
               "fix": "server needs VLLM_SERVER_DEV_MODE=1 and --enable-sleep-mode (ENABLE_SLEEP_MODE=1 bench/launch.sh <cell>)", "written_utc": now()})
        sys.exit(7)
    raise
if asleep0:
    print("[sleep_wake] server already asleep; waking first", file=sys.stderr); call("/wake_up"); poll(False)

def probe():
    t, body = call("/v1/completions", "POST", {"model": model, "prompt": "The quick brown fox jumps over the", "max_tokens": 16, "temperature": 0, "seed": 1234}, timeout=600)
    return t, json.loads(body)["choices"][0]["text"]

def cycle(level):
    rec = {"level": level, "mem_awake_before": smi(), "host_ram_before": host_ram()}
    rec["probe_before_s"] = round(probe()[0], 3)
    t, _ = call(f"/sleep?level={level}"); rec["sleep_call_s"] = round(t, 3)
    rec["sleep_settle_s"] = round(poll(True), 3); time.sleep(1.0); rec["mem_asleep"] = smi(); rec["host_ram_asleep"] = host_ram()
    if split_wake:
        tw, _ = call("/wake_up?tags=weights"); rec["wake_weights_s"] = round(tw, 3); rec["mem_after_weights"] = smi()
        tk, _ = call("/wake_up?tags=kv_cache"); rec["wake_kv_cache_s"] = round(tk, 3); rec["wake_call_s"] = round(tw + tk, 3)
    else:
        t, _ = call("/wake_up"); rec["wake_call_s"] = round(t, 3)
    rec["wake_settle_s"] = round(poll(False), 3)
    if level == 2:
        t, body = call("/collective_rpc", "POST", {"method": "reload_weights", "args": [], "kwargs": {}}); rec["reload_weights_s"] = round(t, 3)
    rec["mem_awake_after"] = smi()
    t, text = probe(); rec["first_request_after_wake_s"] = round(t, 3); rec["probe_text"] = text
    rec["mem_freed_mib_total"] = used_total(rec["mem_awake_before"]) - used_total(rec["mem_asleep"])
    rec["mem_restored_delta_mib_total"] = used_total(rec["mem_awake_after"]) - used_total(rec["mem_awake_before"])
    return rec

results = {**common, "status": "ok", "cycles": [], "level2_cycles": [], "started_utc": now()}
try:
    for i in range(cycles):
        print(f"[sleep_wake] level-1 cycle {i + 1}/{cycles}", file=sys.stderr); results["cycles"].append(cycle(1))
    if level2:
        print("[sleep_wake] level-2 cycle (+ collective_rpc reload_weights)", file=sys.stderr); results["level2_cycles"].append(cycle(2))
except Exception as e:  # noqa: BLE001
    results["status"], results["error"] = "error", repr(e)

def summarise(cs):
    if not cs: return None
    def m(k):
        v = [c[k] for c in cs if isinstance(c.get(k), (int, float))]; return round(sum(v) / len(v), 3) if v else None
    return {"n": len(cs), "sleep_call_s_mean": m("sleep_call_s"), "sleep_settle_s_mean": m("sleep_settle_s"),
            "wake_call_s_mean": m("wake_call_s"), "wake_settle_s_mean": m("wake_settle_s"),
            "wake_weights_s_mean": m("wake_weights_s"), "wake_kv_cache_s_mean": m("wake_kv_cache_s"), "reload_weights_s_mean": m("reload_weights_s"),
            "first_request_after_wake_s_mean": m("first_request_after_wake_s"), "probe_before_s_mean": m("probe_before_s"),
            "mem_freed_mib_total_mean": m("mem_freed_mib_total"),
            "mem_awake_before_gb_per_gpu": [round(r["used_mib"] / 1024, 1) for r in cs[0]["mem_awake_before"] if "used_mib" in r],
            "mem_asleep_gb_per_gpu": [round(r["used_mib"] / 1024, 1) for r in cs[0]["mem_asleep"] if "used_mib" in r]}
results["summary_level1"] = summarise(results["cycles"]); results["summary_level2"] = summarise(results["level2_cycles"])
results["notes"] = ["level 1 offloads weights to host RAM: the host_ram_asleep.used_gb delta is the RAM cost of parking the model",
                    "compare wake_call_s + first_request_after_wake_s with launch.json.seconds_to_ready for a cold start"]
results["written_utc"] = now()
write(results)
print(json.dumps({"status": results["status"], "summary_level1": results["summary_level1"], "pessimistic_tp": results.get("pessimistic_tp")}, indent=2)); print("wrote", out)
sys.exit(0 if results["status"] == "ok" else 1)
PY
