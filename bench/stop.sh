#!/usr/bin/env bash
# =============================================================================
# bench/stop.sh [<cell> | --all] [--quiet]
#
#   stop.sh <cell>   stop everything that belongs to ONE cell: the tmux session bench_<cell>
#                    (every window), the vLLM/SGLang API server(s), their EngineCore / TP
#                    worker subprocesses — found through the BENCH_CELL marker launch.sh puts
#                    in the process environment, `--served-model-name <alias>` on the command
#                    line, or CUDA_VISIBLE_DEVICES on the cell's GPUs for anything vllm-like —
#                    the cell's rr_proxy and its own dmon sampler (exact `-i <GPU_IDS>` match, so
#                    train/lora_cotenant.sh's `-i $GPU_IDS,$TRAIN_GPU` sampler survives).  Graceful TERM to the API
#                    server first, then TERM/KILL of whatever is left, then waits for the
#                    cell's GPUs to release memory.  Other cells and a co-tenant trainer on
#                    other GPUs are left alone.
#   stop.sh --all    kill every bench_* session, every vllm/sglang process, proxies, dmon
#                    samplers and stray bench clients on the box.  Must be spelled out: a bare
#                    `stop.sh` prints this usage and exits 1.
#                    NOTE: also kills a co-tenant training job.  The nvidia-smi last resort only
#                    kills PIDs whose /proc cmdline matches the bench pattern or whose environment
#                    carries BENCH_CELL (nvidia-smi shows HOST pids inside a container).
#   --list           only print the processes that WOULD be killed (pid, kind, cmdline) and exit.
#
# Env: STOP_GRACE_S=30 (graceful window before escalating), GPU_IDLE_MIB=1500 (memory.used
#      below this on every cell GPU = drained)
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

TARGET=""; QUIET=0; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quiet) QUIET=1 ;;
    --all)   TARGET="--all" ;;
    --list)  LIST=1 ;;
    -h|--help) sed -n '3,/^# =\{5,\}/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 1 ;;   # header block
    *) TARGET="$1" ;;
  esac
  shift
done
# No target = usage, never --all: a bare `bench/stop.sh` (typo, forgotten cell name) must not kill the
# co-tenant trainer and every other cell on the box.  --all has to be spelled out.
[ -n "$TARGET" ] || { sed -n '3,/^# =\{5,\}/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 1; }
say() { if [ "$QUIET" = 0 ]; then log "$@"; fi; }
STOP_GRACE_S="${STOP_GRACE_S:-30}"
GPU_IDLE_MIB="${GPU_IDLE_MIB:-1500}"
# anything that may hold a GPU or belong to a bench run
PATTERN='vllm serve|vllm\.entrypoints|VLLM::|EngineCore|sglang\.launch_server|sglang::|rr_proxy\.py|nvidia-smi dmon|vllm bench|benchmark_serving\.py|_run_server\.sh|server_p[0-9]+\.sh'

# find_pids <all|cell> -> lines "pid<TAB>server|other<TAB>cmdline"  (attribution rules in the header)
find_pids() {
  python3 - "$1" "${CELL_NAME:-}" "${SERVED_MODEL_NAME:-}" "${GPU_IDS:-}" "$PROXY_PORT" "${REPLICAS:-1}" "$PATTERN" <<'PY' 2>/dev/null || true
import os, re, sys
mode, cell, alias, gpu_ids, proxy_port, replicas, pattern = sys.argv[1:8]
gpus = {g for g in gpu_ids.split(",") if g}
pat = re.compile(pattern)
server_pat = re.compile(r"vllm serve|sglang\.launch_server|vllm\.entrypoints\.openai\.api_server")
protect = {os.getpid(), os.getppid()}
pid = os.getpid()
while pid > 1:  # never touch our own ancestors (the calling shell / launch.sh / tmux)
    try:
        with open(f"/proc/{pid}/stat") as f:
            pid = int(f.read().rsplit(")", 1)[1].split()[1])
    except Exception:
        break
    protect.add(pid)
for d in os.listdir("/proc"):
    if not d.isdigit():
        continue
    p = int(d)
    if p in protect:
        continue
    try:
        with open(f"/proc/{p}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except Exception:
        continue
    if not cmd or "bench/stop.sh" in cmd or "stop.sh --" in cmd:
        continue  # never another stop.sh (it carries the pattern list on its command line)
    env = {}
    try:
        with open(f"/proc/{p}/environ", "rb") as f:
            for kv in f.read().split(b"\0"):
                if b"=" in kv:
                    k, v = kv.split(b"=", 1)
                    env[k.decode(errors="replace")] = v.decode(errors="replace")
    except Exception:
        pass
    hit = False
    if mode == "all":
        hit = bool(pat.search(cmd))
    else:
        if cell and env.get("BENCH_CELL") == cell:
            hit = True
        elif alias and re.search(r"served-model-name %s( |$)" % re.escape(alias), cmd):
            hit = True
        elif pat.search(cmd):
            cvd = env.get("CUDA_VISIBLE_DEVICES")
            if cvd is not None and gpus and (set(cvd.split(",")) & gpus):
                hit = True
            elif "rr_proxy.py" in cmd and replicas != "1" and re.search(r"--port %s( |$)" % re.escape(proxy_port), cmd):
                hit = True
            elif "nvidia-smi dmon" in cmd and gpu_ids and re.search(r"nvidia-smi dmon\b.*\s-i\s+%s(\s|$)" % re.escape(gpu_ids), cmd):
                hit = True   # exact `-i <GPU_IDS>` token only: a co-tenancy sampler on "-i 0,2,3" is NOT this cell's
    if hit:
        print(f"{p}\t{'server' if server_pat.search(cmd) else 'other'}\t{cmd[:160]}")
PY
}
alive() { kill -0 "$1" 2>/dev/null; }
kill_list() {  # <signal> <pid...>
  local sig="$1"; shift
  local p
  for p in "$@"; do kill "-$sig" "$p" 2>/dev/null || true; done
}
collect() {  # [verbose] -> SERVERS / OTHERS arrays from find_pids (verbose: say each match)
  SERVERS=(); OTHERS=()
  local pid kind cmd
  while IFS=$'\t' read -r pid kind cmd; do
    [ -n "$pid" ] || continue
    if [ "$kind" = server ]; then SERVERS+=( "$pid" ); else OTHERS+=( "$pid" ); fi
    if [ "${1:-}" = verbose ]; then say "  $kind pid $pid: ${cmd:0:110}"; fi
  done < <(find_pids "$MODE")
}
gpu_drained() {
  if [ "$MODE" = all ]; then
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)" = 0 ]
  else
    local used u
    used="$(nvidia-smi -i "$GPU_IDS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    [ -n "$used" ] || return 0
    for u in $used; do
      case "$u" in ''|*[!0-9]*) continue ;; esac
      [ "$u" -lt "$GPU_IDLE_MIB" ] || return 1
    done
    return 0
  fi
}
gpu_report() {
  if [ "$MODE" = all ]; then nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';'
  else nvidia-smi -i "$GPU_IDS" --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';'; fi
}

if [ "$TARGET" = "--all" ]; then
  MODE=all
else
  MODE=cell
  load_cell "$TARGET"
fi
if [ "$LIST" = 1 ]; then
  echo "processes stop.sh $TARGET would kill (pid, kind, cmdline):"
  find_pids "$MODE" | sed 's/^/  /'
  exit 0
fi
if [ "$MODE" = all ]; then
  say "stopping ALL bench sessions and GPU servers (co-tenant trainer included)"
  for s in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^bench_' || true); do
    tmux kill-session -t "$s" 2>/dev/null || true
  done
else
  say "stopping cell $CELL_NAME (session $SESSION, served-model-name $SERVED_MODEL_NAME, GPUs $GPU_IDS)"
fi

# 1. graceful: TERM the API servers (they shut their EngineCore/workers down), then the session
SERVERS=(); OTHERS=()
collect verbose
say "found ${#SERVERS[@]} server process(es) and ${#OTHERS[@]} related process(es)"
kill_list TERM ${SERVERS[@]+"${SERVERS[@]}"}
if [ "$MODE" = cell ]; then tmux kill-session -t "$SESSION" 2>/dev/null || true; fi
deadline=$(( $(date +%s) + STOP_GRACE_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  remaining=0
  for p in ${SERVERS[@]+"${SERVERS[@]}"} ${OTHERS[@]+"${OTHERS[@]}"}; do
    if alive "$p"; then remaining=$((remaining + 1)); fi
  done
  if [ "$remaining" = 0 ] && gpu_drained; then break; fi
  sleep 2
done

# 2. re-scan (re-parented children, orphans) -> TERM, then KILL what survives
collect
LEFT=( ${SERVERS[@]+"${SERVERS[@]}"} ${OTHERS[@]+"${OTHERS[@]}"} )
if [ ${#LEFT[@]} -gt 0 ]; then
  say "TERM ${#LEFT[@]} leftover process(es): ${LEFT[*]}"
  kill_list TERM "${LEFT[@]}"
  sleep 5
  collect
  LEFT=( ${SERVERS[@]+"${SERVERS[@]}"} ${OTHERS[@]+"${OTHERS[@]}"} )
  if [ ${#LEFT[@]} -gt 0 ]; then
    say "KILL ${#LEFT[@]} process(es) still alive: ${LEFT[*]}"
    kill_list KILL "${LEFT[@]}"
    sleep 3
  fi
fi
# 3. --all: last resort via nvidia-smi.  Inside a Vast container nvidia-smi reports HOST pids; the same
#    number in our PID namespace may be an unrelated process (tmux server, sshd, a download session).
#    A pid is killed ONLY when /proc/<pid>/cmdline matches $PATTERN or /proc/<pid>/environ carries
#    BENCH_CELL; anything else is reported and left alone.
if [ "$MODE" = all ] && ! gpu_drained; then
  leftover=""
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    if [ ! -d "/proc/$pid" ]; then leftover="$leftover $pid(not-in-this-pid-namespace)"; continue; fi
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if printf '%s' "$cmd" | grep -qE "$PATTERN" || tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q '^BENCH_CELL='; then
      say "KILL GPU pid $pid (bench process): ${cmd:0:100}"
      kill -9 "$pid" 2>/dev/null || true
    else
      leftover="$leftover $pid(${cmd:0:40})"
    fi
  done
  [ -z "$leftover" ] || say "note: GPU pids that are not ours (or not resolvable here) were left alone:$leftover"
  sleep 3
fi

# 4. final drain wait + report
for _ in $(seq 1 15); do
  if gpu_drained; then break; fi
  sleep 2
done
if ! gpu_drained; then
  if [ "$MODE" = cell ]; then
    say "note: GPU memory still in use on GPUs $GPU_IDS (another cell or the training job?): $(gpu_report)"
  else
    say "note: GPU processes remain: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  fi
fi
if [ "$MODE" = cell ]; then rm -f "$RESULTS_DIR"/server_p*.pid 2>/dev/null || true; fi
say "done. GPU memory now: $(gpu_report)"
