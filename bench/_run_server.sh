#!/usr/bin/env bash
# =============================================================================
# bench/_run_server.sh — internal runner used inside tmux windows by launch.sh.
#   _run_server.sh <logfile> <envfile> <command...>
# Sources <envfile> (KEY=VALUE, shell-quoted) so the server sees exactly the
# environment launch.sh resolved (tmux does not reliably inherit it), tees
# stdout+stderr to <logfile>, records the server PID in <logfile%.log>.pid (so
# stop.sh can address the process tree) and writes the exit code to
# <logfile%.log>.exit so launch.sh can detect a dead server while waiting on /health.
# The generated server_p<port>.sh exports BENCH_CELL=<cell> before exec-ing this
# script; every descendant (API server, EngineCore, TP workers, tee) inherits it,
# which is how stop.sh attributes orphaned GPU processes to a cell.
# =============================================================================
set -uo pipefail
LOG="$1"; ENVFILE="$2"; shift 2
PIDFILE="${LOG%.log}.pid"; EXITFILE="${LOG%.log}.exit"
mkdir -p "$(dirname "$LOG")"
rm -f "$EXITFILE"
if [ -f "$ENVFILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENVFILE"
  set +a
fi
{
  echo "[$(date -Is)] HOST=$(hostname) PWD=$PWD BENCH_CELL=${BENCH_CELL:-?} CUDA_VISIBLE_DEVICES(cmd)=$(printf '%s ' "$@" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' || true)"
  echo "[$(date -Is)] CMD: $*"
} | tee -a "$LOG"
"$@" > >(tee -a "$LOG") 2>&1 &
pid=$!
echo "$pid" > "$PIDFILE"
# Forward TERM/INT/HUP (tmux kill-session sends HUP) to the server so it can shut its
# workers down cleanly; stop.sh escalates to KILL if that takes too long.
trap 'kill -TERM "$pid" 2>/dev/null' TERM INT HUP
wait "$pid"; rc=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
echo "$rc" > "$EXITFILE"
echo "[$(date -Is)] EXIT $rc" | tee -a "$LOG"
rm -f "$PIDFILE"
exit "$rc"
