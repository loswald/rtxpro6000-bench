#!/usr/bin/env bash
# sync.sh -- move the harness to a Vast.ai instance and pull results back (run on your laptop).
#
#   ./vast/sync.sh push  [ID]            local rtxpro6000-bench/ -> root@instance:/workspace/rtxpro6000-bench  (results/ and models excluded)
#   ./vast/sync.sh pull  [ID]            instance:/workspace/rtxpro6000-bench/results/ -> local results/  (+ results/hw/decisions.env, env.sh, logs)
#
# Remote layout on the box (do not point REMOTE_ROOT at the other two):
#   /workspace/rtxpro6000-bench   this harness (REMOTE_ROOT)        /workspace/models   weights, hf download --local-dir layout
#   /workspace/bench              scratch scripts already on the box (not ours; never overwritten)
#   ./vast/sync.sh watch [ID] [SECS]     pull every SECS seconds (default 300) until Ctrl-C
#   ./vast/sync.sh ssh   [ID] [CMD...]   interactive shell, or run CMD remotely
#   ./vast/sync.sh tmux  [ID] [SESSION]  attach to a tmux session on the instance (default: downloads)
#   ./vast/sync.sh url   [ID]            print host/port only
#
# ID defaults to $VAST_INSTANCE_ID, else the single *running* instance on the account.
# Needs: vastai CLI (api key set via 'vastai set api-key' -- never put the key in files), ssh.
# rsync is used when present locally (Git Bash on Windows usually lacks it -> falls back to tar-over-ssh).
# Works from Git Bash / WSL / macOS / Linux.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$HERE/.." && pwd)}"      # .../rtxpro6000-bench
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/rtxpro6000-bench}"   # /workspace/bench is the box's own scratch dir, /workspace/models the weights
KNOWN_HOSTS="${KNOWN_HOSTS:-$HOME/.ssh/known_hosts_vast}"   # Vast hosts/ports are recycled; keep them out of your main known_hosts
SSH_KEY_OPT=""; [ -n "${VAST_SSH_KEY:-}" ] && SSH_KEY_OPT="-i $VAST_SSH_KEY"
SSH_OPTS="-p __PORT__ $SSH_KEY_OPT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=20"

die() { echo "sync.sh: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
have vastai || die "vastai CLI not found (pip install vastai  or  uv tool install vastai; then: vastai set api-key <KEY>)"
# a *working* python for JSON parsing (the Windows Store 'python3' alias exists but fails); uv can supply one
find_python() {
  local c; for c in python3 python "py -3"; do $c -c 'import sys' >/dev/null 2>&1 && { echo "$c"; return 0; }; done
  have uv && { echo "uv run --no-project -q python"; return 0; }; return 1
}
PY="$(find_python || true)"

cmd="${1:-}"; shift || true
ID="${1:-${VAST_INSTANCE_ID:-}}"; [ $# -gt 0 ] && shift || true

resolve_id() {
  if [ -z "$ID" ]; then
    local ids
    if [ -n "$PY" ]; then
      ids="$(vastai show instances --raw | $PY -c 'import sys,json; print("\n".join(str(i["id"]) for i in json.load(sys.stdin) if i.get("actual_status")=="running"))' 2>/dev/null)"
    else
      ids="$(vastai show instances 2>/dev/null | awk 'NR>1 && $2=="running"{print $1}')"
    fi
    [ -n "$ids" ] || die "no running instance found; pass ID explicitly"
    [ "$(echo "$ids" | wc -l | tr -d ' ')" = "1" ] || die "several running instances ($(echo $ids | tr '\n' ' ')); pass ID explicitly"
    ID="$ids"
  fi
}

resolve_hostport() {
  # vastai ssh-url prints ssh://root@HOST:PORT (direct IP:port when the instance was created with --direct)
  local url
  url="$(vastai ssh-url "$ID" 2>/dev/null | tr -d '\r' | tail -1)"
  if [[ "$url" =~ ^ssh://root@([^:]+):([0-9]+) ]]; then
    HOST="${BASH_REMATCH[1]}"; PORT="${BASH_REMATCH[2]}"
  else
    # fallback: raw JSON
    local js; js="$(vastai show instance "$ID" --raw)"
    HOST="$(echo "$js" | grep -o '"ssh_host": *"[^"]*"' | head -1 | sed 's/.*: *"//;s/"$//')"
    PORT="$(echo "$js" | grep -o '"ssh_port": *[0-9]*' | head -1 | grep -o '[0-9]*$')"
  fi
  [ -n "$HOST" ] && [ -n "$PORT" ] || die "could not resolve ssh host/port for instance $ID (is it running? vastai show instance $ID)"
  SSH_OPTS="${SSH_OPTS/__PORT__/$PORT}"
  mkdir -p "$(dirname "$KNOWN_HOSTS")"
}

rssh() { ssh $SSH_OPTS "root@$HOST" "$@"; }

wait_ssh() {
  local n=0
  until rssh -o BatchMode=yes true 2>/dev/null; do
    n=$((n+1)); [ $n -ge 30 ] && die "ssh to root@$HOST:$PORT not answering after 5 min"
    echo "waiting for sshd on $HOST:$PORT ... ($n)"; sleep 10
  done
}

EXCLUDES=(--exclude 'results/' --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' --exclude 'models/' --exclude '*.safetensors' --exclude '.venv/' --exclude 'node_modules/')

do_push() {
  wait_ssh
  echo "push $LOCAL_ROOT/ -> root@$HOST:$PORT:$REMOTE_ROOT/"
  rssh "mkdir -p $REMOTE_ROOT/results"
  if have rsync && rssh 'command -v rsync >/dev/null'; then
    rsync -az --info=progress2 --delete "${EXCLUDES[@]}" -e "ssh $SSH_OPTS" "$LOCAL_ROOT/" "root@$HOST:$REMOTE_ROOT/"
  else
    echo "(rsync not available on one side -> tar over ssh; no --delete)"
    tar czf - -C "$LOCAL_ROOT" --exclude='./results' --exclude='./.git' --exclude='__pycache__' --exclude='./models' . \
      | rssh "tar xzf - -C $REMOTE_ROOT"
  fi
  # Windows editors leave CRLF and drop the exec bit; fix both remotely so bash does not choke.
  # cells/*.env are sourced by bash (a stray \r would turn GPU_IDS=0,2,1,3 into CUDA_VISIBLE_DEVICES garbage), *.md for tidiness.
  rssh "cd $REMOTE_ROOT && find . -type f \( -name '*.sh' -o -name '*.py' -o -name '*.env' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.txt' -o -name '*.jsonl' \) -not -path './results/*' -exec sed -i 's/\r\$//' {} + ; find . -name '*.sh' -not -path './results/*' -exec chmod +x {} + ; echo pushed: \$(find . -type f -not -path './results/*' | wc -l) files"
}

do_pull() {
  wait_ssh
  mkdir -p "$LOCAL_ROOT/results"
  echo "pull root@$HOST:$PORT:$REMOTE_ROOT/results/ -> $LOCAL_ROOT/results/"
  if have rsync && rssh 'command -v rsync >/dev/null'; then
    rsync -az --info=progress2 -e "ssh $SSH_OPTS" "root@$HOST:$REMOTE_ROOT/results/" "$LOCAL_ROOT/results/"
    rsync -az -e "ssh $SSH_OPTS" --include='env.sh' --include='*.log' --exclude='*' "root@$HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/results/instance-logs/" 2>/dev/null || true
  else
    rssh "cd $REMOTE_ROOT && tar czf - results \$(ls env.sh *.log 2>/dev/null)" | tar xzf - -C "$LOCAL_ROOT"
  fi
  # keep a copy of the onstart log for the record
  rssh "cat /workspace/onstart.log 2>/dev/null" > "$LOCAL_ROOT/results/onstart-$ID.log" 2>/dev/null || true
  echo "results now local: $(find "$LOCAL_ROOT/results" -name '*.json' | wc -l | tr -d ' ') json files"
}

case "$cmd" in
  push)  resolve_id; resolve_hostport; do_push ;;
  pull)  resolve_id; resolve_hostport; do_pull ;;
  watch) resolve_id; resolve_hostport; SECS="${1:-300}"; while true; do do_pull || true; echo "next pull in ${SECS}s (Ctrl-C to stop)"; sleep "$SECS"; done ;;
  ssh)   resolve_id; resolve_hostport; if [ $# -gt 0 ]; then rssh "$@"; else ssh $SSH_OPTS -t "root@$HOST" "cd $REMOTE_ROOT 2>/dev/null; exec bash -l"; fi ;;
  tmux)  resolve_id; resolve_hostport; ssh $SSH_OPTS -t "root@$HOST" "tmux attach -t ${1:-downloads} || tmux ls" ;;
  url)   resolve_id; resolve_hostport; echo "ssh -p $PORT root@$HOST"; echo "id=$ID host=$HOST port=$PORT" ;;
  *)     sed -n '2,15p' "$0"; exit 2 ;;
esac
