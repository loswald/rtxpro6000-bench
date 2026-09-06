#!/usr/bin/env bash
# vast_destroy.sh <instance-id> <hostfile> <done-marker> <chain-log>
# Destroys a box ONLY if its chain has printed its DONE marker (checked over ssh) - never a box mid-run.
export PATH="$HOME/.local/bin:$PATH"
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$(_p "$1")";; esac; }
ID="$1"; HF="$2"; MARK="$3"; LOG="$4"
read -r _ H P _ < "$(_p "$HF")"
done_=$(ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" "grep -c '$MARK' $LOG 2>/dev/null; tmux ls 2>/dev/null | wc -l" 2>/dev/null | paste -sd" ")
set -- $done_; marks="${1:-0}"; sessions="${2:-?}"
echo "  $ID: marker '$MARK' seen $marks time(s); tmux sessions: $sessions"
if [ "$marks" -ge 1 ] 2>/dev/null; then
  echo y | vastai destroy instance "$ID" 2>&1 | tail -1 | sed 's/^/  /'
else
  echo "  not done - keeping"
fi
