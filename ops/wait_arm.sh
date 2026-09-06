#!/usr/bin/env bash
# wait_arm.sh <hostfile> <remote-log> <arm-regex> [max-minutes]
# Exit (one notification) when the named arm's router line, or its failure, appears after its launch line.
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$(_p "$1")";; esac; }
read -r _ H P _ < "$(_p "$1")"; LOG="$2"; ARM="$3"; N="${4:-60}"
for i in $(seq 1 "$N"); do
  out=$(timeout 40 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" \
    "awk '/$ARM/{f=1} f' $LOG 2>/dev/null | grep -E 'router|promptopt|FAILED|failed|healthy|skip|DONE' | head -6" 2>/dev/null)
  if echo "$out" | grep -qE "^\s*router|FAILED|failed|DONE"; then date -u +"%H:%M UTC"; echo "$out"; exit 0; fi
  sleep 60
done
echo "TIMEOUT waiting for $ARM"; exit 1
