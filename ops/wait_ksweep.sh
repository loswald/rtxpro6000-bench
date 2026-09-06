#!/usr/bin/env bash
# wait_ksweep.sh <hostfile> <ksweep-log> <tag> [max-minutes]: exit when the named ksweep arm prints its first router
# line, or every kernel combination for it fails ("NO combination served"), or the log reports a Python error.
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$(_p "$1")";; esac; }
read -r _ H P _ < "$(_p "$1")"; LOG="$2"; TAG="$3"; N="${4:-60}"
for i in $(seq 1 "$N"); do
  out=$(timeout 40 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" \
    "awk '/########## $TAG /{f=1} f' $LOG 2>/dev/null | grep -E 'router|promptopt|healthy| x |served|Error|quality20' | head -8" 2>/dev/null)
  if echo "$out" | grep -qE "^\s*router|served"; then date -u +"%H:%M UTC"; echo "$out" | cut -c1-260; exit 0; fi
  sleep 60
done
echo "TIMEOUT waiting for $TAG"; exit 1
