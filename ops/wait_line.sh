#!/usr/bin/env bash
# wait_line.sh <hostfile> <remote-file> <regex> [max-minutes]: exit when the regex appears in the file; prints the last matching lines.
KEY="$HOME/.ssh/id_ed25519"
SP=/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad
read -r _ H P _ < "$SP/$1"; F="$2"; RE="$3"; N="${4:-90}"
for i in $(seq 1 "$N"); do
  out=$(timeout 40 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" "grep -E '$RE' $F 2>/dev/null | tail -3" 2>/dev/null)
  if [ -n "$out" ]; then date -u +"%H:%M UTC"; echo "$out" | cut -c1-200; exit 0; fi
  sleep 60
done
echo "TIMEOUT waiting for $RE in $F"; exit 1
