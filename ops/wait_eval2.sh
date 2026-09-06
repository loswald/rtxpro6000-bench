#!/usr/bin/env bash
# wait_eval2.sh <hostfile> <eval-json-path> <chain-log> [max-minutes]: exit when the eval JSON is complete
# (the runner checkpoints partial results under the final name, so partial=False is the signal) or the chain ends.
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$SP/$1";; esac; }
read -r _ H P _ < "$(_p "$1")"; F="$2"; CL="$3"; N="${4:-90}"
for i in $(seq 1 "$N"); do
  out=$(timeout 40 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" \
    "[ -f $F ] && python3 -c \"import json; d=json.load(open('$F')); a=d.get('aggregate',{}); print('n', a.get('n_scored'), 'acc', a.get('acc_micro'), 'partial', d.get('partial'), 'wall', d.get('wall_s')); fam=a.get('by_family') or a.get('families') or {}; print({k:(round(v['acc'],3) if isinstance(v,dict) and 'acc' in v else v) for k,v in fam.items()} if fam else '')\" 2>/dev/null; grep -hE 'DONE' $CL 2>/dev/null | tail -1" 2>/dev/null)
  if echo "$out" | grep -qE "partial False|DONE"; then date -u +"%H:%M UTC"; echo "$out"; exit 0; fi
  sleep 60
done
echo "TIMEOUT waiting for $F"; exit 1
