#!/usr/bin/env bash
# wait_glob_eval.sh <hostfile> <remote-glob> [max-minutes]: exit when any eval JSON matching the glob is complete.
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$SP/$1";; esac; }
read -r _ H P _ < "$(_p "$1")"; G="$2"; N="${3:-90}"
for i in $(seq 1 "$N"); do
  out=$(timeout 40 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" \
    "for f in $G; do case \"\$f\" in *.run.json) continue;; esac; [ -f \"\$f\" ] && python3 -c \"import json; d=json.load(open('\$f')); a=d['aggregate']; print('\$f'.split('/')[-1], 'n', a.get('n_scored'), 'acc', a.get('acc_micro'), 'partial', d.get('partial'), 'trunc', a.get('trunc_rate'))\"; done 2>/dev/null" 2>/dev/null)
  if echo "$out" | grep -q "partial False"; then date -u +"%H:%M UTC"; echo "$out"; exit 0; fi
  sleep 60
done
echo "TIMEOUT waiting for $G"; exit 1
