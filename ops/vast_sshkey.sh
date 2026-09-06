#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
KEY="$HOME/.ssh/id_ed25519"; ID="${1:-49977359}"; HF="${2:-inst6000c.ssh}"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$SP/$1";; esac; }
read -r _ H P _ < "$(_p "$HF")"
try(){ timeout 25 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 "root@$H" 'echo "  ssh ok: $(hostname) $(nvidia-smi --query-gpu=name,power.limit,power.max_limit --format=csv,noheader | head -1)"' 2>&1 | tail -1; }
for i in 1 2 3; do out=$(try); case "$out" in *"ssh ok"*) echo "$out"; exit 0;; esac; echo "  attempt $i: $out"; sleep 30; done
echo "  attaching the account key to the instance and retrying"
vastai attach ssh "$ID" "$(cat "$KEY.pub")" 2>&1 | head -2 | sed 's/^/  /'
for i in 1 2 3 4; do sleep 30; out=$(try); case "$out" in *"ssh ok"*) echo "$out"; exit 0;; esac; echo "  attempt $i after attach: $out"; done
exit 1
