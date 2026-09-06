#!/usr/bin/env bash
# Event stream for the Monitor tool: one line when a box's chain prints its final marker, one line when a box
# stops answering for three polls in a row (a stop or crash), one line when a chain has no tmux session left
# without having finished. Polls every five minutes; silent otherwise.
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$(_p "$1")";; esac; }
declare -A MISS=( [a]=0 [c]=0 ) FIRED=( [a]=0 [c]=0 )
probe(){ # hostfile marker log -> prints "DONE"/"NOSESSION"/"RUNNING" or nothing on failure
  read -r _ H P _ < "$(_p "$1")"
  timeout 40 ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" \
    "if grep -q '$2' $3 2>/dev/null; then echo DONE; elif [ \$(tmux ls 2>/dev/null | wc -l) -eq 0 ]; then echo NOSESSION; else echo RUNNING; fi" 2>/dev/null
}
while true; do
  for k in a c; do
    case $k in
      a) hf=inst6000a.ssh; mark="CHAIN600W[567] DONE"; log=/workspace/results/chain600w.log; name="original 600 W box (49694407)";;
      c) hf=inst6000c.ssh; mark="CHAINC[23] DONE"; log=/workspace/results/chain_c.log; name="third box (49977359)";;
    esac
    [ "${FIRED[$k]}" = 1 ] && continue
    r=$(probe "$hf" "$mark" "$log")
    case "$r" in
      DONE) echo "DONE: $name finished its chain ($mark) - pull results and destroy it now"; FIRED[$k]=1;;
      NOSESSION) echo "DEAD: $name has no tmux session and no final marker - re-arm it (rearm_all.sh arm)"; MISS[$k]=0;;
      RUNNING) MISS[$k]=0;;
      *) MISS[$k]=$(( ${MISS[$k]} + 1 )); [ "${MISS[$k]}" -eq 3 ] && echo "UNREACHABLE: $name has not answered for 15 minutes - check vast_status.sh (stopped? endpoint changed?)";;
    esac
  done
  [ "${FIRED[a]}" = 1 ] && [ "${FIRED[c]}" = 1 ] && { echo "ALL DONE: both chains finished"; exit 0; }
  sleep 300
done
