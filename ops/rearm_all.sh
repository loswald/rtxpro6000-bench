#!/usr/bin/env bash
# Bring both boxes back after a Vast stop and re-arm every chain. Safe to run repeatedly: every chain step is
# idempotent - ksweep skips arms that already have their judge files or eval JSON, glm_eval skips finished
# tags, kldiff skips measured pairs - so a chain restarted from the top resumes where the stop cut it.
#
#   rearm_all.sh start     start both instances (needs a positive balance) and wait for SSH
#   rearm_all.sh arm       re-create the tmux chains on both boxes
export PATH="$HOME/.local/bin:$PATH"
KEY="$HOME/.ssh/id_ed25519"
SP="${SP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; _p(){ case "$1" in /*) echo "$1";; *) for d in "$SP" "${SCRATCH:-}" "/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad"; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$(_p "$1")";; esac; }
A=(49694407 inst6000a.ssh)   # 600 W Server Edition
B=(49774868 inst6000b.ssh)   # 400 W Workstation Edition (stopped, cards re-rented)
C=(49977359 inst6000c.ssh)   # 600 W Server Edition, the third box (Bulgaria)
sshto(){ # hostfile cmd
  read -r _ H P _ < "$(_p "$1")"; shift
  ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "root@$H" "$@"
}
refresh_endpoints(){ # a restarted instance gets new ports; rewrite the host files from Vast's records
  for id_file in "49694407 inst6000a.ssh" "49774868 inst6000b.ssh" "49977359 inst6000c.ssh"; do
    set -- $id_file
    vastai show instance "$1" --raw 2>/dev/null | python3 -W ignore -c '
import json,sys
i=json.load(sys.stdin); ports=i.get("ports") or {}
p22=(ports.get("22/tcp") or [{}])[0].get("HostPort")
if p22 and i.get("public_ipaddr"): print("direct %s %s"%(i["public_ipaddr"],p22))
else: print("proxy %s %s"%(i.get("ssh_host"),i.get("ssh_port")))' > "$(_p "$2.new")" && mv "$(_p "$2.new")" "$(_p "$2")"
    echo "  $2 -> $(cat "$(_p "$2")")"
  done
}
case "${1:-status}" in
  start)
    for id in "${A[0]}" "${B[0]}" "${C[0]}"; do vastai start instance "$id" 2>&1 | sed "s/^/  start $id: /"; done
    for i in $(seq 1 40); do
      sleep 30; refresh_endpoints >/dev/null
      ok=0; for hf in "${A[1]}" "${C[1]}"; do sshto "$hf" 'true' 2>/dev/null && ok=$((ok+1)); done
      echo "  $(date +%H:%M:%S) reachable: $ok/2 (600 W boxes)"; [ "$ok" = 2 ] && break
    done
    refresh_endpoints;;
  arm)
    echo "== 600 W: chain600w3d (resumes) + chain600w4 waiter =="
    sshto "${A[1]}" 'R=/workspace/results; B=/workspace/bench; source $B/hardkill.sh; kill_all >/dev/null 2>&1
      echo "[$(date +%H:%M:%S)] 600W-3: re-armed after the Vast stop; every step resumes where it left off" >> $R/chain600w.log
      tmux new-session -d -s q600f "bash $B/chain600w3d.sh >> $R/chain600w.log 2>&1"
      tmux new-session -d -s q600g "bash $B/chain600w4.sh >> $R/chain600w.log 2>&1"
      tmux ls | cut -d: -f1 | paste -sd" " | sed "s/^/  sessions: /"'
    echo "== third box: chain_c (resumes) =="
    sshto "${C[1]}" 'R=/workspace/results; B=/workspace/bench; source $B/hardkill.sh; kill_all >/dev/null 2>&1
      echo "[$(date +%H:%M:%S)] CHAINC: re-armed after a stop; every step resumes" >> $R/chain_c.log
      tmux new-session -d -s chainc "bash $B/chain_c.sh >> $R/chain_c.log 2>&1"; tmux ls | cut -d: -f1 | paste -sd" " | sed "s/^/  sessions: /"'
    echo "== 400 W: chain_master2 (resumes) + post2 + post3 waiters (only if it ever restarts) =="
    sshto "${B[1]}" 'R=/workspace/results; B=/workspace/bench; source $B/hardkill.sh; kill_all >/dev/null 2>&1
      echo "[$(date +%H:%M:%S)] MASTER2: re-armed after the Vast stop; every step resumes where it left off" >> $R/chain6000.log
      tmux new-session -d -s master2 "bash $B/chain_master2.sh >> $R/chain6000.log 2>&1"
      tmux new-session -d -s post2 "bash $B/chain_post2.sh >> $R/chain6000.log 2>&1"
      tmux new-session -d -s post3 "bash $B/chain_post3.sh >> $R/chain6000.log 2>&1"
      tmux ls | cut -d: -f1 | paste -sd" " | sed "s/^/  sessions: /"';;
  status)
    bash "$(_p "vast_status.sh")";;
esac
