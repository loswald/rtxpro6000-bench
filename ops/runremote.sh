#!/usr/bin/env bash
set -u
KEY="$HOME/.ssh/id_ed25519"
SP=/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad
WHICH="${1:-inst6000b.ssh}"; SCRIPT="${2:-remote_cleanup.sh}"
read -r _ H P _ < "$SP/$WHICH"
scp -q -i "$KEY" -P "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SP/$SCRIPT" "root@$H:/workspace/bench/_r.sh"
ssh -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30 -o LogLevel=ERROR "root@$H" 'bash /workspace/bench/_r.sh'
