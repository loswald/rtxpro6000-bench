#!/usr/bin/env bash
# First contact with a fresh box: directories, the whole bench tree, the eval suite, then provisioning in tmux.
set -u
KEY="$HOME/.ssh/id_ed25519"
SP=/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad
A=/mnt/c/Users/ushni/Downloads/AIRR
read -r _ H P _ < "$SP/${1:-inst6000c.ssh}"
O=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20)
ssh -q "${O[@]}" -p "$P" "root@$H" 'mkdir -p /workspace/bench/lists /workspace/results/smoke /workspace/results/probe /workspace/results/eval /workspace/models && echo "  dirs ready"'
scp -q "${O[@]}" -P "$P" "$A"/box/*.sh "$A"/box/*.py "root@$H:/workspace/bench/" && echo "  bench scripts shipped"
scp -q "${O[@]}" -P "$P" "$A"/box/lists/* "root@$H:/workspace/bench/lists/" && echo "  lists shipped"
[ -f "$SP/evalsuite.tgz" ] && scp -q "${O[@]}" -P "$P" "$SP/evalsuite.tgz" "root@$H:/workspace/bench/" && echo "  evalsuite.tgz shipped"
ssh -q "${O[@]}" -p "$P" "root@$H" 'cd /workspace/bench && [ -f evalsuite.tgz ] && tar xzf evalsuite.tgz && ls evalsuite | head -3 | paste -sd" " | sed "s/^/  evalsuite: /"; chmod +x /workspace/bench/*.sh; tmux new-session -d -s prov "bash /workspace/bench/provision2.sh > /workspace/results/provision2.log 2>&1"; sleep 2; tmux ls | cut -d: -f1 | paste -sd" " | sed "s/^/  sessions: /"'
