#!/usr/bin/env bash
# Copy the user's key file to the third box without opening it, lock it down, start the OpenRouter comparison.
set -u
KEY="$HOME/.ssh/id_ed25519"; SP=/mnt/c/Users/ushni/AppData/Local/Temp/claude/C--Users-ushni-Downloads-AIRR/ba0185bd-2c4e-4173-bafa-b54fc63ae431/scratchpad
read -r _ H P _ < "$SP/inst6000c.ssh"
SRC=/mnt/c/Users/ushni/Downloads/rtxpro6000-bench/openrouter.txt
[ -f "$SRC" ] || { echo "source file missing"; exit 1; }
scp -q -i "$KEY" -P "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SRC" "root@$H:/workspace/.openrouter_key" || { echo "scp failed"; exit 1; }
ssh -q -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "root@$H" '
chmod 600 /workspace/.openrouter_key; echo "remote key file: $(wc -c < /workspace/.openrouter_key) bytes, mode $(stat -c %a /workspace/.openrouter_key)"
tmux has-session -t =oreval 2>/dev/null && echo "oreval already running" || tmux new-session -d -s oreval "bash /workspace/bench/or_eval.sh > /workspace/results/or_eval.log 2>&1"
sleep 20; tmux ls | cut -d: -f1 | paste -sd" "; cat /workspace/results/or_eval.log | cut -c1-160; tail -2 /workspace/results/eval_or/or_glm.log 2>/dev/null | cut -c1-200'
