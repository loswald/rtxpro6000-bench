#!/usr/bin/env bash
# Kill anything holding GPU memory, self-safely (patterns never match this script's own cmdline).
echo "before:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd" "
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "compute apps holding memory: ${PIDS:-none}"
for p in $PIDS; do kill -9 "$p" 2>/dev/null && echo "  killed $p"; done
sleep 8
echo "after:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd" "
