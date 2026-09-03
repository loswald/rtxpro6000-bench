#!/usr/bin/env bash
for i in $(seq 1 400); do grep -q "ROUND3 DONE" /workspace/results/round3.log 2>/dev/null && break; sleep 10; done
echo "[$(date +%H:%M:%S)] round3 done -> round4 (plateau hunt + KV quality gate)"
bash /workspace/bench/round4.sh > /workspace/results/round4.log 2>&1
echo "[$(date +%H:%M:%S)] ROUND4 CHAINED-DONE"
