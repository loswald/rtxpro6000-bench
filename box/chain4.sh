#!/usr/bin/env bash
for i in $(seq 1 360); do grep -q "SATURATION DONE" /workspace/results/sat.log 2>/dev/null && break; sleep 10; done
echo "[$(date +%H:%M:%S)] saturation done -> round 3"
bash /workspace/bench/round3.sh > /workspace/results/round3.log 2>&1
echo "[$(date +%H:%M:%S)] ROUND3 CHAINED-DONE"
