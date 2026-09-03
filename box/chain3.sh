#!/usr/bin/env bash
for i in $(seq 1 400); do grep -q "RUN-ALL COMPLETE" /workspace/results/run_all.log 2>/dev/null && break; sleep 10; done
echo "[$(date +%H:%M:%S)] starting round 2"
bash /workspace/bench/round2.sh > /workspace/results/round2.log 2>&1
echo "[$(date +%H:%M:%S)] ROUND2 CHAINED-DONE"
