#!/usr/bin/env bash
for i in $(seq 1 300); do grep -q "PRIORITY DONE" /workspace/results/priority.log 2>/dev/null && break; sleep 10; done
echo "[$(date +%H:%M:%S)] priority done -> full campaign"
bash /workspace/bench/full.sh > /workspace/results/full.log 2>&1
echo "[$(date +%H:%M:%S)] FULL CHAINED-DONE"
