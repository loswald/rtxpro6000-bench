#!/usr/bin/env bash
for i in $(seq 1 200); do grep -q "CAMPAIGN4 DONE" /workspace/results/campaign4.log 2>/dev/null && break; sleep 15; done
echo "[$(date +%H:%M:%S)] campaign4 finished; starting MoE A/B"
bash /workspace/bench/moe_ab.sh > /workspace/results/moe_ab.log 2>&1
echo "[$(date +%H:%M:%S)] MOE-AB COMPLETE"
