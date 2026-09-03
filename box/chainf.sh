#!/usr/bin/env bash
for i in $(seq 1 200); do grep -q "SGL-DS4 DONE" /workspace/results/sgl_ds4.log 2>/dev/null && break; sleep 15; done
echo "[$(date +%H:%M:%S)] deepseek comparison done"
for i in $(seq 1 240); do grep -q "FLEET-DL-DONE" /workspace/results/dl_fleet.log 2>/dev/null && break; sleep 15; done
echo "[$(date +%H:%M:%S)] downloads done -> fleet campaign"
bash /workspace/bench/fleet.sh > /workspace/results/fleet.log 2>&1
echo "[$(date +%H:%M:%S)] FLEET CHAINED-DONE"
