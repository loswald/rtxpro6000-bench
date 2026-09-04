#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 1440); do grep -q "CHAIN6000 DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] chain2: control6000 (DSpark arm)"; bash /workspace/bench/control6000.sh >> $R/control6000.log 2>&1
echo "[$(date +%H:%M:%S)] chain2: fleet2";                    GLM_ARMS=base,mtp bash /workspace/bench/fleet2.sh > $R/fleet2b.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000B DONE"
