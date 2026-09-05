#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "MASTER DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] Laguna-S-2.1: thinking on (with room to finish) vs off"
MODE=eval EVAL_BUDGET=2400 bash /workspace/bench/ksweep.sh /workspace/bench/lists/laguna6000.txt > $R/keval_laguna.log 2>&1
echo "[$(date +%H:%M:%S)] LAGUNA DONE"
