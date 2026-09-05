#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "MASTER DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] logit pass, second run: the BF16 ladder and speculation"
POS=16 bash /workspace/bench/kldiff.sh >> $R/kldiff.log 2>&1
echo "[$(date +%H:%M:%S)] KLDIFF2 DONE"
