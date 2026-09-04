#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "CHAIN6000C DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] chainKL: logit-level quality (control, NVFP4 vs FP8, KV dtype, kernel equivalence, speculation)"
POS=16 bash /workspace/bench/kldiff.sh > $R/kldiff.log 2>&1
echo "[$(date +%H:%M:%S)] KLDIFF6000 DONE"
