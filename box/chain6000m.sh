#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "QUANT DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] thinking on vs off, gemma-4-26B BF16"
MODE=eval EVAL_BUDGET=3600 bash /workspace/bench/ksweep.sh /workspace/bench/lists/thinkmode6000.txt > $R/keval_think.log 2>&1
echo "[$(date +%H:%M:%S)] THINKMODE DONE"
