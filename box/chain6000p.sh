#!/usr/bin/env bash
R=/workspace/results
bash /workspace/bench/chain_spec.sh
echo "[$(date +%H:%M:%S)] resuming the quality pass"
MODE=eval EVAL_BUDGET=1800 bash /workspace/bench/ksweep.sh /workspace/bench/lists/eval6000.txt >> $R/keval6000.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000C DONE"
