#!/usr/bin/env bash
R=/workspace/results
echo "[$(date +%H:%M:%S)] speculation: base-vs-base control, then base-vs-MTP"
ARMS=spec bash /workspace/bench/glm_eval.sh > $R/specdiff_glm2.log 2>&1
echo "[$(date +%H:%M:%S)] SPECDIFF2 DONE"
echo "[$(date +%H:%M:%S)] resuming the quality pass"
MODE=eval EVAL_BUDGET=1800 bash /workspace/bench/ksweep.sh /workspace/bench/lists/eval6000.txt >> $R/keval6000.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000C DONE"
