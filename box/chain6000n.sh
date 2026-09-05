#!/usr/bin/env bash
R=/workspace/results
echo "[$(date +%H:%M:%S)] GLM-5.3-Flash quality, corrected: parser glm45, Z.AI sampling, no token cap"
ARMS="base mtp" bash /workspace/bench/glm_eval.sh > $R/glm_eval_fixed.log 2>&1
echo "[$(date +%H:%M:%S)] resuming the quality pass (resumable: scored configs are skipped)"
MODE=eval EVAL_BUDGET=1800 bash /workspace/bench/ksweep.sh /workspace/bench/lists/eval6000.txt >> $R/keval6000.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000C DONE"
