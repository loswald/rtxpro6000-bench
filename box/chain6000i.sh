#!/usr/bin/env bash
R=/workspace/results
echo "[$(date +%H:%M:%S)] GLM-5.3-Flash quality (base, then MTP) - ahead of the queue"
bash /workspace/bench/glm_eval.sh > $R/glm_eval.log 2>&1
echo "[$(date +%H:%M:%S)] resuming: retry sweep"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/retry6000.txt >> $R/ksweep6000.log 2>&1
echo "[$(date +%H:%M:%S)] quality pass over everything measured on this box"
MODE=eval EVAL_BUDGET=1200 bash /workspace/bench/ksweep.sh /workspace/bench/lists/eval6000.txt > $R/keval6000.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000C DONE"
