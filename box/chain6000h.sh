#!/usr/bin/env bash
R=/workspace/results
echo "[$(date +%H:%M:%S)] chain3b: retry sweep (restarted with the disk guard and the launcher-death check)"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/retry6000.txt >> $R/ksweep6000.log 2>&1
echo "[$(date +%H:%M:%S)] chain3b: quality pass over everything measured on this box"
MODE=eval EVAL_BUDGET=900 bash /workspace/bench/ksweep.sh /workspace/bench/lists/eval6000.txt > $R/keval6000.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000C DONE"
