#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "CHAIN6000C DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] GLM rerun, no token cap (base and MTP)"
ARMS="long mtp" bash /workspace/bench/glm_eval.sh > $R/glm_eval_long.log 2>&1
echo "[$(date +%H:%M:%S)] quantisation ladder: fetching the two missing rungs"
source /workspace/bench/dlget.sh
get Qwen/Qwen3.8-27B                        Qwen3.8-27B
get RedHatAI/gemma-4-26B-A4B-it-NVFP4       gemma-4-26B-A4B-NVFP4
echo "[$(date +%H:%M:%S)] quantisation ladder: throughput"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/quant6000.txt > $R/ksweep_quant.log 2>&1
echo "[$(date +%H:%M:%S)] quantisation ladder: quality, uncapped"
MODE=eval EVAL_BUDGET=2400 bash /workspace/bench/ksweep.sh /workspace/bench/lists/quant6000.txt > $R/keval_quant.log 2>&1
echo "[$(date +%H:%M:%S)] QUANT DONE"
