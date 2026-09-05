#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "CHAIN6000C DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
echo "[$(date +%H:%M:%S)] GLM rerun with no token cap (base and MTP)"
ARMS="long mtp" bash /workspace/bench/glm_eval.sh > $R/glm_eval_long.log 2>&1
echo "[$(date +%H:%M:%S)] GLMLONG DONE"
