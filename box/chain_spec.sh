#!/usr/bin/env bash
R=/workspace/results
echo "[$(date +%H:%M:%S)] GLM speculation: is the MTP head lossless? greedy, bit-for-bit"
ARMS=spec bash /workspace/bench/glm_eval.sh > $R/specdiff_glm.log 2>&1
echo "[$(date +%H:%M:%S)] SPECDIFF DONE"
