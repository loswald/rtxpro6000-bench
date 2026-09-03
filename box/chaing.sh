#!/usr/bin/env bash
for i in $(seq 1 400); do grep -q "FLEET DONE" /workspace/results/fleet.log 2>/dev/null && break; sleep 15; done
echo "[$(date +%H:%M:%S)] fleet done -> SGLang retry with warmed JIT cache"
bash /workspace/bench/sgl_retry.sh > /workspace/results/sgl_retry.log 2>&1
echo "[$(date +%H:%M:%S)] SGL RETRY CHAINED-DONE"
