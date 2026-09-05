#!/usr/bin/env bash
# 400 W box, after chain_post2: the reliability axis, which nothing in this campaign had measured. Two hours of
# continuous load on the configuration the node would actually run (Qwen3.8-27B NVFP4, four replicas, W4A4),
# then the same on the official FP8 build. Throughput here is relative (400 W cap); stability is not.
R=/workspace/results; B=/workspace/bench
step(){ echo "[$(date +%H:%M:%S)] POST3: $*"; }
for i in $(seq 1 2880); do grep -q "POST2 DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
step "soak: Qwen3.8-27B NVFP4 b12x, four replicas, 120 min"
TAG=soak_q27_nvfp4 DIR=/workspace/models/Qwen27B-NVFP4-RTX5090 LIN=b12x MINUTES=120 bash $B/soak.sh > $R/soak_q27_nvfp4.log 2>&1
step "soak: Qwen3.8-27B FP8 b12x, four replicas, 120 min"
TAG=soak_q27_fp8 DIR=/workspace/models/Qwen3.8-27B-FP8 LIN=b12x MINUTES=120 bash $B/soak.sh > $R/soak_q27_fp8.log 2>&1
step "POST3 DONE"
