#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 4320); do grep -q "CHAIN6000D DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
uv pip install --system pydivsufsort > /dev/null 2>&1
echo "[$(date +%H:%M:%S)] chain6: Motif-3 on the vendor fork (re-run: VRAM was starved, and pydivsufsort was missing)"
MD=/workspace/models/Motif-3-NVFP4 bash /workspace/bench/motif_vllm.sh > $R/motif2.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000E DONE"
