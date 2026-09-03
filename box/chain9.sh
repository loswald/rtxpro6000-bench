#!/usr/bin/env bash
for i in $(seq 1 400); do grep -q "QUALITY GATE DONE" /workspace/results/quality_gate.log 2>/dev/null && break; sleep 10; done
echo "[$(date +%H:%M:%S)] quality gate done -> per-model kernel sweep (replica models)"
bash /workspace/bench/persweep.sh > /workspace/results/persweep.log 2>&1
echo "[$(date +%H:%M:%S)] -> per-model kernel sweep (TP4 models)"
bash /workspace/bench/persweep_tp4.sh > /workspace/results/persweep_tp4.log 2>&1
echo "[$(date +%H:%M:%S)] -> full ladder on each model best config"
bash /workspace/bench/full2.sh > /workspace/results/full2.log 2>&1
echo "[$(date +%H:%M:%S)] ALL DONE"
