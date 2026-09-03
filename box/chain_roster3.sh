#!/usr/bin/env bash
# wait for the fleet to finish (or its session to die), then run the roster rotation
while ! grep -q "FLEET2 DONE" /workspace/results/fleet2.log 2>/dev/null; do
  tmux has-session -t chainfleet 2>/dev/null || { sleep 60; grep -q "FLEET2 DONE" /workspace/results/fleet2.log 2>/dev/null || echo "[$(date +%H:%M:%S)] chainfleet gone without FLEET2 DONE; starting roster3 anyway"; break; }
  sleep 120
done
bash /workspace/bench/roster3.sh > /workspace/results/roster3.log 2>&1
