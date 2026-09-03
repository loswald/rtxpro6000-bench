#!/usr/bin/env bash
# Kill every stale campaign session, then wait until the GPUs AND the ports are genuinely
# free. A previous run lost two workloads because an orphaned chained session ran its own
# kill_all against my servers mid-benchmark, and because relaunching before TIME_WAIT
# cleared gave "Address already in use".
set -u
echo "== killing every campaign session =="
for s in chainf chaing dlf dlnv qt nv srv glm gimg sgls; do
  tmux kill-session -t "=$s" 2>/dev/null && echo "  killed $s"
done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill -9 "$pid" 2>/dev/null
done
sleep 5
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill -9 "$pid" 2>/dev/null
done

echo "== waiting for VRAM to drain =="
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${used:-1}" -lt 2000 ] && { echo "  VRAM free after $((i*5))s"; break; }
  sleep 5
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'

echo "== waiting for ports 8000-8003 to release =="
for i in $(seq 1 40); do
  busy=0
  for p in 8000 8001 8002 8003; do
    ss -lnt 2>/dev/null | grep -q ":$p " && busy=1
  done
  [ "$busy" = 0 ] && { echo "  ports free after $((i*3))s"; break; }
  sleep 3
done
ss -lnt 2>/dev/null | grep -E ":800[0-3] " | sed 's/^/    STILL BOUND: /'
echo "== clean =="
tmux ls 2>/dev/null | sed 's/^/  session left: /'