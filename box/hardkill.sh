#!/usr/bin/env bash
# Shared teardown, sourced by every campaign script. A previous run lost two workloads
# because an orphaned session ran its own kill_all against my servers mid-benchmark, and
# because relaunching before the sockets cleared gave "Address already in use".
kill_all(){
  tmux kill-session -t =srv 2>/dev/null
  tmux kill-session -t =glmsrv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 5
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  # VRAM is released asynchronously; launching before it drains causes a false OOM.
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [ "${used:-1}" -lt 2000 ] && break
    sleep 5
  done
  # Sockets sit in TIME_WAIT after shutdown, which gives "Address already in use".
  for i in $(seq 1 40); do
    busy=0
    for p in 8000 8001 8002 8003; do ss -lnt 2>/dev/null | grep -q ":$p " && busy=1; done
    [ "$busy" = 0 ] && break
    sleep 3
  done
}