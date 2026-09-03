#!/usr/bin/env bash
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -9 "$pid" 2>/dev/null; done
  sleep 8; }
log "STAGE 1: MoE backend A/B on gpt-oss-120b x4 (nightly; every backend)"
bash $B/moe_ab.sh > $R/moe_ab.log 2>&1
log "STAGE 1 done"
log "STAGE 2: DeepSeek TP4 no-EP + b12x MoE (like-for-like vs host A)"
kill_all
tmux new-session -d -s srv "bash $B/ds4_b12x.sh > $S/n_ds4_tp4_b12x.log 2>&1; echo EXIT=\$? >> $S/n_ds4_tp4_b12x.log"
t=0; ok=0
while [ $t -lt 900 ]; do
  curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if grep -q "^EXIT=" $S/n_ds4_tp4_b12x.log 2>/dev/null; then
    log "ds4 b12x DIED after ${t}s"
    grep -iE "does not support|invalid|Error|error" $S/n_ds4_tp4_b12x.log | grep -vE "import_utils|deep_ep" | head -3 | cut -c1-220
    break
  fi
  sleep 10; t=$((t+10))
done
if [ "$ok" = 1 ]; then
  log "ds4 b12x healthy in ${t}s; kernel: $(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend" $S/n_ds4_tp4_b12x.log)"
  python3 $B/quality20.py ds4flash http://127.0.0.1:8000 $P/n_ds4_tp4_b12x_quality20.json 2>&1 | tail -1
  bash $B/probe_v2.sh n_ds4_tp4_b12x ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 full > $P/n_ds4_tp4_b12x.log 2>&1
fi
kill_all
log "RUN-ALL COMPLETE"
