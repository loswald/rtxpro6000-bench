#!/usr/bin/env bash
# Automated campaign for the remaining credit: runs stages back-to-back, each guarded by health checks.
R=/workspace/results; P=$R/probe; S=$R/smoke; B=/workspace/bench; MD=/workspace/models
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =probe 2>/dev/null; tmux kill-session -t =smoke 2>/dev/null
  for pid in $(pgrep -f "vllm serv[e]"); do kill "$pid" 2>/dev/null; done; sleep 12
  for pid in $(pgrep -f "vllm serv[e]"); do kill -9 "$pid" 2>/dev/null; done; sleep 4; }
wait_health(){ local ports="$1" limit="$2" t=0 ok
  while [ "$t" -lt "$limit" ]; do ok=1; for p in $ports; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && return 0; if grep -qE "EXIT=" "$3" 2>/dev/null; then return 1; fi; sleep 10; t=$((t+10)); done; return 1; }
serve(){ # tag launcher ports limit -> 0 if healthy
  local tag="$1" launcher="$2" ports="$3" limit="$4"
  kill_all; log "launch $tag"
  tmux new-session -d -s smoke "bash $launcher > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  if wait_health "$ports" "$limit" "$S/$tag.log"; then log "$tag healthy"; return 0; fi
  log "$tag FAILED to become healthy"; grep -E "Error|error" "$S/$tag.log" | grep -vE "import_utils|deep_ep" | tail -n 3 | cut -c1-200; return 1; }
log "stage 0: wait for ds4flash_tp4_b12xmoe probe (<= 40 min)"
for i in $(seq 1 240); do grep -q "PROBE DONE" $P/ds4flash_tp4_b12xmoe.log 2>/dev/null && break; sleep 10; done
log "stage 1: DP4+EP4 (marlin) quick"
if serve ds4flash_dp4ep4 $B/launch_ds4flash_dp4ep4.sh 8000 720; then
  python3 $B/quality20.py ds4flash http://127.0.0.1:8000 $P/ds4flash_dp4ep4_quality20.json
  bash $B/probe_v2.sh ds4flash_dp4ep4 ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 quick > $P/ds4flash_dp4ep4.log 2>&1; fi
log "stage 2: TP4 b12x + DSpark spec decode"
if serve ds4flash_tp4_b12xmoe_dspark $B/launch_ds4flash_tp4_b12xmoe_dspark.sh 8000 720; then
  python3 $B/quality20.py ds4flash http://127.0.0.1:8000 $P/ds4flash_tp4_b12xmoe_dspark_quality20.json
  bash $B/probe_v2.sh ds4flash_tp4_b12xmoe_dspark ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 spec > $P/ds4flash_tp4_b12xmoe_dspark.log 2>&1
  grep -E "SpecDecoding metrics" $S/ds4flash_tp4_b12xmoe_dspark.log | tail -n 3 | cut -c1-220 > $P/ds4flash_tp4_b12xmoe_dspark_acceptance.txt; fi
log "stage 3: Qwen3.8-27B-FP8 x4 replicas"
printf "#!/usr/bin/env bash\nexec bash $B/launch_x4.sh $MD/Qwen3.8-27B-FP8 qwen27b --kv-cache-dtype fp8 --kernel-config.linear_backend b12x\n" > $B/launch_qwen27b_x4.sh
if serve qwen27b_x4 $B/launch_qwen27b_x4.sh "8000 8001 8002 8003" 900; then
  python3 $B/quality20.py qwen27b http://127.0.0.1:8000 $P/qwen27b_x4_quality20.json
  bash $B/probe4.sh qwen27b_x4 qwen27b $MD/Qwen3.8-27B-FP8 auto quick > $P/qwen27b_x4.log 2>&1
else
  printf "#!/usr/bin/env bash\nexec bash $B/launch_x4.sh $MD/Qwen3.8-27B-FP8 qwen27b --kernel-config.linear_backend b12x\n" > $B/launch_qwen27b_x4b.sh
  if serve qwen27b_x4 $B/launch_qwen27b_x4b.sh "8000 8001 8002 8003" 900; then
    python3 $B/quality20.py qwen27b http://127.0.0.1:8000 $P/qwen27b_x4_quality20.json
    bash $B/probe4.sh qwen27b_x4 qwen27b $MD/Qwen3.8-27B-FP8 auto quick > $P/qwen27b_x4.log 2>&1; fi; fi
log "stage 4: gpt-oss-120b x4 replicas"
printf "#!/usr/bin/env bash\nexec bash $B/launch_x4.sh $MD/gpt-oss-120b gptoss\n" > $B/launch_gptoss_x4.sh
if serve gptoss_x4 $B/launch_gptoss_x4.sh "8000 8001 8002 8003" 900; then
  grep -m1 -E "Mxfp4 MoE backend" $S/gptoss_x4.log | cut -c1-160
  python3 $B/quality20.py gptoss http://127.0.0.1:8000 $P/gptoss_x4_quality20.json
  bash $B/probe4.sh gptoss_x4 gptoss $MD/gpt-oss-120b auto quick > $P/gptoss_x4.log 2>&1; fi
log "stage 5: TP4+EP4 marlin clean quick (unique seeds)"
if serve ds4flash_tp4ep4_marlin_v2 $B/launch_ds4flash_tp4ep4.sh 8000 720; then
  bash $B/probe_v2.sh ds4flash_tp4ep4_marlin_v2 ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 quick > $P/ds4flash_tp4ep4_marlin_v2.log 2>&1; fi
log "CAMPAIGN DONE"
