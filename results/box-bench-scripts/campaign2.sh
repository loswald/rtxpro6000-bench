#!/usr/bin/env bash
# Campaign 2: full shape matrix on the two replica winners + spec-decode A/Bs. Budget ~60 min.
R=/workspace/results; P=$R/probe; S=$R/smoke; B=/workspace/bench; MD=/workspace/models
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =probe 2>/dev/null; tmux kill-session -t =smoke 2>/dev/null
  for pid in $(pgrep -f "vllm serv[e]"); do kill "$pid" 2>/dev/null; done; sleep 10
  for pid in $(pgrep -f "vllm serv[e]"); do kill -9 "$pid" 2>/dev/null; done; sleep 4
}
wait_health(){
  local ports="$1" limit="$2" t=0 ok
  while [ "$t" -lt "$limit" ]; do
    ok=1; for p in $ports; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && return 0
    sleep 10; t=$((t+10))
  done
  return 1
}
serve(){
  local tag="$1" launcher="$2" ports="$3" limit="$4"
  kill_all; log "launch $tag"
  tmux new-session -d -s smoke "bash $launcher > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  if wait_health "$ports" "$limit"; then log "$tag healthy"; return 0; fi
  log "$tag FAILED to become healthy"
  grep -iE "error|not supported|invalid|unrecognized" "$S/$tag.log" 2>/dev/null | grep -vE "import_utils|deep_ep" | tail -3 | cut -c1-200
  return 1
}
log "stage A: gpt-oss-120b x4 replicas, FULL shape matrix"
if serve gptoss_x4_full $B/l_gptoss.sh "8000 8001 8002 8003" 600; then
  bash $B/probe4.sh gptoss_x4_full gptoss $MD/gpt-oss-120b auto full > $P/gptoss_x4_full.log 2>&1
fi
log "stage B: gpt-oss-120b x4 + ngram speculative decoding"
if serve gptoss_x4_ngram $B/l_gptoss_ngram.sh "8000 8001 8002 8003" 600; then
  python3 $B/quality20.py gptoss http://127.0.0.1:8000 $P/gptoss_x4_ngram_quality20.json
  bash $B/probe4.sh gptoss_x4_ngram gptoss $MD/gpt-oss-120b auto quick > $P/gptoss_x4_ngram.log 2>&1
  grep -hE "SpecDecoding metrics" $S/gptoss_p800*.log 2>/dev/null | tail -4 | cut -c1-220 > $P/gptoss_x4_ngram_acceptance.txt
fi
log "stage C: Qwen3.8-27B-FP8 x4 replicas, FULL shape matrix"
if serve qwen27b_x4_full $B/l_qwen27b.sh "8000 8001 8002 8003" 600; then
  bash $B/probe4.sh qwen27b_x4_full qwen27b $MD/Qwen3.8-27B-FP8 auto full > $P/qwen27b_x4_full.log 2>&1
fi
log "stage D: Qwen3.8-27B-FP8 x4 + MTP k=3"
if serve qwen27b_x4_mtp $B/l_qwen27b_mtp.sh "8000 8001 8002 8003" 600; then
  python3 $B/quality20.py qwen27b http://127.0.0.1:8000 $P/qwen27b_x4_mtp_quality20.json
  bash $B/probe4.sh qwen27b_x4_mtp qwen27b $MD/Qwen3.8-27B-FP8 auto quick > $P/qwen27b_x4_mtp.log 2>&1
  grep -hE "SpecDecoding metrics" $S/qwen27b_p800*.log 2>/dev/null | tail -4 | cut -c1-220 > $P/qwen27b_x4_mtp_acceptance.txt
fi
log "CAMPAIGN2 DONE"
kill_all
