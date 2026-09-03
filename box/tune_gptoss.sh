#!/usr/bin/env bash
# Knob-tuning sweep on the current champion: gpt-oss-120b as 4 TP1 replicas.
# Each variant serves, runs a 3-point probe (router C256, promptopt C256, judge C64), then tears down.
R=/workspace/results; P=$R/probe; S=$R/smoke; B=/workspace/bench; MD=/workspace/models
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =probe 2>/dev/null; tmux kill-session -t =smoke 2>/dev/null
  for pid in $(pgrep -f "vllm serv[e]"); do kill "$pid" 2>/dev/null; done; sleep 10
  for pid in $(pgrep -f "vllm serv[e]"); do kill -9 "$pid" 2>/dev/null; done; sleep 4; }
wait_health(){ local t=0; while [ "$t" -lt 600 ]; do local ok=1
    for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && return 0; sleep 10; t=$((t+10)); done; return 1; }
variant(){ # tag, then extra vllm args
  local tag="$1"; shift
  kill_all; log "variant $tag :: $*"
  printf '#!/usr/bin/env bash\nexec bash %s/launch_x4.sh %s/gpt-oss-120b gptoss %s\n' "$B" "$MD" "$*" > $B/l_v_$tag.sh
  chmod +x $B/l_v_$tag.sh
  tmux new-session -d -s smoke "bash $B/l_v_$tag.sh > $S/tune_$tag.log 2>&1; echo EXIT=\$? >> $S/tune_$tag.log"
  if wait_health; then
    log "$tag healthy"
    bash $B/probe4.sh tune_$tag gptoss $MD/gpt-oss-120b auto tune > $P/tune_$tag.log 2>&1
    grep -m1 -E "Mxfp4 MoE backend|GPU KV cache size" $S/tune_$tag.log | sed 's/.*\] //' | cut -c1-120
  else
    log "$tag FAILED"; grep -iE "error|invalid|unrecognized|not supported" $S/tune_$tag.log | grep -vE "import_utils|deep_ep" | tail -2 | cut -c1-180
  fi
}
# baseline is the config already measured (mnbt 8192, seqs 256, util 0.92, marlin auto)
variant mnbt16k  --max-num-batched-tokens 16384
variant mnbt4k   --max-num-batched-tokens 4096
variant seqs512  --max-num-seqs 512 --max-num-batched-tokens 16384
variant ficutlass --moe-backend flashinfer_cutlass
variant b12xmoe  --moe-backend b12x
variant asyncsched --async-scheduling
variant util96   --gpu-memory-utilization 0.96 --max-num-batched-tokens 16384
log "TUNE DONE"
kill_all
