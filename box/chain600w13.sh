#!/usr/bin/env bash
# Original 600 W box to 02:00 UTC 6 Sept. Runs after chain600w12's lever sweep (GLMPERF4 DONE): the DP4 + EP
# 403-item quality run that chain600w12 lost to a port collision, then 65k tokens of output room on DP4 + EP if
# its quality is clean (else on TP4), then GLM at native FP8 if time, then the control arms. Prints CHAIN600W7 DONE.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-13 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 720); do grep -q "GLMPERF4 DONE" $R/glm_perf4.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =q600q 2>/dev/null && step "$(left)" "chain600w12 stopped after its lever sweep; chain600w13 runs the quality work"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
# a server from another session can still hold :8000 without a GPU allocation yet; wait for the port
for i in $(seq 1 40); do ss -lnt 2>/dev/null | grep -q ":8000 " || break; sleep 3; done
step "$(left)" "GLM-5.3-Flash on TP1 x DP4 + EP: 403-item quality (budget 3600s)"
DP4_BUDGET=3600 ARMS=dp4 bash $B/glm_eval.sh > $R/glm_eval_dp4.log 2>&1
acc=$(python3 -c "import json; d=json.load(open('$R/eval/glm53f_dp4ep4.json')); print(round(d['aggregate']['acc_micro'],3))" 2>/dev/null)
step "$(left)" "DP4 + EP quality: ${acc:-none}"
L=$(left)
if [ "$L" -gt 80 ]; then
  b=$(( (L - 120) * 60 )); [ "$b" -lt 3600 ] && b=3600; [ "$b" -gt 5400 ] && b=5400
  if python3 -c "import sys; sys.exit(0 if float('${acc:-0}') >= 0.77 else 1)"; then
    step "$L" "GLM on DP4 + EP with 65k tokens of output room, budget ${b}s"
    BEST_BUDGET=$b LONG_CONC=64 ARMS=dp4long bash $B/glm_eval.sh > $R/glm_eval_dp4long.log 2>&1
  else
    step "$L" "DP4 + EP quality ${acc:-none} not clean: 65k tokens of output room on TP4 instead, budget ${b}s"
    BEST_FLAGS="--max-num-seqs 512 --max-num-batched-tokens 16384" BEST_BUDGET=$b LONG_CONC=64 ARMS=bestlong bash $B/glm_eval.sh > $R/glm_eval_bestlong.log 2>&1
  fi
fi
L=$(left)
if [ "$L" -gt 110 ]; then
  step "$L" "GLM-5.3-Flash at native FP8, 330 GB, TP4, 32 sequences: quality (budget $(( (L - 45) * 60 ))s)"
  FP8_BUDGET=$(( (L - 45) * 60 )) ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
fi
L=$(left)
if [ "$L" -gt 40 ]; then
  step "$L" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
  MODE=eval EVAL_BUDGET=$(( (L - 10) * 60 / 2 )) bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
