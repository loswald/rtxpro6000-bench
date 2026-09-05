#!/usr/bin/env bash
# Original 600 W box to 02:00 UTC 6 Sept (replaces chain600w11). The DP2 x TP2 + EP layout that won the throughput
# sweep produces degenerate output (8 of 20 probe items; 0.643 on 403), so TP2 is broken in this port and the
# fast layout is TP1 x DP4 + EP (1,073 out tok/s, clean probe). Order: its 403-item quality run; its throughput
# levers; 65k tokens of output room on it (or on TP4 if its quality fails); GLM at native FP8 if time; controls.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-12 (${1:-}min left): ${*:2}"; }
for s in q600p q600n q600m q600k; do tmux kill-session -t =$s 2>/dev/null && step "$(left)" "$s stopped; chain600w12 takes over"; done
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
# keep the DP2 x TP2 result under its own name; glm53f_best is retired
for e in json items.jsonl log run.json; do [ -f $R/eval/glm53f_best.$e ] && mv -f $R/eval/glm53f_best.$e $R/eval/glm53f_dp2tp2ep2.$e; done
python3 - <<'PY' 2>/dev/null
import json; p="/workspace/results/eval/glm53f_dp2tp2ep2.json"; d=json.load(open(p)); d["tag"]="glm53f_dp2tp2ep2"; json.dump(d,open(p,"w"))
PY
L=$(left)
step "$L" "GLM-5.3-Flash on TP1 x DP4 + EP: 403-item quality (budget 3600s)"
DP4_BUDGET=3600 ARMS=dp4 bash $B/glm_eval.sh > $R/glm_eval_dp4.log 2>&1
acc=$(python3 -c "import json; d=json.load(open('$R/eval/glm53f_dp4ep4.json')); print(round(d['aggregate']['acc_micro'],3))" 2>/dev/null)
step "$(left)" "DP4 + EP quality: ${acc:-none}"
L=$(left)
if [ "$L" -gt 150 ]; then
  step "$L" "GLM-5.3-Flash on TP1 x DP4 + EP: 256 sequences, 16-bit SSM state at 384, 32k prefill chunk, FP8 KV"
  bash $B/glm_perf4.sh > $R/glm_perf4.log 2>&1
fi
L=$(left)
if [ "$L" -gt 90 ]; then
  b=$(( (L - 130) * 60 )); [ "$b" -lt 3600 ] && b=3600; [ "$b" -gt 5400 ] && b=5400
  if python3 -c "import sys; sys.exit(0 if float('${acc:-0}') >= 0.77 else 1)"; then
    step "$L" "GLM on DP4 + EP with 65k tokens of output room, budget ${b}s"
    BEST_BUDGET=$b LONG_CONC=64 ARMS=dp4long bash $B/glm_eval.sh > $R/glm_eval_dp4long.log 2>&1
  else
    step "$L" "DP4 + EP quality ${acc:-none} is not clean either: 65k tokens of output room on TP4 instead, budget ${b}s"
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
