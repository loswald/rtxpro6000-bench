#!/usr/bin/env bash
# Original 600 W box: recovery after the tmux server died at ~23:33 UTC with the native-FP8 quality run at 260 of 403.
# Resume that run under the same tag and caps, then the FP8 throughput shapes, then the TP2 probes. CHAIN600W7 DONE.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-17 (${1:-}min left): ${*:2}"; }
source $B/hardkill.sh; kill_all >/dev/null 2>&1
L=$(left)
if [ "$L" -gt 45 ]; then
  b=$(( (L - 70) * 60 )); [ "$b" -lt 1800 ] && b=1800; [ "$b" -gt 3600 ] && b=3600
  step "$L" "GLM-5.3-Flash at native FP8: resuming the 403-item run from item 261 (budget ${b}s)"
  FORCE=1 EVAL_ARGS="--resume" FP8_BUDGET=$b ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8_resume.log 2>&1
  python3 -c "import json; d=json.load(open('$R/eval/glm53f_fp8.json')); a=d['aggregate']; print('  native FP8 quality:', a.get('n_scored'), 'items, acc', round(a['acc_micro'],3), 'partial', d.get('partial'), 'trunc', a.get('trunc_rate'))" 2>/dev/null
fi
L=$(left)
if [ "$L" -gt 35 ]; then
  step "$L" "GLM-5.3-Flash at native FP8, TP4: throughput shapes at 128 sequences"
  bash $B/glm_fp8_shapes.sh > $R/glm_fp8_shapes.log 2>&1
fi
L=$(left)
if [ "$L" -gt 25 ]; then
  step "$L" "GLM at TP2: eager, Marlin MoE, both, and TP2 without EP - probe each"
  bash $B/glm_perf5.sh > $R/glm_perf5.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
