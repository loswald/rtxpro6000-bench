#!/usr/bin/env bash
# Third box: Qwen3.8-Flash-Next with the FP8 PLE fix, taking the MiniMax slot from chain_c4. Prints CHAINC3 DONE.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-05 22:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC5 (${1:-}min left): ${*:2}"; }
tmux kill-session -t =chainc4 2>/dev/null && step "$(left)" "chain_c4 stopped (MiniMax); Qwen3.8-Flash-Next with the FP8 PLE fix takes the slot"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
python3 $B/patch_ple.py 2>&1 | sed 's/^/  /'
export EXTRA_ENV="VLLM_QWEN4EXP_PLE_FP8=1"
step "$(left)" "Qwen3.8-Flash-Next throughput: 2 x TP2, two shapes"
SHAPES=fast bash $B/ksweep.sh $B/lists/qwen38fn_c3.txt > $R/ksweep_qfn2.log 2>&1
best=$(python3 $B/pick_best.py "$P/qwen38fn_*" router 1024 2>$R/qfn_best.tps)
L=$(left)
if [ -n "$best" ] && [ "$L" -gt 30 ]; then
  step "$L" "Qwen3.8-Flash-Next quality: $best ($(cat $R/qfn_best.tps 2>/dev/null) out tok/s at router C1024), budget $(( (L - 12) * 60 ))s"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$(( (L - 12) * 60 )) bash $B/ksweep.sh $B/lists/qwen38fn_c3.txt > $R/keval_qfn.log 2>&1
else
  step "$L" "Qwen3.8-Flash-Next: did not serve, or no time for its quality run (best: ${best:-none})"
fi
unset EXTRA_ENV
L=$(left)
if [ "$L" -gt 45 ]; then
  step "$L" "MiniMax-M3 quality (budget $(( (L - 20) * 60 ))s), then two throughput shapes"
  MODE=eval FIRST_ONLY=1 EVAL_BUDGET=$(( (L - 20) * 60 )) bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/keval_minimax.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/ksweep_minimax.log 2>&1
else
  step "$L" "MiniMax-M3 not run (time)"
fi
step "$(left)" "CHAINC3 DONE"
kill_all
