#!/usr/bin/env bash
# Third box, extended window to 02:00 UTC 6 Sept (replaces chain_c7: the auto MoE backend cannot serve this model).
# Order: Qwen3.8-Flash-Next as one TP4 engine (then its W4A4 kernel, then quality) - MiniMax-M3 quality and
# throughput - DeepSeek-V4-Flash on its fastest layout with 65k tokens of output room - MiniMax with CUDA graphs.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC8 (${1:-}min left): ${*:2}"; }
for s in chainc7 chainc6 chainc5 chainc4; do tmux kill-session -t =$s 2>/dev/null && step "$(left)" "$s stopped; chain_c8 runs the extended window"; done
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
L=$(left)
if [ "$L" -gt 150 ]; then
  export EXTRA_ENV="VLLM_QWEN4EXP_PLE_FP8=1"
  step "$L" "Qwen3.8-Flash-Next: Marlin MoE: two TP2 replicas, one TP4 engine, TP4 with W4A4 linears"
  SHAPES=fast bash $B/ksweep.sh $B/lists/qwen38fn_c6.txt > $R/ksweep_qfn4.log 2>&1
  best=$(python3 $B/pick_best.py "$P/qwen38fn_*" router 1024 2>$R/qfn_best.tps)
  L=$(left)
  if [ -n "$best" ] && [ "$L" -gt 120 ]; then
    line=$(grep -vE "^#|^$" $B/lists/qwen38fn_c6.txt | while IFS="|" read -r t rest; do case "$best" in "$t"_*) echo "$t|$rest";; esac; done | head -1)
    printf '%s\n' "$line" > $B/lists/qfn_best.txt
    step "$L" "Qwen3.8-Flash-Next quality: $best ($(cat $R/qfn_best.tps 2>/dev/null) out tok/s at router C1024), budget 2700s"
    MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=2700 bash $B/ksweep.sh $B/lists/qfn_best.txt > $R/keval_qfn.log 2>&1
  else
    step "$L" "Qwen3.8-Flash-Next: no layout served (best: ${best:-none})"
  fi
  unset EXTRA_ENV
fi
L=$(left)
if [ "$L" -gt 90 ]; then
  b=$(( (L - 130) * 60 )); [ "$b" -lt 3000 ] && b=3000; [ "$b" -gt 6000 ] && b=6000
  step "$L" "MiniMax-M3 quality (budget ${b}s), then two throughput shapes"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$b bash $B/ksweep.sh $B/lists/minimax_c3.txt > $R/keval_minimax.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c3.txt > $R/ksweep_minimax.log 2>&1
fi
L=$(left)
if [ "$L" -gt 70 ]; then
  step "$L" "DeepSeek-V4-Flash, fastest layout, 65k tokens of output room: quality (budget $(( (L - 35) * 60 ))s)"
  EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1" MODE=eval FIRST_ONLY=1 EVAL_CONC=48 EVAL_MAXTOK=65536 \
    EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
    EVAL_BUDGET=$(( (L - 35) * 60 )) bash $B/ksweep.sh $B/lists/ds_long.txt > $R/keval_dslong.log 2>&1
fi
L=$(left)
if [ "$L" -gt 35 ] && ls -d $P/minimaxm3_* >/dev/null 2>&1; then
  step "$L" "MiniMax-M3 with CUDA graphs on"
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c4.txt > $R/ksweep_minimax2.log 2>&1
fi
step "$(left)" "CHAINC3 DONE"
kill_all
