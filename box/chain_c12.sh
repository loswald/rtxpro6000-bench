#!/usr/bin/env bash
# Third box to 02:00 UTC 6 Sept, one chain (replaces chain_c10 and chain_c11 after they collided): DeepSeek DP4 + EP at
# 2,048 streams; resume DeepSeek's 65k-room quality run where chain_c8 left it; Flash-Next with 65k of room; a MiniMax
# retry without the block-size flag the Marlin kernel refused. CHAINC3 DONE at the end.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC12 (${1:-}min left): ${*:2}"; }
for s in chainc10 chainc11 chainc8; do tmux kill-session -t =$s 2>/dev/null && step "$(left)" "$s stopped"; done
source $B/hardkill.sh; kill_all >/dev/null 2>&1
for i in $(seq 1 40); do ss -lnt 2>/dev/null | grep -q ":8000 " || break; sleep 3; done
export EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1"
L=$(left)
if [ "$L" -gt 120 ]; then
  step "$L" "DeepSeek-V4-Flash DP4 + EP at 2,048 streams: router and shared-prefix shapes"
  SHAPES=deep bash $B/ksweep.sh $B/lists/ds_deep.txt > $R/ksweep_dsdeep.log 2>&1
fi
L=$(left)
if [ "$L" -gt 60 ]; then
  step "$L" "DeepSeek-V4-Flash, fastest layout, 65k tokens of output room: resuming the quality run (budget $(( (L - 60) * 60 ))s)"
  EVAL_RESUME=1 MODE=eval FIRST_ONLY=1 EVAL_CONC=48 EVAL_MAXTOK=65536 \
    EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
    EVAL_BUDGET=$(( (L - 60) * 60 )) bash $B/ksweep.sh $B/lists/ds_long.txt > $R/keval_dslong.log 2>&1
fi
export EXTRA_ENV="VLLM_QWEN4EXP_PLE_FP8=1"
L=$(left)
if [ "$L" -gt 45 ]; then
  b=$(( (L - 40) * 60 )); [ "$b" -gt 2700 ] && b=2700
  step "$L" "Qwen3.8-Flash-Next with 65k tokens of output room: quality (budget ${b}s)"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_MAXTOK=65536 EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
    EVAL_BUDGET=$b bash $B/ksweep.sh $B/lists/qwen38fn_long.txt > $R/keval_qfnlong.log 2>&1
fi
unset EXTRA_ENV
L=$(left)
if [ "$L" -gt 25 ]; then
  step "$L" "MiniMax-M3 retry: Marlin MoE without the block-size flag, then block 256; quality with the remaining budget"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$(( (L - 15) * 60 )) bash $B/ksweep.sh $B/lists/minimax_c5.txt > $R/keval_minimax2.log 2>&1
fi
step "$(left)" "CHAINC3 DONE"
kill_all
