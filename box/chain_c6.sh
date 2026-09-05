#!/usr/bin/env bash
# Third box, extended window to 02:00 UTC 6 Sept. Takes over from chain_c5 once its Qwen3.8-Flash-Next stage is
# done: Flash-Next's W4A4 kernel arm (with quality), MiniMax-M3 quality and throughput, DeepSeek-V4-Flash on its
# fastest layout with 65k tokens of output room, then the remaining Flash-Next layouts and MiniMax with CUDA
# graphs. Prints CHAINC3 DONE at the end so the finish-watcher needs no change.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC6 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 720); do
  grep -E "CHAINC5.*(MiniMax-M3|CHAINC3 DONE|did not serve)" $R/chain_c.log 2>/dev/null | grep -q . && break
  sleep 20
done
tmux kill-session -t =chainc5 2>/dev/null && step "$(left)" "chain_c5 stopped after its Flash-Next stage; continuing as chain_c6 to 02:00 UTC"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
served=0; [ -d "$P/qwen38fn_tp2x2_---" ] && served=1
L=$(left)
if [ "$served" = 1 ] && [ "$L" -gt 200 ]; then
  export EXTRA_ENV="VLLM_QWEN4EXP_PLE_FP8=1"
  step "$L" "Qwen3.8-Flash-Next under the W4A4 linear kernel: two shapes, then quality (budget 2400s)"
  SHAPES=fast bash $B/ksweep.sh <(grep -E "^qwen38fn_tp2x2b12x" $B/lists/qwen38fn_c4.txt) > $R/ksweep_qfn_b12x.log 2>&1
  [ -d "$P/qwen38fn_tp2x2b12x_b12x--" ] && MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=2400 bash $B/ksweep.sh <(grep -E "^qwen38fn_tp2x2b12x" $B/lists/qwen38fn_c4.txt) > $R/keval_qfn_b12x.log 2>&1
  unset EXTRA_ENV
fi
L=$(left)
if [ "$L" -gt 90 ]; then
  step "$L" "MiniMax-M3 quality (budget $(( (L - 150) > 40 ? (L - 150) * 60 : 2400 ))s), then two throughput shapes"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$(( (L - 150) > 40 ? (L - 150) * 60 : 2400 )) bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/keval_minimax.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/ksweep_minimax.log 2>&1
fi
L=$(left)
if [ "$L" -gt 80 ]; then
  step "$L" "DeepSeek-V4-Flash, fastest layout, 65k tokens of output room: quality (budget $(( (L - 40) * 60 ))s)"
  EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1" MODE=eval FIRST_ONLY=1 EVAL_CONC=48 EVAL_MAXTOK=65536 \
    EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
    EVAL_BUDGET=$(( (L - 40) * 60 )) bash $B/ksweep.sh $B/lists/ds_long.txt > $R/keval_dslong.log 2>&1
fi
L=$(left)
if [ "$served" = 1 ] && [ "$L" -gt 60 ]; then
  step "$L" "Qwen3.8-Flash-Next: two engines of two cards with expert parallelism, then one TP4 engine"
  EXTRA_ENV="VLLM_QWEN4EXP_PLE_FP8=1" SHAPES=fast bash $B/ksweep.sh <(grep -vE "^#|^$|tp2x2b12x" $B/lists/qwen38fn_c4.txt) > $R/ksweep_qfn_layouts.log 2>&1
fi
L=$(left)
if [ "$L" -gt 35 ] && ls -d $P/minimaxm3_* >/dev/null 2>&1; then
  step "$L" "MiniMax-M3 with CUDA graphs on"
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c2.txt > $R/ksweep_minimax2.log 2>&1
fi
step "$(left)" "CHAINC3 DONE"
kill_all
