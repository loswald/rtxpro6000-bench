#!/usr/bin/env bash
# Third box, the slack after chain_c8 (replaces chain_c9): Qwen3.8-Flash-Next with 65k tokens of output room first
# (the 32k caps cut 7.7% of its answers), then Qwen's own FP8 checkpoint if time remains; CHAINC3 DONE at the end.
R=/workspace/results; B=/workspace/bench; P=$R/probe
HARD=$(( $(date -d "2026-09-06 02:15:00" +%s) ))
left(){ echo $(( (HARD - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC10 (${1:-}min to 02:15): ${*:2}"; }
n0=$(grep -c "CHAINC3 DONE" $R/chain_c.log 2>/dev/null || echo 0)
for i in $(seq 1 1200); do
  n=$(grep -c "CHAINC3 DONE" $R/chain_c.log 2>/dev/null || echo 0); [ "$n" -gt "$n0" ] && break
  sleep 15
done
sleep 5; source $B/hardkill.sh; kill_all >/dev/null 2>&1
export EXTRA_ENV="VLLM_QWEN4EXP_PLE_FP8=1"
L=$(left)
if [ "$L" -gt 20 ]; then
  step "$L" "Qwen3.8-Flash-Next with 65k tokens of output room: quality (budget $(( (L - 8) * 60 ))s, partial if the window ends first)"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_MAXTOK=65536 EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
    EVAL_BUDGET=$(( (L - 8) * 60 )) bash $B/ksweep.sh $B/lists/qwen38fn_long.txt > $R/keval_qfnlong.log 2>&1
fi
unset EXTRA_ENV
L=$(left)
if [ "$L" -gt 30 ] && grep -q "QFN8DL DONE" $R/dl_c.log 2>/dev/null; then
  step "$L" "Qwen3.8-Flash-Next at Qwen's FP8, TP4: probe and router shape"
  SHAPES=fast bash $B/ksweep.sh $B/lists/qwen38fn_fp8.txt > $R/ksweep_qfn8.log 2>&1
fi
step "$(left)" "CHAINC3 DONE"
kill_all
