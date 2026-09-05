#!/usr/bin/env bash
# Third box, the slack after chain_c8 (which prints CHAINC3 DONE about 01:25 UTC): Qwen3.8-Flash-Next at Qwen's own FP8
# - router shape, probe, and as much of the 403-item run as fits before 02:15 - then CHAINC3 DONE again and kill_all.
R=/workspace/results; B=/workspace/bench; P=$R/probe
HARD=$(( $(date -d "2026-09-06 02:15:00" +%s) ))
left(){ echo $(( (HARD - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC9 (${1:-}min to 02:15): ${*:2}"; }
n0=$(grep -c "CHAINC3 DONE" $R/chain_c.log 2>/dev/null || echo 0)
for i in $(seq 1 1200); do
  n=$(grep -c "CHAINC3 DONE" $R/chain_c.log 2>/dev/null || echo 0); [ "$n" -gt "$n0" ] && break
  sleep 15
done
sleep 5; source $B/hardkill.sh; kill_all >/dev/null 2>&1
grep -q "QFN8DL DONE" $R/dl_c.log 2>/dev/null || { step "$(left)" "FP8 checkpoint not downloaded; nothing to run"; step "$(left)" "CHAINC3 DONE"; exit 0; }
L=$(left)
if [ "$L" -gt 25 ]; then
  step "$L" "Qwen3.8-Flash-Next at Qwen's FP8, TP4: probe and router shape"
  SHAPES=fast bash $B/ksweep.sh $B/lists/qwen38fn_fp8.txt > $R/ksweep_qfn8.log 2>&1
fi
L=$(left)
if [ "$L" -gt 15 ] && ls -d $P/qwen38fn_fp8_* >/dev/null 2>&1; then
  step "$L" "Qwen3.8-Flash-Next FP8 quality, budget $(( (L - 8) * 60 ))s (partial if the window ends first)"
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$(( (L - 8) * 60 )) bash $B/ksweep.sh $B/lists/qwen38fn_fp8.txt > $R/keval_qfn8.log 2>&1
fi
step "$(left)" "CHAINC3 DONE"
kill_all
