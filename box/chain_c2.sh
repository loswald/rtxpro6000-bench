#!/usr/bin/env bash
# Third box: the five-hour programme (Nish, 5 Sept 16:40 UTC). Priorities: the ladder's two anchor throughputs at
# 600 W (native BF16 and the QAT build under W4A4 - the Pareto point needs a number that transfers); the logit
# ladder against the BF16 parent (every rung, plus the DFlash2 drafter) - the pairs already measured on the 400 W
# box are not repeated; MiniMax-M3 (index 36, the highest model never measured), quality then throughput, its
# download running from the first minute. A 45-minute soak of the QAT build runs only if time remains. Dropped for
# time: the ladder's quality re-run (measured on the 400 W box; quality does not depend on the power cap), Inkling,
# Qwen3.8-Flash-Next, the gemma thinking pairs, gemma-31B, Ling, Laguna, Hy3, Ornith, MiMo.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models
DEADLINE=$(( $(date +%s) + 4*3600 + 40*60 ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC2 (${1:-}min left): ${*:2}"; }
source $B/hardkill.sh; kill_all >/dev/null 2>&1
tmux kill-session -t =dl 2>/dev/null
tmux new-session -d -s dl "bash -c 'source $B/dlget.sh; L=$R/dl_c.log get olka-fi/MiniMax-M3-MXFP4 MiniMax-M3-MXFP4; get nvidia/MiniMax-M3-DSpark MiniMax-M3-DSpark; echo \"[\$(date +%H:%M:%S)] MMDL DONE\" >> $R/dl_c.log'"

step "$(left)" "ladder anchors at 600 W: native BF16 and QAT under W4A4, throughput"
bash $B/ksweep.sh $B/lists/quant_c.txt > $R/ksweep_quant.log 2>&1
step "$(left)" "logit ladder against the BF16 parent, and the DFlash2 drafter"
LADDER_ONLY=1 POS=16 bash $B/kldiff.sh > $R/kldiff.log 2>&1

step "$(left)" "MiniMax-M3: waiting for the download"
for i in $(seq 1 60); do grep -q "MMDL DONE" $R/dl_c.log 2>/dev/null && break; sleep 30; done
L=$(left); BUD=$(( (L - 70) * 60 )); [ "$BUD" -lt 2400 ] && BUD=2400
step "$L" "MiniMax-M3 quality (budget ${BUD}s), then throughput"
MODE=eval FIRST_ONLY=1 EVAL_BUDGET=$BUD bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/keval_minimax.log 2>&1
bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/ksweep_minimax.log 2>&1

L=$(left)
if [ "$L" -gt 60 ]; then
  step "$L" "soak: the QAT build, four replicas, $(( L - 15 )) min"
  TAG=soak_q27_qat DIR=$MD/Qwen27B-QUASAR-NVFP4 LIN=b12x MINUTES=$(( L - 15 )) bash $B/soak.sh > $R/soak_q27_qat.log 2>&1
else
  step "$L" "no time left for the soak - dropped"
fi
step "$(left)" "CHAINC2 DONE"
kill_all
