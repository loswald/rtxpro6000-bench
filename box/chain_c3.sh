#!/usr/bin/env bash
# Third box: the rest of the window (Nish: forty minutes over five hours is fine -> 22:00 UTC), re-cut around
# DeepSeek-V4-Flash, throughput AND quality. Waits for chain_c2's logit ladder, replaces it, then: four DeepSeek
# layout/kernel/budget arms (two shapes each); a 403-item quality run on whichever arm is fastest, paired against the
# TP4 baseline on identical items; MiniMax-M3 if time allows. Qwen3.8-Flash-Next fails on the original box's older
# vLLM build under every kernel and would need this box's newer one; it is not in this window.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-05 22:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC3 (${1:-}min left): ${*:2}"; }
source $B/dlget.sh
tmux has-session -t =dl 2>/dev/null || tmux new-session -d -s dl "bash -c 'source $B/dlget.sh; L=$R/dl_c.log get deepseek-ai/DeepSeek-V4-Flash-0731 DeepSeek-V4-Flash; echo \"[\$(date +%H:%M:%S)] DSDL DONE\" >> $R/dl_c.log'"
for i in $(seq 1 600); do grep -q "KLDIFF-BF16 DONE" $R/kldiff.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =chainc2 2>/dev/null && step "$(left)" "chain_c2 stopped after its logit ladder; continuing as chain_c3"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "DeepSeek-V4-Flash: waiting for the download, applying the sm_120 o_proj fallback"
for i in $(seq 1 60); do grep -q "DSDL DONE" $R/dl_c.log 2>/dev/null && break; sleep 30; done
python3 $B/patch_oproj.py 2>&1 | tail -2 | sed 's/^/  /'
export EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1"
step "$(left)" "DeepSeek-V4-Flash throughput: four layout/kernel/budget arms, two shapes each"
SHAPES=fast bash $B/ksweep.sh $B/lists/ds_perf.txt > $R/ksweep_dsperf.log 2>&1

best=$(python3 $B/pick_best.py "$P/ds4flash_*" router 1024 2>$R/ds_best.tps)
L=$(left)
if [ -n "$best" ] && [ "$L" -gt 50 ]; then
  line=$(grep -E "^${best%%_b12x*}\|" $B/lists/ds_perf.txt | head -1)
  step "$L" "DeepSeek quality on the fastest layout: $best ($(cat $R/ds_best.tps 2>/dev/null) out tok/s at router C1024), budget $(( (L - 15) * 60 ))s"
  printf '%s\n' "$line" > $B/lists/ds_best.txt
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$(( (L - 15) * 60 )) bash $B/ksweep.sh $B/lists/ds_best.txt > $R/keval_dsbest.log 2>&1
else
  step "$L" "no time for the quality run on the fastest DeepSeek layout (best: ${best:-none})"
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
