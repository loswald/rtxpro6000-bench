#!/usr/bin/env bash
# Original 600 W box: the five-hour programme (Nish, 5 Sept 16:40 UTC: "ALL the high priority work wrapped up in
# 5 hours on these two machines"). Priorities, in order: the two leaderboard rows still missing items to the old
# request timeout; the suite's noise-floor repeat and the gittensor weights under Qwen's own template; RedHat and
# unsloth under the W4A4 kernel (the Pareto question for community four-bit); Qwen3.8-Flash-Next, index 46 with 6B
# active parameters, whose only quality attempt died on the W4A4 kernel; GLM-5.3-Flash at its NATIVE FP8 -
# the fidelity measurement on the best model, downloaded in the background from the first minute. GLM with 65k
# tokens of room runs only if time remains. Dropped for time: the remaining DeepSeek throughput arms, the
# community-build logit pairs, Nemotron-3.5-Lightning and Qwen3.6 quality, DeepSeek at 65k.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models
DEADLINE=$(( $(date +%s) + 4*3600 + 40*60 ))          # leave twenty minutes to pull results and destroy
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-5 (${1:-}min left): ${*:2}"; }
source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "free space for GLM's FP8 release and start its download in the background"
for m in gemma-4-31B-it MiniMax-M3-DSpark Inkling-Small-DSpark Qwen27B-DSpark-NVFP4 Nemotron-3.5-Lightning-DSpark; do
  [ -d $MD/$m ] && { python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" $MD/$m; echo "  freed $m"; }
done
df -h /workspace | tail -1 | awk '{print "  disk: "$4" free"}'
tmux kill-session -t =dl 2>/dev/null
tmux new-session -d -s dl "bash -c 'source $B/dlget.sh; L=$R/dl600w.log get zai-org/GLM-5.3-Flash GLM-5.3-Flash-FP8; echo \"[\$(date +%H:%M:%S)] GLMDL DONE\" >> $R/dl600w.log'"

step "$(left)" "finish the rows that lost items to the request timeout: Qwen FP8 (15), Muse-Glimmer (14)"
MODE=eval EVAL_RESUME=1 EVAL_BUDGET=2400 bash $B/ksweep.sh $B/lists/resume600w.txt > $R/keval_resume.log 2>&1
step "$(left)" "Qwen3.8-Flash-Next (index 46, 6B active): quality, then throughput - two TP2 replicas, auto kernel, BF16 KV"
MODE=eval FIRST_ONLY=1 EVAL_BUDGET=2700 bash $B/ksweep.sh $B/lists/qwen38fn_c.txt > $R/keval_qwen38fn.log 2>&1
FIRST_ONLY=1 bash $B/ksweep.sh $B/lists/qwen38fn_c.txt > $R/ksweep_qwen38fn.log 2>&1
step "$(left)" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
MODE=eval EVAL_BUDGET=2400 bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
step "$(left)" "RedHat NVFP4 under the W4A4 kernel: quality, then throughput"
MODE=eval EVAL_BUDGET=2400 bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/keval_fixb12x.log 2>&1
bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/ksweep_fixb12x.log 2>&1

step "$(left)" "GLM-5.3-Flash at native FP8: waiting for the download"
for i in $(seq 1 120); do grep -q "GLMDL DONE" $R/dl600w.log 2>/dev/null && break; sleep 30; done
if [ -f $MD/GLM-5.3-Flash-FP8/.dl_complete ]; then
  L=$(left); BUD=$(( (L - 45) * 60 )); [ "$BUD" -lt 3000 ] && BUD=3000
  step "$L" "GLM-5.3-Flash native FP8: quality (budget ${BUD}s) then throughput"
  FP8_BUDGET=$BUD ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
else
  step "$(left)" "GLM FP8 download did not complete - skipping the fidelity arm"
fi

L=$(left)
if [ "$L" -gt 75 ]; then
  step "$L" "GLM-5.3-Flash MTP with room to 65k tokens (budget to the deadline)"
  MTP64K_BUDGET=$(( (L - 15) * 60 )) ARMS=mtp64k bash $B/glm_eval.sh > $R/glm_eval_mtp64k.log 2>&1
else
  step "$L" "no time left for the 65k arm - dropped"
fi
step "$(left)" "CHAIN600W5 DONE"
kill_all
