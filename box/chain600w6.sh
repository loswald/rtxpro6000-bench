#!/usr/bin/env bash
# Original 600 W box: the rest of the five-hour window, re-cut at 17:10 UTC around Nish's concern - the throughput
# of GLM-5.3-Flash and DeepSeek-V4-Flash, "the really usable models". Waits for chain600w5's Qwen3.8-Flash-Next
# step to finish (it is in flight), then replaces it: the noise-floor and template arms (30 min), the GLM
# throughput ceiling (1,024 sequences, then MTP), RedHat under W4A4, and GLM at native FP8 only if the deadline
# still allows an hour of it. DeepSeek's layout and sequence-budget arms run on the third box in parallel.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models
DEADLINE=$(( $(date -d "2026-09-05 21:20:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-6 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 480); do grep -q "600W-5.*noise floor" $R/chain600w.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =q600h 2>/dev/null && step "$(left)" "chain600w5 stopped after its Qwen3.8-Flash-Next step; continuing as chain600w6"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
MODE=eval EVAL_BUDGET=2400 bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
step "$(left)" "GLM-5.3-Flash throughput ceiling: 1,024 sequences at C256/C1024, then with the MTP head"
bash $B/glm_perf.sh > $R/glm_perf.log 2>&1
step "$(left)" "RedHat NVFP4 under the W4A4 kernel: quality, then throughput"
MODE=eval EVAL_BUDGET=2400 bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/keval_fixb12x.log 2>&1
bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/ksweep_fixb12x.log 2>&1
L=$(left)
if [ "$L" -gt 75 ] && [ -f $MD/GLM-5.3-Flash-FP8/.dl_complete ]; then
  step "$L" "GLM-5.3-Flash native FP8: quality with the time that is left"
  FP8_BUDGET=$(( (L - 25) * 60 )) ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
else
  step "$L" "GLM native FP8 not run (time); its checkpoint is downloaded for another day"
fi
step "$(left)" "CHAIN600W6 DONE"
kill_all
