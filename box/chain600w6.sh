#!/usr/bin/env bash
# Original 600 W box: the rest of the five-hour window, re-cut at 17:10 UTC around Nish's concern - the throughput
# of GLM-5.3-Flash and DeepSeek-V4-Flash, "the really usable models". Waits for chain600w5's Qwen3.8-Flash-Next
# step to finish (it is in flight), then replaces it: the noise-floor and template arms (30 min), the GLM
# throughput ceiling (five layout/kernel arms at 1,024 sequences, then MTP) FIRST, then the noise-floor and
# template arms, then RedHat under W4A4 if time allows. DeepSeek's layout and kernel arms run on the third box.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models
DEADLINE=$(( $(date -d "2026-09-05 21:20:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-6 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 480); do grep -q "600W-5.*noise floor" $R/chain600w.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =q600h 2>/dev/null && step "$(left)" "chain600w5 stopped after its Qwen3.8-Flash-Next step; continuing as chain600w6"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "GLM-5.3-Flash throughput ceiling: five layout/kernel arms at 1,024 sequences, then MTP"
bash $B/glm_perf.sh > $R/glm_perf.log 2>&1
step "$(left)" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
MODE=eval EVAL_BUDGET=2400 bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
L=$(left)
if [ "$L" -gt 45 ]; then
  step "$L" "RedHat NVFP4 under the W4A4 kernel: quality, then throughput (two shapes)"
  MODE=eval EVAL_BUDGET=1800 bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/keval_fixb12x.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/ksweep_fixb12x.log 2>&1
else
  step "$L" "RedHat under W4A4 not run (time)"
fi
step "$(left)" "GLM native FP8 not run in this window; its checkpoint is downloaded for another day"
step "$(left)" "CHAIN600W6 DONE"
kill_all
