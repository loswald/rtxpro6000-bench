#!/usr/bin/env bash
# Original 600 W box: after the GLM throughput arms, quality on whichever layout won. Nish: "BOTH throughput AND
# quality" - a fast kernel that changes the answers is not a result. Waits for glm_perf.sh to finish, replaces
# chain600w6, then: a 403-item quality run on the fastest non-MTP GLM layout (paired against the TP4 baseline on
# identical items), the noise-floor and template arms if time allows, RedHat under W4A4 if time allows.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-05 22:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-7 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 720); do grep -q "GLMPERF DONE" $R/glm_perf.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =q600i 2>/dev/null && step "$(left)" "chain600w6 stopped after the GLM arms; continuing as chain600w7"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

best=$(python3 $B/pick_best.py "$P/glm53f_*s512*" router 1024 mtp 2>$R/glm_best.tps)
case "$best" in
  glm53f_s512)              flags="";;
  glm53f_s512_fib12x)       flags="--moe-backend flashinfer_b12x";;
  glm53f_tp4ep4_s512)       flags="--enable-expert-parallel";;
  glm53f_dp4ep4_s512)       flags="--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel";;
  *)                        flags=""; best="glm53f_s512 (fallback: no arm served)";;
esac
L=$(left)
if [ "$L" -gt 55 ]; then
  step "$L" "GLM quality on the fastest layout: $best ($(cat $R/glm_best.tps 2>/dev/null) out tok/s at router C1024), 512 sequences, budget $(( (L - 15) * 60 ))s"
  echo "$flags" > $R/glm_best.flags
  BEST_FLAGS="$flags --max-num-seqs 512 --max-num-batched-tokens 16384" BEST_BUDGET=$(( (L - 15) * 60 )) ARMS=best bash $B/glm_eval.sh > $R/glm_eval_best.log 2>&1
else
  step "$L" "no time for the quality run on the fastest GLM layout"
fi
L=$(left)
if [ "$L" -gt 35 ]; then
  step "$L" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
  MODE=eval EVAL_BUDGET=$(( (L - 10) * 60 / 2 )) bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
fi
L=$(left)
if [ "$L" -gt 40 ]; then
  step "$L" "RedHat NVFP4 under the W4A4 kernel: quality, then two throughput shapes"
  MODE=eval EVAL_BUDGET=$(( (L - 20) * 60 )) bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/keval_fixb12x.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/ksweep_fixb12x.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
