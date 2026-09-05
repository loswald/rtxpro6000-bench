#!/usr/bin/env bash
# Original 600 W box: replaces chain600w7. After the first GLM sweep, run the two layouts that remove the per-layer
# all-reduce at the sequence budgets the vendor build allows (glm_perf2.sh), then a 403-item quality run on the
# fastest non-MTP layout of either sweep, then the noise-floor/template arms and RedHat under W4A4 if time allows.
# Prints CHAIN600W7 DONE at the end so the finish-watcher and the hourly check need no change.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-05 22:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-8 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 720); do grep -q "GLMPERF DONE" $R/glm_perf.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =q600i 2>/dev/null && step "$(left)" "chain600w6 stopped after the GLM arms; continuing as chain600w8"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "GLM-5.3-Flash: DP4+EP at 192 sequences and DP2xTP2+EP at 384 - the layouts without the all-reduce"
bash $B/glm_perf2.sh > $R/glm_perf2.log 2>&1

best=$(python3 $B/pick_best.py "$P/glm53f_*" router 1024 mtp 2>$R/glm_best.tps)
case "$best" in
  glm53f_s512)              flags="--max-num-seqs 512";;
  glm53f_tp4ep4_s512)       flags="--enable-expert-parallel --max-num-seqs 512";;
  glm53f_dp4ep4_s192)       flags="--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-num-seqs 192";;
  glm53f_dp2tp2ep2_s384)    flags="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-num-seqs 384";;
  *)                        flags="--max-num-seqs 512"; best="glm53f_s512 (fallback: ${best:-none})";;
esac
L=$(left)
if [ "$L" -gt 55 ]; then
  step "$L" "GLM quality on the fastest layout: $best ($(cat $R/glm_best.tps 2>/dev/null) out tok/s at router C1024), budget $(( (L - 15) * 60 ))s"
  echo "$flags" > $R/glm_best.flags
  BEST_FLAGS="$flags --max-num-batched-tokens 16384" BEST_BUDGET=$(( (L - 15) * 60 )) ARMS=best bash $B/glm_eval.sh > $R/glm_eval_best.log 2>&1
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
